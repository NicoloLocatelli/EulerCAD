"""STEP -> 2D visible-line projection via OpenCASCADE hidden-line removal (HLR)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np

# cadquery-ocp is a compiled pybind11 extension with no .pyi stubs, so ty
# can't resolve these members even though they exist at runtime.
from OCP.BRepAdaptor import BRepAdaptor_Curve  # ty: ignore[unresolved-import]
from OCP.BRepMesh import BRepMesh_IncrementalMesh  # ty: ignore[unresolved-import]
from OCP.GCPnts import GCPnts_QuasiUniformDeflection  # ty: ignore[unresolved-import]
from OCP.GeomAbs import GeomAbs_Line  # ty: ignore[unresolved-import]
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt  # ty: ignore[unresolved-import]
from OCP.HLRAlgo import HLRAlgo_Projector  # ty: ignore[unresolved-import]
from OCP.HLRBRep import (
    HLRBRep_Algo,  # ty: ignore[unresolved-import]
    HLRBRep_HLRToShape,  # ty: ignore[unresolved-import]
    HLRBRep_PolyAlgo,  # ty: ignore[unresolved-import]
    HLRBRep_PolyHLRToShape,  # ty: ignore[unresolved-import]
)
from OCP.ShapeFix import ShapeFix_Shape  # ty: ignore[unresolved-import]
from OCP.STEPControl import STEPControl_Reader  # ty: ignore[unresolved-import]
from OCP.TopAbs import TopAbs_EDGE  # ty: ignore[unresolved-import]
from OCP.TopExp import TopExp_Explorer  # ty: ignore[unresolved-import]
from OCP.TopoDS import TopoDS

_IFSELECT_RET_DONE = 1

log = logging.getLogger(__name__)

# Last load/projection call entered by this process. A call that hangs never
# raises anything itself - reading this after killing the process pinpoints
# the exact stage it was stuck at, without attaching a profiler.
_last_stage = ""

# Optional sink for stage updates, in addition to _last_stage. A caller
# running this code in a worker process it spawned (see step_to_views.py)
# can't call current_stage() on that process directly - a genuinely stuck
# native OCC call never yields back to Python, so nothing short of an
# OS-level kill of the whole process can recover from it, and the process
# is gone once killed. Registering a hook that writes into e.g. shared
# memory lets the parent read the last stage *before* the hang, from
# outside the stuck process.
_stage_hook: Callable[[str], None] | None = None


def current_stage() -> str:
    """Most recent load/projection stage entered by this process.

    Returns:
        Human-readable stage label, or "" if nothing has run yet.
    """
    return _last_stage


def set_stage_hook(hook: Callable[[str], None] | None) -> None:
    """Register a callback invoked with every stage label, alongside current_stage().

    Args:
        hook: called synchronously with the stage string before each risky
            OCC call; must not raise or block. Pass None to clear it.
    """
    global _stage_hook
    _stage_hook = hook


def _set_stage(stage: str) -> None:
    global _last_stage
    _last_stage = stage
    log.debug("stage: %s", stage)
    if _stage_hook is not None:
        _stage_hook(stage)


def load_step(path: Path):
    _set_stage(f"load_step: reading {path.name}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != _IFSELECT_RET_DONE:
        raise RuntimeError(f"failed to read STEP file: {path}")

    _set_stage(f"load_step: transferring roots for {path.name}")
    reader.TransferRoots()

    _set_stage(f"load_step: extracting shape for {path.name}")
    shape = reader.OneShape()

    # Real-world STEP corpora carry a fair share of degenerate geometry (surfaces
    # with out-of-range parametric bounds, tiny/self-intersecting wires) that
    # parses fine but segfaults OCC's classic HLR algorithm later. ShapeFix_Shape
    # is OCC's standard repair pass for exactly this class of issue - cheap, and
    # a no-op on already-valid shapes.
    _set_stage(f"load_step: ShapeFix_Shape.Perform for {path.name}")
    fixer = ShapeFix_Shape(shape)
    fixer.Perform()

    _set_stage(f"load_step: ShapeFix_Shape.Shape() for {path.name}")
    return fixer.Shape()


def _ax2(direction: tuple[float, float, float], up: tuple[float, float, float]) -> gp_Ax2:
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    u = np.array(up, dtype=float)
    x_dir = np.cross(u, d)
    x_norm = np.linalg.norm(x_dir)
    if x_norm < 1e-8:
        # up parallel (or anti-parallel) to direction: cross product degenerates to
        # ~0, so x_dir/=x_norm would divide by ~0 and feed NaN into gp_Dir - which
        # OCC turns into an opaque native construction error far from the real
        # cause. Every current caller (VIEW_PRESETS, random_views) avoids this by
        # construction, but nothing enforces it for future callers.
        raise ValueError(f"direction {direction} and up {up} are parallel; cannot build a projection frame")
    x_dir /= x_norm
    return gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*d), gp_Dir(*x_dir))


def _edge_segments(edge, deflection: float) -> list[tuple[float, float, float, float]]:
    curve = BRepAdaptor_Curve(edge)
    if curve.GetType() == GeomAbs_Line:
        # A straight edge has zero deflection error at any subdivision - the
        # quasi-uniform sampler would still chop long ones into multiple
        # collinear points, which weld_collinear only has to merge back
        # together downstream. Emit the single true segment instead.
        p1 = curve.Value(curve.FirstParameter())
        p2 = curve.Value(curve.LastParameter())
        return [(p1.X(), p1.Y(), p2.X(), p2.Y())]

    sampler = GCPnts_QuasiUniformDeflection(curve, deflection)
    if not sampler.IsDone() or sampler.NbPoints() < 2:
        log.debug("edge tessellation failed or produced <2 points (deflection=%s)", deflection)
        return []
    pts = [sampler.Value(i) for i in range(1, sampler.NbPoints() + 1)]
    return [(a.X(), a.Y(), b.X(), b.Y()) for a, b in pairwise(pts)]


def model_diagonal(shape) -> float:
    """3D bounding-box diagonal of a loaded shape, in model units.

    This is what makes a tessellation tolerance relative: dividing a fixed fraction
    into it gives every model the same approximation quality regardless of how
    physically large it is. Callers decide what a degenerate shape means to them,
    so this reports 0.0 rather than raising.

    Returns:
        The diagonal length, or 0.0 if the shape has no finite extent.
    """
    from OCP.Bnd import Bnd_Box  # ty: ignore[unresolved-import]
    from OCP.BRepBndLib import BRepBndLib  # ty: ignore[unresolved-import]

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
    return diag if np.isfinite(diag) and diag > 0 else 0.0


def face_count(shape) -> int:
    """Number of faces in a shape.

    The cheapest useful predictor of triangulation cost - roughly 0.3 ms against a
    ~46 ms load - so it is what callers use to refuse models that would blow up a
    fine mesh.

    Returns:
        The face count.
    """
    from OCP.TopAbs import TopAbs_FACE  # ty: ignore[unresolved-import]
    from OCP.TopExp import TopExp_Explorer  # ty: ignore[unresolved-import]

    n = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        n += 1
        explorer.Next()
    return n


def _explode_visible(to_shape, stage: str, label: str) -> list:
    """Collect edges from the three compounds OCCT considers visible.

    Shared by both hidden-line-removal algorithms so that "visible" has exactly one
    definition: sharp edges, smooth silhouettes, and outlines. Both
    `HLRBRep_HLRToShape` and `HLRBRep_PolyHLRToShape` expose this same trio.

    Returns:
        The visible `TopoDS_Edge` handles.
    """
    edges = []
    for i, compound in enumerate([to_shape.VCompound(), to_shape.Rg1LineVCompound(), to_shape.OutLineVCompound()]):
        if compound.IsNull():
            log.debug("%s: compound %d is null (%s)", stage, i, label)
            continue
        _set_stage(f"{stage}: exploring compound {i} ({label})")
        explorer = TopExp_Explorer(compound, TopAbs_EDGE)
        while explorer.More():
            edges.append(TopoDS.Edge_s(explorer.Current()))
            explorer.Next()
    return edges


def _tessellate(edges, deflection: float) -> list[tuple[float, float, float, float]]:
    """Sample every edge into straight segments at one tolerance.

    Returns:
        The 2D segments, concatenated across edges.
    """
    segments: list[tuple[float, float, float, float]] = []
    for edge in edges:
        segments.extend(_edge_segments(edge, deflection))
    return segments


def _visible_edges(
    shape,
    direction: tuple[float, float, float],
    up: tuple[float, float, float],
    label: str = "",
) -> list:
    """Run hidden-line removal once and collect the visible edges.

    This is the expensive half of a projection and it does not depend on the
    tessellation tolerance at all - measured on real models, HLR takes the same
    time whether the curves are later sampled coarsely or finely. Splitting it out
    lets a caller pay for it once and tessellate the same edges at several
    tolerances (see `project_step_to_lines_multi`).

    Returns:
        The visible `TopoDS_Edge` handles, still in 3D-curve form.
    """
    _set_stage(f"project: building HLR projector ({label})")
    projector = HLRAlgo_Projector(_ax2(direction, up))
    algo = HLRBRep_Algo()

    _set_stage(f"project: algo.Add ({label})")
    algo.Add(shape)

    _set_stage(f"project: algo.Projector ({label})")
    algo.Projector(projector)

    _set_stage(f"project: algo.Update ({label})")
    algo.Update()

    _set_stage(f"project: algo.Hide ({label})")
    algo.Hide()

    _set_stage(f"project: HLRBRep_HLRToShape ({label})")
    return _explode_visible(HLRBRep_HLRToShape(algo), "project", label)


def project_step_to_lines(
    shape,
    direction: tuple[float, float, float],
    up: tuple[float, float, float],
    deflection: float,
    label: str = "",
) -> list[tuple[float, float, float, float]]:
    """HLR-project a shape to its visible 2D edges, as (x1, y1, x2, y2) segments.

    Only visible edges are kept (sharp + smooth-silhouette + outline); hidden
    lines are dropped so the result reads as a clean sketch, not a full
    engineering drawing.

    Args:
        shape: the loaded STEP shape.
        direction: viewing direction.
        up: up vector; must not be parallel to `direction`.
        deflection: curve -> polyline tessellation tolerance, in model units
            (STEP files are typically in mm; tune per dataset scale)
        label: identifies this call in `current_stage()` (e.g. "model/view");
            purely diagnostic, no effect on the result.

    Returns:
        a list of 2D segments in the projection plane, in model units.
    """
    return _tessellate(_visible_edges(shape, direction, up, label), deflection)


def project_step_to_lines_poly(
    shape,
    direction: tuple[float, float, float],
    up: tuple[float, float, float],
    deflection: float,
    mesh_deflection: float | None = None,
    angular_deflection: float = 0.5,
    relative_mesh: bool = False,
    label: str = "",
) -> list[tuple[float, float, float, float]]:
    """HLR-project via OpenCASCADE's *polygonal* algorithm instead of the exact one.

    `project_step_to_lines` runs HLRBRep_Algo, which computes silhouettes and edge
    intersections analytically on the exact surfaces. This runs HLRBRep_PolyAlgo,
    which first triangulates the shape and then resolves visibility on the mesh:
    far cheaper, because edge-versus-triangle tests replace curve-curve
    intersection, but the result is only as good as the triangulation.

    Both are object-space algorithms returning vector segments - "polygonal" refers
    to the input being a mesh, not to the output being pixels.

    Two consequences worth knowing:

    - `mesh_deflection`, not `deflection`, is the real fidelity knob here. The edges
      handed back are already polylines, so `deflection` only sub-samples what the
      mesher produced. It defaults to `deflection`.
    - Silhouettes are faceted. A smooth cylinder's outline follows mesh edges rather
      than a true smooth curve, so it zig-zags at the triangulation scale.

    Note this **mutates `shape`**, storing the triangulation inside it. Re-meshing
    the same shape at a coarser tolerance may keep the finer existing mesh; reload
    the shape between tolerances if that matters.

    Args:
        shape: the loaded STEP shape. Triangulated in place.
        direction: viewing direction.
        up: up vector; must not be parallel to `direction`.
        deflection: curve -> polyline tolerance applied to the returned edges.
        mesh_deflection: triangulation tolerance. In model units when
            `relative_mesh` is False; a dimensionless fraction of each edge's own
            size when it is True. Defaults to `deflection`.
        angular_deflection: triangulation angular tolerance, in radians. Being an
            angle it is inherently scale-free, so it imposes a minimum number of
            chords per curve regardless of model size - which can matter more than
            `mesh_deflection` on small parts.
        relative_mesh: interpret `mesh_deflection` per edge rather than in model
            units (OpenCASCADE's `isRelative` flag).
        label: diagnostic only, as in `project_step_to_lines`.

    Returns:
        a list of 2D segments in the projection plane, in model units.
    """
    _set_stage(f"project_poly: meshing ({label})")
    BRepMesh_IncrementalMesh(
        shape,
        mesh_deflection if mesh_deflection is not None else deflection,
        relative_mesh,
        angular_deflection,
        True,
    )

    _set_stage(f"project_poly: PolyAlgo load/project ({label})")
    algo = HLRBRep_PolyAlgo()
    algo.Load(shape)
    algo.Projector(HLRAlgo_Projector(_ax2(direction, up)))

    _set_stage(f"project_poly: algo.Update ({label})")
    algo.Update()

    _set_stage(f"project_poly: PolyHLRToShape ({label})")
    to_shape = HLRBRep_PolyHLRToShape()
    to_shape.Update(algo)
    return _tessellate(_explode_visible(to_shape, "project_poly", label), deflection)


def project_step_to_lines_multi(
    shape,
    direction: tuple[float, float, float],
    up: tuple[float, float, float],
    deflections: list[float],
    label: str = "",
) -> dict[float, list[tuple[float, float, float, float]]]:
    """Project once, tessellate at several tolerances.

    Equivalent to calling `project_step_to_lines` per tolerance, but runs the
    hidden-line removal a single time. Since HLR dominates the cost and is
    tolerance-independent, sweeping N tolerances this way costs roughly one
    projection rather than N.

    Args:
        shape: the loaded STEP shape.
        direction: viewing direction.
        up: up vector; must not be parallel to `direction`.
        deflections: tessellation tolerances, in model units.
        label: diagnostic only, as in `project_step_to_lines`.

    Returns:
        Mapping from each requested tolerance to its 2D segments.
    """
    edges = _visible_edges(shape, direction, up, label)
    return {deflection: _tessellate(edges, deflection) for deflection in deflections}
