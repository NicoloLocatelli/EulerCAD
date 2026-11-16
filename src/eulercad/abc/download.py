import argparse
import logging
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import py7zr
from tqdm import tqdm

from eulercad.paths import get_absolute_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# format -> (index URL, output dir for extracted files)
FORMATS = {
    "step": ("https://deep-geometry.github.io/abc-dataset/data/step_v00.txt", get_absolute_path("data/ABC/step")),
    "feat": ("https://deep-geometry.github.io/abc-dataset/data/feat_v00.txt", get_absolute_path("data/ABC/features")),
    "stat": ("https://deep-geometry.github.io/abc-dataset/data/stat_v00.txt", get_absolute_path("data/ABC/stats")),
}
RAW_DIR = get_absolute_path("data/ABC/raw")


def _open_url(url: str, timeout: float):
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"refusing to open non-http(s) URL: {url}")
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 -- scheme checked above  # nosec B310


def parse_chunk_spec(spec: str) -> set[int]:
    chunks: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            chunks.update(range(int(lo), int(hi) + 1))
        else:
            chunks.add(int(part))
    return chunks


def fetch_index(fmt: str) -> list[tuple[str, str]]:
    index_url, _ = FORMATS[fmt]
    index_cache = get_absolute_path(f"data/ABC/{fmt}_v00.txt")
    if not index_cache.exists():
        index_cache.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading chunk index: %s", index_url)
        with _open_url(index_url, timeout=30) as resp:
            index_cache.write_bytes(resp.read())
    lines = index_cache.read_text().splitlines()
    pairs = []
    for line in lines:
        if not line.strip():
            continue
        url, filename = line.split()
        pairs.append((url, filename))
    return pairs


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        log.info("Already downloaded: %s", dest.name)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _open_url(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def extract_chunk(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract ABC dataset chunks.")
    parser.add_argument("--chunks", required=True, help="chunk indices to fetch, e.g. '0-4,7,10'")
    parser.add_argument(
        "--formats",
        default="step,feat,stat",
        help="comma-separated formats to fetch: step (BRep geometry), feat (per-face/edge surface types), "
        "stat (per-model summary stats)",
    )
    parser.add_argument("--keep-archives", action="store_true", help="keep downloaded .7z files after extraction")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    wanted = parse_chunk_spec(args.chunks)
    formats = [f.strip() for f in args.formats.split(",")]
    for fmt in formats:
        if fmt not in FORMATS:
            raise ValueError(f"unknown format: {fmt!r}, must be one of {sorted(FORMATS)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        index = fetch_index(fmt)
        if max(wanted) >= len(index):
            raise ValueError(f"chunk index out of range for format {fmt!r}: max available is {len(index) - 1}")

        out_dir = FORMATS[fmt][1]
        for i in sorted(wanted):
            url, filename = index[i]
            archive = RAW_DIR / filename
            log.info("Chunk %d [%s]: %s", i, fmt, filename)
            download_file(url, archive)
            extract_chunk(archive, out_dir)
            if not args.keep_archives:
                archive.unlink()

    log.info("Done. Formats %s extracted under %s", formats, RAW_DIR.parent)


if __name__ == "__main__":
    main()
