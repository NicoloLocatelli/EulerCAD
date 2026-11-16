from collections import defaultdict


def _walk(graph, lines, visited, start_p, start_line):
    chain = [start_p]
    cur, prev_p = start_line, start_p
    while True:
        visited[cur] = True
        ln = lines[cur]
        next_p = (
            (ln["end_x"], ln["end_y"]) if (ln["start_x"], ln["start_y"]) == prev_p else (ln["start_x"], ln["start_y"])
        )
        chain.append(next_p)
        cands = [j for j in graph[next_p] if not visited[j]]
        if len(graph[next_p]) != 2 or len(cands) != 1:
            break
        prev_p, cur = next_p, cands[0]
    return chain


def walk_chains(lines: list[dict]) -> list[list[tuple[float, float]]]:
    """Group consecutive Line rows into polyline chains.

    Boundaries = degree-1 / degree-≥3 endpoints. Pure-degree-2 loops
    (closed curves) handled separately.

    Returns:
        a list of chains, where each chain is a list of (x, y) points
        representing a connected sequence of lines.
    """
    graph: dict[tuple[float, float], list[int]] = defaultdict(list)
    for i, ln in enumerate(lines):
        graph[ln["start_x"], ln["start_y"]].append(i)
        graph[ln["end_x"], ln["end_y"]].append(i)

    visited = [False] * len(lines)
    chains: list[list[tuple[float, float]]] = []

    for p, incident in graph.items():
        if len(incident) == 2:
            continue
        for i in incident:
            if visited[i]:
                continue
            chains.append(_walk(graph, lines, visited, p, i))

    for i, v in enumerate(visited):
        if v:
            continue
        start_p = (lines[i]["start_x"], lines[i]["start_y"])
        chains.append(_walk(graph, lines, visited, start_p, i))

    return chains
