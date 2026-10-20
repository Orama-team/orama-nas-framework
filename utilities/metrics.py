from __future__ import annotations

from math import sqrt
from typing import Iterable

Point2D = tuple[float, float]


def _to_minimization(point: Point2D, directions: tuple[int, int]) -> Point2D:
    # direction +1 means maximize, -1 means minimize
    return (-directions[0] * point[0], -directions[1] * point[1])


def _dominates(a: Point2D, b: Point2D, directions: tuple[int, int]) -> bool:
    ax, ay = a
    bx, by = b

    if directions[0] == 1:
        no_worse_x = ax >= bx
        better_x = ax > bx
    else:
        no_worse_x = ax <= bx
        better_x = ax < bx

    if directions[1] == 1:
        no_worse_y = ay >= by
        better_y = ay > by
    else:
        no_worse_y = ay <= by
        better_y = ay < by

    return no_worse_x and no_worse_y and (better_x or better_y)


def non_dominated(points: Iterable[Point2D], directions: tuple[int, int]) -> list[Point2D]:
    pts = [p for p in points if not _is_nan_point(p)]
    front: list[Point2D] = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if _dominates(q, p, directions):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def _is_nan_point(p: Point2D) -> bool:
    return p[0] != p[0] or p[1] != p[1]


def hypervolume_2d(
    front: Iterable[Point2D],
    directions: tuple[int, int],
    reference_point: Point2D,
) -> float:
    pts = [_to_minimization(p, directions) for p in front if not _is_nan_point(p)]
    if not pts:
        return 0.0

    ref = _to_minimization(reference_point, directions)
    nd = non_dominated([(x, y) for x, y in pts], (-1, -1))
    nd_sorted = sorted(nd, key=lambda p: (p[0], p[1]))

    hv = 0.0
    current_y = ref[1]
    for x, y in nd_sorted:
        width = ref[0] - x
        height = current_y - y
        if width > 0.0 and height > 0.0:
            hv += width * height
        if y < current_y:
            current_y = y
    return hv


def normalized_hypervolume_2d(
    front: Iterable[Point2D],
    directions: tuple[int, int],
    reference_point: Point2D,
    normalization_front: Iterable[Point2D],
) -> float:
    """
    Calculate normalized 2D hypervolume.

    Divides the front's hypervolume by the hypervolume of a reference front,
    using the same reference point for both.
    
    Args:
        front: Pareto front points
        directions: Optimization directions (1 for maximize, -1 for minimize)
        reference_point: Reference point for HV calculation
        normalization_front: Reference Pareto front used as the normalization baseline
    
    Returns:
        Hypervolume ratio relative to the normalization front
    """
    raw_hv = hypervolume_2d(front, directions, reference_point)

    normalization_hv = hypervolume_2d(
        normalization_front, directions, reference_point
    )
    if normalization_hv <= 0:
        return 0.0

    return raw_hv / normalization_hv


def igd_plus(
    front: Iterable[Point2D],
    reference_front: Iterable[Point2D],
    directions: tuple[int, int],
) -> float:
    approx = [_to_minimization(p, directions) for p in front if not _is_nan_point(p)]
    ref = [_to_minimization(p, directions) for p in reference_front if not _is_nan_point(p)]

    if not ref:
        return 0.0
    if not approx:
        return float("inf")

    total = 0.0
    for r in ref:
        best = float("inf")
        for a in approx:
            dx = max(a[0] - r[0], 0.0)
            dy = max(a[1] - r[1], 0.0)
            dist = sqrt(dx * dx + dy * dy)
            if dist < best:
                best = dist
        total += best
    return total / len(ref)


def c_metric(
    front_a: Iterable[Point2D],
    front_b: Iterable[Point2D],
    directions: tuple[int, int],
) -> float:
    a = [p for p in front_a if not _is_nan_point(p)]
    b = [p for p in front_b if not _is_nan_point(p)]
    if not b:
        return 0.0
    if not a:
        return 0.0

    dominated_count = 0
    for pb in b:
        if any(_dominates(pa, pb, directions) for pa in a):
            dominated_count += 1
    return dominated_count / len(b)
