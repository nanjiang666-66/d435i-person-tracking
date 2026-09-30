"""Estimate a tracked person's range from colour-aligned depth pixels."""

import numpy as np


def distance_in_box(depth, encoding, box, depth_scale, reference_depth=None):
    """Return a supported torso range in metres, or None if unavailable.

    The lower middle of a person box is more likely to contain the torso when
    someone leans or tilts their head. A nearby depth group must occupy at
    least 20% of valid samples, so a few foreground outliers are ignored while
    the background cannot take over the median merely by being larger. Once a
    person is locked, prefer the group near the last reliable torso range;
    a closer occluder must not silently replace it.
    """
    x1, y1, x2, y2 = (int(value) for value in box)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None

    left = max(0, x1 + int(width * 0.25))
    right = min(depth.shape[1], x1 + int(width * 0.75))
    top = max(0, y1 + int(height * 0.4))
    bottom = min(depth.shape[0], y1 + int(height * 0.85))
    if left >= right or top >= bottom:
        return None

    pixels = depth[top:bottom, left:right].astype(np.float32)
    if encoding == "16UC1":
        pixels *= depth_scale
    elif encoding != "32FC1":
        return None

    valid = pixels[np.isfinite(pixels) & (pixels >= 0.2) & (pixels <= 8.0)]
    if valid.size < 20:
        return None

    # Find well-supported 0.25 m depth bands, then collapse each contiguous
    # run of supported bins into a candidate depth group.
    counts, edges = np.histogram(valid, bins=np.arange(0.2, 8.251, 0.05))
    support = np.convolve(counts, np.ones(5, dtype=np.int32), mode="same")
    threshold = max(20, int(np.ceil(valid.size * 0.2)))
    candidates = np.flatnonzero(support >= threshold)
    if candidates.size == 0:
        return None

    groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1)
    depths = []
    for group in groups:
        peak = int(group[np.argmax(counts[group])])
        centre = (edges[peak] + edges[peak + 1]) / 2.0
        cluster = valid[(valid >= centre - 0.125) & (valid <= centre + 0.125)]
        if cluster.size >= threshold:
            depths.append(float(np.median(cluster)))
    if not depths:
        return None

    if reference_depth is None:
        return min(depths)
    closest = min(depths, key=lambda value: abs(value - reference_depth))
    return closest if abs(closest - reference_depth) <= 0.30 else None
