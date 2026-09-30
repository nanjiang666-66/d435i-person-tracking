"""Conservative, short-term recovery when a selected tracker ID changes.

This compares appearance, position and depth. It is a heuristic, not person
identification; ambiguous candidates deliberately remain unselected.
"""

import numpy as np


def appearance_signature(frame, box):
    """Compact colour description of the centre of a detected person."""
    x1, y1, x2, y2 = (int(value) for value in box)
    width, height = x2 - x1, y2 - y1
    left = max(0, x1 + int(width * 0.25))
    right = min(frame.shape[1], x1 + int(width * 0.75))
    top = max(0, y1 + int(height * 0.15))
    bottom = min(frame.shape[0], y1 + int(height * 0.8))
    if right - left < 8 or bottom - top < 12:
        return None

    region = frame[top:bottom:2, left:right:2]
    halves = np.array_split(region, 2, axis=0)
    parts = []
    for half in halves:
        if half.size == 0:
            return None
        for channel in range(3):
            hist, _ = np.histogram(half[:, :, channel], bins=16, range=(0, 256))
            hist = hist.astype(np.float32)
            parts.append(hist / max(float(hist.sum()), 1.0))
    return np.concatenate(parts)


def appearance_similarity(first, second):
    """Mean Bhattacharyya coefficient of six normalised colour histograms."""
    if first is None or second is None:
        return 0.0
    return float(np.sqrt(first * second).reshape(6, 16).sum(axis=1).mean())


class TargetRecovery:
    def __init__(self, max_gap=3.0, min_gap=0.2, stable_frames=3):
        self.max_gap = max_gap
        self.min_gap = min_gap
        self.stable_frames = stable_frames
        self.reset()

    def reset(self):
        self.signature = None
        self.last_box = None
        self.last_seen = None
        self.last_depth = None
        self.last_depth_time = None
        self.other_ids = {}
        self.candidate_counts = {}

    def depth_is_plausible(self, depth, now):
        if depth is None:
            return False
        if self.last_depth is None:
            return True
        elapsed = max(0.0, now - self.last_depth_time)
        return abs(depth - self.last_depth) <= min(0.30, 0.15 + 1.5 * elapsed)

    def record_selected(self, frame, box, depth, other_ids, now):
        self.last_seen = now
        self.last_box = tuple(float(value) for value in box)
        self.candidate_counts.clear()
        self.other_ids = {
            track_id: seen_at for track_id, seen_at in self.other_ids.items()
            if now - seen_at <= self.max_gap + 1.0
        }
        for track_id in other_ids:
            self.other_ids[track_id] = now
        if depth is None:
            return
        current = appearance_signature(frame, box)
        if current is not None and (
            self.signature is None
            or appearance_similarity(self.signature, current) >= 0.78
        ):
            self.signature = (
                current if self.signature is None
                else 0.85 * self.signature + 0.15 * current
            )
        self.last_depth = depth
        self.last_depth_time = now

    def find_match(self, frame, candidates, now):
        """Return (new ID, score) only for a unique, stable, plausible match.

        candidates contains (track_id, box, depth_m) entries from this frame.
        """
        if self.last_seen is None or self.signature is None or self.last_depth is None:
            return None
        gap = now - self.last_seen
        if gap > self.max_gap or gap < 0.0:
            return None

        self.candidate_counts = {
            track_id: self.candidate_counts.get(track_id, 0) + 1
            for track_id, _box, _depth in candidates
        }
        if gap < self.min_gap:
            return None

        height, width = frame.shape[:2]
        old_x = (self.last_box[0] + self.last_box[2]) / 2.0
        old_y = (self.last_box[1] + self.last_box[3]) / 2.0
        max_dx = width * (0.22 + 0.09 * gap)
        max_dy = height * (0.22 + 0.06 * gap)
        scores = []
        for track_id, box, depth in candidates:
            if depth is None:
                continue
            new_x = (box[0] + box[2]) / 2.0
            new_y = (box[1] + box[3]) / 2.0
            dx = abs(new_x - old_x)
            dy = abs(new_y - old_y)
            depth_change = abs(depth - self.last_depth)
            max_depth_change = max(0.55, 0.3 * self.last_depth)
            if dx > max_dx or dy > max_dy or depth_change > max_depth_change:
                continue
            similarity = appearance_similarity(
                self.signature, appearance_signature(frame, box)
            )
            if similarity < 0.72:
                continue
            position_score = 1.0 - (dx / max_dx + dy / max_dy) / 2.0
            depth_score = 1.0 - depth_change / max_depth_change
            score = 0.6 * similarity + 0.2 * position_score + 0.2 * depth_score
            scores.append((score, track_id, similarity))

        if not scores:
            return None
        scores.sort(reverse=True)
        best_score, best_id, _similarity = scores[0]
        if best_id in self.other_ids or self.candidate_counts[best_id] < self.stable_frames:
            return None
        if best_score < 0.75:
            return None
        if len(scores) > 1 and best_score - scores[1][0] < 0.12:
            return None
        return best_id, best_score
