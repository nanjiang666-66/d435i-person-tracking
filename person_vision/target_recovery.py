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


def smaller_box_overlap(first, second):
    """Fraction of the smaller box shared by two detections."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 0.0 else 0.0


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
        self.candidate_depth_runs = {}
        self.pending_depth = None
        self.pending_depth_count = 0
        self.depth_recovery_diagnostics = None

    def relock_depth(self, frame, box, depth, confidence, now):
        """Accept a farther range only after the same visible person supports it."""
        similarity = appearance_similarity(
            self.signature, appearance_signature(frame, box)
        )
        elapsed = (
            max(0.0, now - self.last_depth_time)
            if self.last_depth_time is not None else 0.0
        )
        plausible_change = min(1.0, max(0.35, elapsed))
        # Report the first failed gate without changing the acceptance rules.
        # These values distinguish appearance rejection from a depth problem.
        reason = None
        if depth is None:
            reason = "no_supported_depth"
        elif self.last_depth is None:
            reason = "no_reference_depth"
        elif confidence < 0.75:
            reason = "low_detection_confidence"
        elif similarity < 0.82:
            reason = "appearance_mismatch"
        elif depth - self.last_depth <= 0.30:
            reason = "not_farther_than_reference"
        elif depth - self.last_depth > plausible_change:
            reason = "beyond_range_limit"
        self.depth_recovery_diagnostics = {
            "reason": reason or "waiting_stable_depth",
            "candidate": depth,
            "reference": self.last_depth,
            "confidence": confidence,
            "appearance": similarity,
            "range_limit": plausible_change,
            "stable_frames": 0,
        }
        if reason is not None:
            self.pending_depth = None
            self.pending_depth_count = 0
            return None

        if self.pending_depth is None or abs(depth - self.pending_depth) > 0.08:
            self.pending_depth = depth
            self.pending_depth_count = 1
        else:
            self.pending_depth = 0.8 * self.pending_depth + 0.2 * depth
            self.pending_depth_count += 1
        self.depth_recovery_diagnostics["stable_frames"] = self.pending_depth_count
        if self.pending_depth_count < 5:
            return None
        self.depth_recovery_diagnostics["reason"] = "accepted"
        confirmed = self.pending_depth
        self.pending_depth = None
        self.pending_depth_count = 0
        return confirmed

    def depth_is_plausible(self, depth, now):
        if depth is None:
            return False
        if self.last_depth is None:
            return True
        elapsed = max(0.0, now - self.last_depth_time)
        return abs(depth - self.last_depth) <= min(0.30, 0.15 + 1.5 * elapsed)

    def record_selected(self, frame, box, depth, other_tracks, now):
        self.last_seen = now
        self.last_box = tuple(float(value) for value in box)
        self.candidate_counts.clear()
        self.candidate_depth_runs.clear()
        self.other_ids = {
            track_id: state for track_id, state in self.other_ids.items()
            if now - state[0] <= self.max_gap + 1.0
        }
        for track_id, other_box, confidence in other_tracks:
            overlap = smaller_box_overlap(self.last_box, other_box)
            previous = self.other_ids.get(track_id)
            if previous is None:
                # A weak overlapping detection can be a duplicate of the
                # selected person. A separate person is never eligible here.
                duplicate = confidence < 0.4 and overlap >= 0.25
            else:
                # Once a track separates from the selected person, treat it
                # as another person even if it later crosses the target.
                duplicate = previous[1] and overlap >= 0.25
            self.other_ids[track_id] = (now, duplicate)
        if depth is None:
            return
        self.pending_depth = None
        self.pending_depth_count = 0
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
        next_depth_runs = {}
        for track_id, _box, depth in candidates:
            if depth is None:
                continue
            previous = self.candidate_depth_runs.get(track_id)
            run = previous[1] + 1 if previous and abs(depth - previous[0]) <= 0.08 else 1
            next_depth_runs[track_id] = (depth, run)
        self.candidate_depth_runs = next_depth_runs
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
            shifted_depth = depth_change > 0.30
            # A closer shifted range is commonly the foreground occluder.
            if shifted_depth and depth < self.last_depth:
                continue
            if similarity < (0.82 if shifted_depth else 0.72):
                continue
            position_score = 1.0 - (dx / max_dx + dy / max_dy) / 2.0
            depth_score = 1.0 - depth_change / max_depth_change
            score = (
                0.7 * similarity + 0.2 * position_score + 0.1 * depth_score
                if shifted_depth else
                0.6 * similarity + 0.2 * position_score + 0.2 * depth_score
            )
            scores.append((score, track_id, similarity, shifted_depth))

        if not scores:
            return None
        scores.sort(reverse=True)
        best_score, best_id, best_similarity, shifted_depth = scores[0]
        prior_track = self.other_ids.get(best_id)
        if prior_track is not None and not prior_track[1]:
            return None
        stable_count = (
            self.candidate_depth_runs[best_id][1]
            if shifted_depth else self.candidate_counts[best_id]
        )
        required_frames = max(self.stable_frames, 5) if shifted_depth else self.stable_frames
        if stable_count < required_frames:
            return None
        if best_score < 0.75:
            return None
        if prior_track is not None and (best_score < 0.82 or best_similarity < 0.80):
            return None
        if len(scores) > 1 and best_score - scores[1][0] < 0.12:
            return None
        return best_id, best_score
