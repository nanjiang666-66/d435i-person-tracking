"""Experimental pose-supported RGB-D recovery, independent of ROS/YOLO.

Joint confidence is not proof of visibility. Require agreement across visible
body patches and with a frozen region-size reference before updating range.
This is short-term association, not biometric identity recognition.
"""

from dataclasses import dataclass

import numpy as np

from person_vision.target_recovery import smaller_box_overlap


@dataclass
class BodyObservation:
    depth: float
    u: float
    v: float
    scale: float
    colour: np.ndarray
    brightness: float
    support: int
    kind: str = "hips"
    scales: dict = None


def torso_regions(box, joints, shape):
    """Find visible torso regions using hips or shoulders independently."""
    if joints is None:
        return []
    k = np.asarray(joints)
    if k.shape != (17, 3) or not np.isfinite(k).all():
        return []
    hips, shoulders = k[[11, 12]], k[[5, 6]]
    regions = []

    def add_region(roi, span, kind):
        x1, y1, x2, y2 = roi
        roi = (max(0, int(x1)), max(0, int(y1)),
               min(shape[1], int(x2)), min(shape[0], int(y2), int(box[3])))
        if roi[2] - roi[0] >= 12 and roi[3] - roi[1] >= 24:
            regions.append((roi, span, kind))

    span = float(np.linalg.norm(hips[0, :2] - hips[1, :2]))
    if (np.min(hips[:, 2]) >= 0.85 and span >= 18
            and abs(hips[0, 1] - hips[1, 1]) <= span * 0.6
            and np.all(hips[:, 1] < shape[0] - 2)):
        bottom = hips[:, :2].mean(axis=0)
        if np.min(shoulders[:, 2]) >= 0.65:
            top = shoulders[:, :2].mean(axis=0)
            width = float(np.linalg.norm(
                (shoulders[0, :2] + hips[0, :2]) / 2
                - (shoulders[1, :2] + hips[1, :2]) / 2
            ))
        elif box[1] <= 8 and bottom[1] > 80:
            top = np.array([bottom[0], max(0.0, box[1])])
            width = span
        else:
            top = None
        if top is not None:
            height = bottom[1] - top[1]
            cx = (top[0] + bottom[0]) / 2
            if height >= 45 and abs(top[0] - bottom[0]) <= span:
                add_region((cx - .3 * width, top[1] + .15 * height,
                            cx + .3 * width, top[1] + .75 * height), span, "hips")

    # Chest below reliable shoulders is usable in a seated/upper-body view.
    # Do not depend on guessed hips below the camera crop.
    shoulder_span = float(np.linalg.norm(shoulders[0, :2] - shoulders[1, :2]))
    if (np.min(shoulders[:, 2]) >= 0.80 and shoulder_span >= 30
            and abs(shoulders[0, 1] - shoulders[1, 1]) <= shoulder_span * .6
            and np.all(shoulders[:, 0] > 2) and np.all(shoulders[:, 0] < shape[1] - 2)
            and np.all(shoulders[:, 1] > 2) and np.all(shoulders[:, 1] < shape[0] - 2)):
        cx, cy = shoulders[:, :2].mean(axis=0)
        add_region((cx - .25 * shoulder_span, cy + .20 * shoulder_span,
                    cx + .25 * shoulder_span, cy + .75 * shoulder_span),
                   shoulder_span, "shoulders")
    return regions


def visible_regions(box, joints, shape):
    """Torso and a small nose/upper-mouth patch, each independently available.

    Eyes supply pixel scale, not depth: the patch sits below the glasses.
    A head outside the frame or low-confidence face landmarks supplies no patch.
    """
    regions = torso_regions(box, joints, shape)
    if joints is None:
        return regions
    k = np.asarray(joints)
    if k.shape != (17, 3) or not np.isfinite(k).all():
        return regions
    nose, eyes = k[0], k[[1, 2]]
    span = float(np.linalg.norm(eyes[0, :2] - eyes[1, :2]))
    midpoint = eyes[:, :2].mean(axis=0)
    if (nose[2] < .85 or np.min(eyes[:, 2]) < .70 or span < 14
            or abs(eyes[0, 1] - eyes[1, 1]) > .7 * span
            or abs(nose[0] - midpoint[0]) > .9 * span
            or not -.2 * span <= nose[1] - midpoint[1] <= 1.2 * span):
        return regions
    x1, y1 = nose[:2] - np.array([.40, .15]) * span
    x2, y2 = nose[:2] + np.array([.40, .55]) * span
    # Do not fabricate a cropped face patch by extending it to an image border.
    if (x1 >= max(1, box[0]) and y1 >= max(1, box[1])
            and x2 < min(shape[1] - 1, box[2])
            and y2 < min(shape[0] - 1, box[3])
            and x2 - x1 >= 10 and y2 - y1 >= 10):
        regions.append(((int(x1), int(y1), int(x2), int(y2)), span, 'face'))
    return regions


def _observe_region(frame, depth, encoding, region, depth_scale, fx, reference,
                    diagnostic):
    roi, span, kind = region
    diagnostic.update(region=kind, roi=list(roi), span_px=round(span, 1),
                      fx=round(float(fx), 1),
                      scale_reference=None if reference is None else round(float(reference), 2))
    x1, y1, x2, y2 = roi
    pixels = depth[y1:y2, x1:x2].astype(np.float32)
    if encoding == "16UC1":
        pixels *= depth_scale
    elif encoding != "32FC1":
        diagnostic['gate'] = 'depth_encoding'
        return None, "unsupported_depth_encoding"
    valid = np.isfinite(pixels) & (pixels >= 0.25) & (pixels <= 6.0)
    raw_count = int(np.count_nonzero(valid))
    raw_median = float(np.median(pixels[valid])) if raw_count else None
    diagnostic.update(raw_valid_pct=round(100 * raw_count / pixels.size, 1),
                      raw_median_m=None if raw_median is None else round(raw_median, 3),
                      projected_width_m=None if raw_median is None else round(raw_median * span / fx, 3))
    limits = {'hips': (.12, .65), 'shoulders': (.18, .85), 'face': (.025, .13)}[kind]
    valid &= (pixels * span / fx >= limits[0]) & (pixels * span / fx <= limits[1])
    width_count = int(np.count_nonzero(valid))
    diagnostic['width_valid_pct'] = round(100 * width_count / pixels.size, 1)
    if reference is not None:
        # Eye separation shrinks when turning the head; it is not a fixed
        # frontal anatomical width. Keep a bounded tolerance for that view.
        low, high = (.55, 1.45) if kind == 'face' else (.70, 1.30)
        valid &= (pixels * span >= low * reference) & (pixels * span <= high * reference)
    diagnostic['reference_valid_pct'] = round(100 * np.count_nonzero(valid) / pixels.size, 1)
    min_count = max(30, pixels.size * 0.3)
    if np.count_nonzero(valid) < min_count:
        diagnostic['gate'] = ('raw_depth_support' if raw_count < min_count else
                              'projected_body_width' if width_count < min_count else
                              'frozen_body_reference')
        return None, "body_depth_inconsistent"
    hist, edges = np.histogram(pixels[valid], bins=np.arange(0.25, 6.051, 0.05))
    band = np.convolve(hist, np.ones(5, dtype=int), mode="same")
    centre = float((edges[np.argmax(band)] + edges[np.argmax(band) + 1]) / 2)
    radius = .06 if kind == 'face' else .15
    selected = valid & (np.abs(pixels - centre) <= radius)
    # Use the modal bin for a narrow face band; a smoothed plateau's first
    # bin can sit too far before the surface to capture its pixels.
    if kind == 'face':
        peak = int(np.argmax(hist))
        centre = float((edges[peak] + edges[peak + 1]) / 2)
        selected = valid & (np.abs(pixels - centre) <= radius)
    if np.count_nonzero(selected) < max(30, pixels.size * 0.3):
        diagnostic['gate'] = 'depth_band_support'
        return None, "insufficient_body_depth"
    z = float(np.median(pixels[selected]))
    diagnostic['candidate_m'] = round(z, 3)
    selected = valid & (np.abs(pixels - z) <= radius)
    support = []
    patch_percentages = []
    for row, ys in enumerate(np.array_split(np.arange(pixels.shape[0]), 3)):
        for col, xs in enumerate(np.array_split(np.arange(pixels.shape[1]), 2)):
            fraction = float(selected[np.ix_(ys, xs)].mean())
            patch_percentages.append(round(100 * fraction, 1))
            if fraction >= 0.50:
                support.append((row, col))
    diagnostic.update(supported_patches=len(support), patch_pct=patch_percentages)
    if len(support) < 4 or len({c for _, c in support}) < 2:
        diagnostic['gate'] = 'patch_agreement'
        return None, 'face_partially_occluded' if kind == 'face' else 'torso_partially_occluded'
    colours = frame[y1:y2, x1:x2][selected].astype(np.float32)
    usable = colours.mean(axis=1) > 20
    if np.count_nonzero(usable) < 30:
        diagnostic['gate'] = 'appearance_support'
        return None, "insufficient_appearance"
    colours = colours[usable]
    colour = np.median(colours / np.maximum(colours.sum(axis=1, keepdims=True), 1), axis=0)
    ys, xs = np.nonzero(selected)
    # Select an actual supported pixel near the median range. Independent x/y
    # medians could put the ray in a depth hole or on an excluded occluder.
    near = np.abs(pixels[ys, xs] - z) <= 0.03
    if not np.any(near):
        near[np.argmin(np.abs(pixels[ys, xs] - z))] = True
    near_x, near_y = xs[near], ys[near]
    centre_x, centre_y = np.median(near_x), np.median(near_y)
    index = np.argmin((near_x - centre_x) ** 2 + (near_y - centre_y) ** 2)
    u = float(x1 + near_x[index])
    v = float(y1 + near_y[index])
    diagnostic['gate'] = 'region_supported'
    return BodyObservation(z, u, v, span * z, colour,
                           float(np.median(colours.mean(axis=1))), len(support),
                           kind, {kind: span * z}), "body_supported"


def _compatible_regions(a, b):
    """A torso surface may be behind the face, not a foreground object."""
    if 'face' in (a.kind, b.kind) and a.kind != b.kind:
        face, body = (a, b) if a.kind == 'face' else (b, a)
        return -.06 <= body.depth - face.depth <= .40
    return abs(a.depth - b.depth) <= .20


def observe_regions(frame, depth, encoding, box, joints, depth_scale, fx,
                    anchors=None, preferred_kind=None):
    """Try every available region; one rejected ROI does not reject the person."""
    regions = visible_regions(box, joints, depth.shape)
    if not regions:
        return [], {'gate': 'visible_body_region'}, 'uncertain_pose'
    if not np.isfinite(fx) or fx <= 0:
        return [], {'gate': 'intrinsics'}, 'no_intrinsics'
    anchors = anchors or {}
    regions.sort(key=lambda r: (r[2] != preferred_kind, r[2] not in anchors))
    observations, attempts = [], []
    for region in regions:
        diagnostic = {}
        obs, reason = _observe_region(frame, depth, encoding, region, depth_scale,
                                      fx, anchors.get(region[2]), diagnostic)
        diagnostic['reason'] = reason
        attempts.append(diagnostic)
        if obs is not None:
            observations.append((obs, diagnostic))
    first = observations[0][1] if observations else attempts[0]
    diagnostic = dict(first, attempts=attempts)
    return observations, diagnostic, 'body_supported' if observations else first['reason']


def observe_body(frame, depth, encoding, box, joints, depth_scale, fx, anchor=None,
                 preferred_kind=None, diagnostic=None):
    """Compatibility helper returning the first supported visible region."""
    anchors = anchor if isinstance(anchor, dict) else ({'hips': anchor} if anchor else {})
    observations, details, reason = observe_regions(
        frame, depth, encoding, box, joints, depth_scale, fx, anchors, preferred_kind)
    if diagnostic is not None:
        diagnostic.clear()
        diagnostic.update(details)
    if not observations:
        return None, reason
    obs = observations[0][0]
    obs.scales = {candidate.kind: candidate.scale for candidate, _ in observations
                  if _compatible_regions(obs, candidate)}
    return obs, reason


class PoseDepthRecovery:
    """Keep trusted range, identity hints and unconfirmed observations separate."""

    def __init__(self, max_gap=3.0):
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.selected_id = -1
        self.anchor = None
        self.anchors = {}
        self.appearances = {}
        self.region_pending = {}
        self.last_kind = None
        self.colour = None
        self.brightness = None
        self.last_box = None
        self.last_depth = None
        self.last_valid_time = None
        self.other_ids = set()
        self.pending = None
        self.invalid = True
        self.last_sample_time = None
        self.reason = "no_target"
        self.diagnostic = {}

    def _appearance_ok(self, obs):
        # Coarse colour compatibility only. Geometry, time and ambiguity gates
        # also apply; these colour values are not an identity probability.
        reference = self.appearances.get(obs.kind)
        if reference is None:
            return True
        colour, brightness = reference
        return (np.max(np.abs(obs.colour - colour)) <= 0.12
                and 0.25 <= obs.brightness / max(brightness, 1) <= 4.0)

    def _confirm_regions(self, observations, chosen, track_id, now):
        """Stage independent region references; never reuse torso colour for skin."""
        supported = set()
        for obs, _ in observations:
            if not _compatible_regions(chosen, obs) or not self._appearance_ok(obs):
                continue
            supported.add(obs.kind)
            if obs.kind in self.anchors:
                continue
            old = self.region_pending.get(obs.kind)
            if (old is None or old[0] != track_id or now - old[3] > .8
                    or abs(obs.scale - old[1]) > .2 * max(old[1], 1)
                    or np.max(np.abs(obs.colour - old[5])) > .12):
                self.region_pending[obs.kind] = (
                    track_id, obs.scale, now, now, 1, obs.colour.copy(), obs.brightness)
            else:
                self.region_pending[obs.kind] = (
                    track_id, .7 * old[1] + .3 * obs.scale, old[2], now,
                    old[4] + 1, old[5], old[6])
        for kind in list(self.region_pending):
            if kind not in supported:
                del self.region_pending[kind]

    def _store_confirmed_regions(self, chosen, now):
        # The chosen measurement passed the main identity/time confirmation.
        self.anchors.setdefault(chosen.kind, chosen.scale)
        self.appearances.setdefault(chosen.kind, (chosen.colour.copy(), chosen.brightness))
        for kind, sample in list(self.region_pending.items()):
            if sample[4] >= 3 and now - sample[2] >= .15:
                self.anchors.setdefault(kind, sample[1])
                self.appearances.setdefault(kind, (sample[5], sample[6]))
                del self.region_pending[kind]

    def _stable(self, track_id, obs, now, required_frames=3, duration=0.15):
        old = self.pending
        if (old is None or old[0] != track_id or old[5] != obs.kind or now - old[3] > 0.8
                or now <= old[3] or abs(obs.depth - old[1]) > 0.15):
            self.pending = (track_id, obs.depth, now, now, 1, obs.kind)
            return False
        self.pending = (track_id, 0.7 * old[1] + 0.3 * obs.depth,
                        old[2], now, old[4] + 1, obs.kind)
        return self.pending[4] >= required_frames and now - self.pending[2] >= duration

    def update(self, frame, depth, encoding, tracks, requested_id, now, fx,
               depth_scale=0.001, auto_reacquire=True):
        """Return (selected_id, accepted observation or None, reason).

        tracks: dictionaries with id, box, confidence and keypoints.
        now must be strictly increasing and in seconds (image timestamps work).
        """
        if requested_id != self.selected_id:
            self.reset()
            self.selected_id = requested_id
        self.diagnostic = {}
        if requested_id < 0:
            return requested_id, None, "no_target"
        if self.last_sample_time is not None and now <= self.last_sample_time:
            self.pending = None
            self.invalid = True
            self.reason = "stale_image"
            return self.selected_id, None, self.reason
        self.last_sample_time = now
        current = next((t for t in tracks if t['id'] == self.selected_id), None)
        if current is not None:
            for t in tracks:
                if t['id'] != self.selected_id and smaller_box_overlap(current['box'], t['box']) < 0.25:
                    self.other_ids.add(t['id'])
        # Only a reliable human observation refreshes the recovery clock.
        observations = []
        regions_by_id = {}
        diagnostics_by_id = {}
        selected_failure = "target_missing"
        for t in tracks:
            same = t['id'] == self.selected_id
            if not same:
                if (current is not None or not auto_reacquire or self.last_valid_time is None
                        or now - self.last_valid_time > self.max_gap or t['id'] in self.other_ids):
                    continue
            # Manual selection/continuing the same ID already supplies identity
            # context. A new ID needs a stronger detector score for association.
            if t['confidence'] < (0.25 if same else 0.65):
                if same:
                    selected_failure = "low_detection_confidence"
                continue
            candidates, diagnostic, reason = observe_regions(
                frame, depth, encoding, t['box'], t.get('keypoints'),
                depth_scale, fx, self.anchors, self.last_kind)
            obs = None
            for candidate, details in candidates:
                face = next((other for other, _ in candidates
                             if other.kind == 'face' and self._appearance_ok(other)), None)
                if (candidate.kind != 'face' and face is not None
                        and not _compatible_regions(candidate, face)):
                    reason = 'body_regions_disagree'
                    continue
                # A previously unseen face can supply its own evidence for a
                # continuing selected ID, with longer confirmation below.
                # An unknown torso ROI cannot relock solely from old range.
                if self.anchors and candidate.kind not in self.anchors:
                    supported_old = any(
                        other.kind in self.anchors and self._appearance_ok(other)
                        and _compatible_regions(candidate, other)
                        for other, _ in candidates)
                    if (not same or (candidate.kind != 'face' and not supported_old and
                            (self.invalid or now - self.last_valid_time > .8
                             or abs(candidate.depth - self.last_depth) > .25))):
                        reason = 'uncalibrated_body_region'
                        continue
                if not self._appearance_ok(candidate):
                    reason = 'body_appearance_inconsistent'
                    continue
                obs = candidate
                diagnostic = dict(details, attempts=diagnostic['attempts'])
                reason = 'body_supported'
                break
            diagnostics_by_id[t['id']] = diagnostic
            regions_by_id[t['id']] = candidates
            if same:
                self.diagnostic = diagnostic
            if obs is not None and not same:
                gap = now - self.last_valid_time
                old = np.array(self.last_box).reshape(2, 2).mean(axis=0)
                new = np.array(t['box']).reshape(2, 2).mean(axis=0)
                limits = np.array(frame.shape[1::-1]) * (0.18 + 0.06 * gap)
                if np.any(np.abs(new - old) > limits):
                    obs, reason = None, "outside_recovery_region"
                elif abs(obs.depth - self.last_depth) > 0.3 + 1.2 * gap:
                    obs, reason = None, "outside_motion_range"
                elif any(other['id'] != t['id'] and smaller_box_overlap(t['box'], other['box']) > 0.45
                         for other in tracks if other['confidence'] >= 0.5):
                    obs, reason = None, "overlapping_people"
            if same:
                selected_failure = reason
            if obs is not None:
                observations.append((t, obs))
        if len(observations) != 1:
            self.pending = None
            self.region_pending.clear()
            self.invalid = True
            self.reason = "ambiguous_candidates" if len(observations) > 1 else selected_failure
            return self.selected_id, None, self.reason
        track, obs = observations[0]
        self.diagnostic = diagnostics_by_id[track['id']]
        initial = self.anchor is None
        gap = float('inf') if self.last_valid_time is None else now - self.last_valid_time
        changed = track['id'] != self.selected_id
        self._confirm_regions(regions_by_id[track['id']], obs, track['id'], now)
        needs_confirmation = (initial or changed or self.invalid or gap > 0.8
                              or obs.kind not in self.anchors
                              or abs(obs.depth - self.last_depth) > min(0.30, 0.15 + 1.2 * gap))
        weak_detection = (track['confidence'] < .65 or
                          (bool(self.anchors) and obs.kind == 'face'
                           and obs.kind not in self.anchors))
        if needs_confirmation and not self._stable(
                track['id'], obs, now, 5 if weak_detection else 3,
                .30 if weak_detection else .15):
            self.reason = "confirming_body_depth"
            return self.selected_id, None, self.reason
        self.pending = None
        if initial:
            self.anchor = obs.scale
            self.colour = obs.colour.copy()
            self.brightness = obs.brightness
        self._store_confirmed_regions(obs, now)
        self.last_kind = obs.kind
        self.selected_id = track['id']
        self.last_box = tuple(track['box'])
        self.last_depth = obs.depth
        self.last_valid_time = now
        self.invalid = False
        self.reason = "reacquired" if changed else "valid"
        return self.selected_id, obs, self.reason
