"""Experimental depth association inside a visible-person segmentation mask.

No body landmarks or frontal body-size assumptions. A mask is a candidate,
not proof: depth components and frozen local appearance describe the observed
surface. Entirely hidden people have no measured position. Colour association
is a heuristic, not biometric identification or a guaranteed occlusion test.
"""

from dataclasses import dataclass

import numpy as np

from person_vision.target_recovery import smaller_box_overlap


@dataclass
class Surface:
    depth: float
    u: float
    v: float
    signature: np.ndarray
    support: int
    fraction: float
    bounds: tuple
    pixels: np.ndarray
    kind: str = "mask"


def colour_signature(colours):
    """Chromaticity histogram plus lightness; allow exposure changes."""
    colours = np.asarray(colours, dtype=np.float32).reshape(-1, 3)
    ratios = colours / np.maximum(colours.sum(axis=1, keepdims=True), 1)
    chroma, _, _ = np.histogram2d(ratios[:, 0], ratios[:, 2], bins=12,
                                 range=((0., 1.), (0., 1.)))
    # Black clothing needs a lightness cue; otherwise grey/white have the same
    # ratios. It receives less weight than chromaticity to tolerate exposure.
    light, _ = np.histogram(colours.mean(axis=1), bins=8, range=(0, 256))
    chroma = chroma.ravel().astype(np.float32)
    light = light.astype(np.float32)
    return np.concatenate((.8 * chroma / max(chroma.sum(), 1),
                           .2 * light / max(light.sum(), 1)))


def similarity(a, b):
    return float(np.sqrt(a * b).sum())


def person_surfaces(frame, depth, encoding, box, mask, depth_scale=.001,
                    excluded=None):
    """Return supported depth-connected surfaces and diagnostics.

    Masks must already match the aligned RGB/depth dimensions (retina_masks).
    Sampling every second pixel keeps component work bounded. Depth bands
    restrict 2-D connected components, so an obstacle and remaining body can
    produce separate candidates even when both were labelled as person.
    """
    import cv2

    details = {"gate": "mask", "candidates": []}
    if mask is None:
        return [], details, "no_person_mask"
    mask = np.asarray(mask)
    if mask.shape != depth.shape[:2] or frame.shape[:2] != mask.shape:
        return [], details, "mask_alignment_error"
    if len(depth.shape) != 2 or not np.isfinite(depth_scale) or depth_scale <= 0:
        return [], details, "invalid_depth_image"
    if encoding not in ("16UC1", "32FC1"):
        return [], details, "unsupported_depth_encoding"
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth.shape[1], x2), min(depth.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return [], details, "empty_person_mask"
    inside = np.zeros(mask.shape, np.uint8)
    inside[y1:y2, x1:x2] = mask[y1:y2, x1:x2] > .5
    if excluded is not None:
        if np.shape(excluded) != mask.shape:
            return [], details, "mask_alignment_error"
        inside[np.asarray(excluded, dtype=bool)] = 0
    # Exclude one-pixel RGB/depth boundaries; no ROI depends on anatomy.
    inside = cv2.erode(inside, np.ones((3, 3), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)[::2, ::2] != 0
    metres = depth[::2, ::2].astype(np.float32)
    if encoding == "16UC1":
        metres *= depth_scale
    valid = inside & np.isfinite(metres) & (metres >= .2) & (metres <= 8.)
    total = int(valid.sum())
    details.update(mask_pixels=int(inside.sum()) * 4, valid_pixels=total * 4)
    if total < 20:
        return [], details, "insufficient_mask_depth"
    values = metres[valid]
    counts, edges = np.histogram(values, bins=np.arange(.2, 8.051, .05))
    density = np.convolve(counts, np.ones(3), mode="same")
    minimum = max(20, int(total * .015))
    # Multiple surfaces, not the nearest depth or a single whole-mask median.
    peaks = []
    for index in np.argsort(density)[::-1]:
        centre = float((edges[index] + edges[index + 1]) / 2)
        if density[index] < minimum:
            break
        if all(abs(centre - peak) >= .18 for peak in peaks):
            peaks.append(centre)
        if len(peaks) == 8:
            break
    candidates = []
    for centre in peaks:
        band = (valid & (np.abs(metres - centre) <= .12)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
        for label in range(1, count):
            support = int(stats[label, cv2.CC_STAT_AREA])
            if support < minimum:
                continue
            rows, cols = np.where(labels == label)
            z = float(np.median(metres[rows, cols]))
            # The same surface can occur in neighbouring overlapping bands.
            if any(abs(z - old.depth) < .08 and
                   old.pixels[rows[0], cols[0]] for old in candidates):
                continue
            colours = frame[rows * 2, cols * 2]
            sig = colour_signature(colours)
            close = np.abs(metres[rows, cols] - z) <= .03
            if not close.any():
                close[np.argmin(np.abs(metres[rows, cols] - z))] = True
            near_rows, near_cols = rows[close], cols[close]
            at = np.argmin((near_rows - np.median(near_rows)) ** 2
                           + (near_cols - np.median(near_cols)) ** 2)
            bounds = (int(cols.min() * 2), int(rows.min() * 2),
                      int(cols.max() * 2 + 1), int(rows.max() * 2 + 1))
            candidates.append(Surface(z, float(near_cols[at] * 2),
                                      float(near_rows[at] * 2), sig, support * 4,
                                      support / total, bounds, labels == label))
    candidates.sort(key=lambda item: item.support, reverse=True)
    details["candidates"] = [{"z": round(c.depth, 3), "pixels": c.support,
                               "fraction": round(c.fraction, 3), "roi": list(c.bounds)}
                              for c in candidates]
    details["gate"] = "depth_components" if candidates else "component_support"
    return candidates, details, "supported_surfaces" if candidates else "no_supported_surface"


class SegmentDepthRecovery:
    """Independent experiment; never modifies the old pose/depth recovery."""

    def __init__(self, max_gap=3.):
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.selected_id = -1
        self.last_sample_time = None
        self.last_seen = None
        self.last_valid_time = None
        self.last_depth = None
        self.last_box = None
        self.bank = []
        self.identity_bank = []
        self.other_ids = set()
        self.pending = None
        self.invalid = True
        self.reason = "no_target"
        self.diagnostic = {}
        self.sample_pixels = None
        self.last_kind = None

    def _appearance(self, surface):
        return max((similarity(surface.signature, sig) for sig in self.bank), default=0.)

    def _remember(self, surface):
        # Frozen colour prototypes cannot gradually become an occluding object.
        if self._appearance(surface) < .96 and len(self.bank) < 12:
            self.bank.append(surface.signature.copy())

    @staticmethod
    def _identity_signature(frame, mask, excluded):
        if mask is None or np.shape(mask) != frame.shape[:2]:
            return None
        usable = np.asarray(mask) > .5
        if excluded is not None:
            usable &= ~excluded
        if np.count_nonzero(usable) < 80:
            return None
        signature = colour_signature(frame[usable])
        # Whole-person association needs more lightness evidence than a local
        # skin/clothing depth patch. A white sliver in a dark background mask
        # must not be sufficient to change the selected identity.
        signature[:-8] *= .5 / .8
        signature[-8:] *= .5 / .2
        return signature

    def _identity_similarity(self, signature):
        if signature is None:
            return 0.
        return max((similarity(signature, old) for old in self.identity_bank), default=0.)

    def _confirm(self, ident, surface, now, frames, duration):
        old = self.pending
        if (old is None or old[0] != ident or now <= old[3] or now - old[3] > .8
                or abs(surface.depth - old[1]) > .12
                or similarity(surface.signature, old[5]) < .7):
            self.pending = (ident, surface.depth, now, now, 1, surface.signature.copy())
            return False
        self.pending = (ident, .7 * old[1] + .3 * surface.depth, old[2], now,
                        old[4] + 1, old[5])
        return self.pending[4] >= frames and now - self.pending[2] >= duration

    def update(self, frame, depth, encoding, tracks, requested_id, now,
               depth_scale=.001, auto_reacquire=True, excluded=None):
        """tracks: id/box/confidence/full-resolution person mask. Image time in s."""
        if requested_id != self.selected_id:
            self.reset()
            self.selected_id = requested_id
        self.diagnostic = {}
        self.sample_pixels = None
        if requested_id < 0:
            self.reason = "no_target"
            return requested_id, None, self.reason
        if not np.isfinite(now) or (self.last_sample_time is not None and now <= self.last_sample_time):
            self.reason = "stale_frame"
            return requested_id, None, self.reason
        self.last_sample_time = now
        selected = next((t for t in tracks if t['id'] == requested_id), None)
        missing_age = float('inf') if self.last_seen is None else now - self.last_seen
        if selected is not None:
            self.last_seen = now  # 2-D detection continuity independent of depth validity.
            for other in tracks:
                overlap = smaller_box_overlap(selected['box'], other['box'])
                target_mask, other_mask = selected.get('mask'), other.get('mask')
                if (target_mask is not None and other_mask is not None
                        and np.shape(target_mask) == np.shape(other_mask)):
                    first, second = target_mask > .5, other_mask > .5
                    overlap = np.count_nonzero(first & second) / max(1, min(first.sum(), second.sum()))
                if other['id'] != requested_id and overlap < .2:
                    self.other_ids.add(other['id'])
        elif not auto_reacquire or not self.bank or missing_age > self.max_gap:
            self.pending = None
            self.invalid = True
            self.reason = "target_missing"
            return requested_id, None, self.reason
        proposals = []
        failure = "target_missing"
        diagnostics = []
        for track in ([selected] if selected is not None else tracks):
            changed = track['id'] != requested_id
            if changed and track['id'] in self.other_ids:
                continue
            if track['confidence'] < (.65 if changed else .25):
                failure = "low_detection_confidence"
                continue
            identity_signature = self._identity_signature(frame, track.get('mask'), excluded)
            identity_score = self._identity_similarity(identity_signature)
            if changed and identity_score < .65:
                failure = 'identity_appearance_mismatch'
                continue
            surfaces, diagnostic, failure = person_surfaces(
                frame, depth, encoding, track['box'], track.get('mask'), depth_scale, excluded)
            diagnostic.update(id=int(track['id']), confidence=round(track['confidence'], 3))
            diagnostics.append(diagnostic)
            if not surfaces:
                continue
            matched = []
            for surface in surfaces:
                appearance = self._appearance(surface)
                if self.bank and appearance < .70:
                    continue
                # Prefer a large component of the known person. Previous range
                # is only a mild ranking cue, never a permanent recovery limit.
                proximity = 0. if self.last_depth is None else float(np.exp(
                    -abs(surface.depth-self.last_depth) / .18))
                rank = .5 * appearance + .2 * surface.fraction + .3 * proximity
                matched.append((rank, surface, appearance))
            unknown = False
            if not self.bank:
                surface = surfaces[0]
                appearance = 0.
            elif matched:
                _, surface, appearance = max(matched, key=lambda item: item[0])
            else:
                # An unseen visible part (e.g. pants after an upper-body crop)
                # can be learned for the same ID at the established range. An
                # unmatched new foreground layer never replaces the reference.
                surface = surfaces[0]
                if (changed or track['confidence'] < .75 or self.last_depth is None
                        or abs(surface.depth - self.last_depth) > .12
                        or surface.depth < self.last_depth - .04 or surface.fraction < .5):
                    failure = "unmatched_surface"
                    continue
                appearance = self._appearance(surface)
                unknown = True
            movement = 0.
            if changed:
                centre = (np.array(track['box'][:2]) + np.array(track['box'][2:])) / 2
                old_centre = (np.array(self.last_box[:2]) + np.array(self.last_box[2:])) / 2
                span = max(self.last_box[2]-self.last_box[0], self.last_box[3]-self.last_box[1], 1)
                movement = float(np.linalg.norm(centre - old_centre) / span)
                if movement > .6 + .5 * missing_age or appearance < .75:
                    continue
            score = .3 * appearance + .3 * identity_score + .2 * track['confidence'] + .2 * max(0, 1-movement)
            diagnostic['identity_appearance'] = round(identity_score, 3)
            proposals.append((score, track, surface, appearance, unknown, surfaces, diagnostic,
                              identity_signature))
        self.diagnostic = {"tracks": diagnostics}
        if not proposals:
            self.pending = None
            self.invalid = True
            self.reason = failure if failure != "supported_surfaces" else "unmatched_surface"
            return requested_id, None, self.reason
        proposals.sort(key=lambda item: item[0], reverse=True)
        if len(proposals) > 1 and proposals[0][0] - proposals[1][0] < .12:
            self.pending = None
            self.invalid = True
            self.reason = "ambiguous_identity"
            return requested_id, None, self.reason
        _, track, surface, appearance, unknown, surfaces, diagnostic, identity_signature = proposals[0]
        diagnostic.update(chosen_m=round(surface.depth, 3), appearance=round(appearance, 3),
                          reference_m=None if self.last_depth is None else round(self.last_depth, 3),
                          roi=list(surface.bounds))
        self.diagnostic.update(diagnostic)
        changed = track['id'] != requested_id
        initial = self.last_depth is None
        gap = float('inf') if self.last_valid_time is None else now - self.last_valid_time
        needs_confirmation = (initial or changed or self.invalid or self.pending is not None
                              or unknown or gap > .8
                              or abs(surface.depth-self.last_depth) > min(.3, .15+1.2*gap))
        frames, duration = (5, .3) if changed or unknown or track['confidence'] < .65 else (3, .15)
        if needs_confirmation and not self._confirm(track['id'], surface, now, frames, duration):
            self.diagnostic['stable_frames'] = self.pending[4]
            self.reason = "confirming_mask_depth"
            return requested_id, None, self.reason
        self.pending = None
        self._remember(surface)
        if identity_signature is not None and len(self.identity_bank) < 8:
            identity_score = self._identity_similarity(identity_signature)
            if not self.identity_bank or (.75 <= identity_score < .96 and surface.fraction >= .35):
                self.identity_bank.append(identity_signature.copy())
        # Initially or while the known person is clearly measurable, store
        # companion body colours at similar ranges (clothes, skin, trousers).
        # A separate closer layer is never learned as a companion.
        if track['confidence'] >= .75:
            for companion in surfaces:
                if -.04 <= companion.depth - surface.depth <= .3 and companion.fraction >= .08:
                    self._remember(companion)
        self.last_box = tuple(track['box'])
        self.last_depth = surface.depth
        self.last_valid_time = now
        self.last_seen = now
        self.last_kind = surface.kind
        self.selected_id = track['id']
        self.invalid = False
        self.sample_pixels = surface.pixels
        self.reason = "reacquired" if changed else "valid"
        return self.selected_id, surface, self.reason
