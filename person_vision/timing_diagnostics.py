"""Bounded timing observations; never modify messages or tracking state."""

from collections import deque
import math


class FrameTiming:
    NAMES = (
        "color_rx", "depth_rx", "color_entry", "depth_entry",
        "xyz_pub", "pair_gap", "compute_to_pub",
    )

    def __init__(self, max_samples=512):
        self.samples = {name: deque(maxlen=max_samples) for name in self.NAMES}

    @staticmethod
    def stamp_seconds(message):
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        if stamp is None:
            return None
        try:
            sec, nsec = float(stamp.sec), float(stamp.nanosec)
        except (AttributeError, TypeError, ValueError):
            return None
        seconds = sec + nsec * 1e-9
        return seconds if math.isfinite(seconds) and seconds > 0 and 0 <= nsec < 1e9 else None

    def record_ms(self, name, milliseconds):
        if math.isfinite(milliseconds):
            self.samples[name].append(milliseconds)

    def record_age(self, name, now_seconds, message):
        stamp = self.stamp_seconds(message)
        if stamp is not None:
            self.record_ms(name, (now_seconds - stamp) * 1000.0)

    def record_pair(self, color, depth):
        first, second = self.stamp_seconds(color), self.stamp_seconds(depth)
        if first is not None and second is not None:
            self.record_ms("pair_gap", abs(first - second) * 1000.0)

    def report_and_reset(self):
        fields = []
        for name, values in self.samples.items():
            if values:
                fields.append(f"{name}={sum(values)/len(values):.0f}/{max(values):.0f} n={len(values)}")
            else:
                fields.append(f"{name}=none n=0")
            values.clear()
        return "Timing ms (mean/max): " + "; ".join(fields)
