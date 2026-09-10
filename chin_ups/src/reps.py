"""Vertical displacement and rep counting.

The signal is the torso's height relative to the hands on the bar, not one
keypoint: a single joint carries its own noise, and a screen position does not
survive a handheld camera. See rep-counting-explained.md for how a rep is found
and timed, and config.py for what each default is doing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .skeleton import KPT_NAMES

_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}

# Which span each cfg.REP_METRIC reports, as (Rep attribute, name, endpoints).
METRICS = {
    "ascent": ("ascent_seconds", "concentric", "valley to peak"),
    "moving": ("moving_seconds", "concentric, moving only", "onset to peak"),
    "total": ("total_seconds", "full cycle", "valley to valley"),
}


def rep_value(rep, metric: str) -> float:
    """The per-rep number the UI plots, per cfg.REP_METRIC."""
    return float(getattr(rep, METRICS[metric][0]))


def phase_label(metric: str) -> str:
    """Human name for the span *metric* measures, e.g. "concentric (valley to peak)"."""
    _, name, span = METRICS[metric]
    return f"{name} ({span})"


def extremes(values, *, tol: float = 1e-6) -> tuple[int, int]:
    """Indices of the fastest and slowest rep, with ties broken by position.

    Reps are frame-quantized, so exact ties are common rather than a corner
    case. Which one gets named is a real choice: the *earliest* rep to reach the
    best time (you got there before fatigue) and the *latest* to reach the worst
    (that is where the set ended up). ``argmin``/``argmax`` would name the first
    of each, reporting a mid-set rep as slowest while later reps match it.
    """
    fastest, slowest = min(values), max(values)
    fast_i = next(i for i, v in enumerate(values) if v <= fastest + tol)
    slow_i = next(i for i in range(len(values) - 1, -1, -1) if values[i] >= slowest - tol)
    return fast_i, slow_i


@dataclass
class Rep:
    """One valley-to-peak-to-valley cycle."""

    number: int
    valley_frame: int
    peak_frame: int
    end_frame: int
    valley_time: float
    peak_time: float
    end_time: float
    ascent_seconds: float
    descent_seconds: float
    total_seconds: float
    height: float
    onset_time: float = 0.0
    # Onset to peak: the ascent minus any dead hang at the bottom. The two differ
    # by however long the body waited before pulling, so which one you want is a
    # judgement call rather than a rounding difference.
    moving_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "number": self.number,
            "valley_time": round(self.valley_time, 3),
            "peak_time": round(self.peak_time, 3),
            "end_time": round(self.end_time, 3),
            "ascent_seconds": round(self.ascent_seconds, 3),
            "moving_seconds": round(self.moving_seconds, 3),
            "onset_time": round(self.onset_time, 3),
            "descent_seconds": round(self.descent_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "height": round(self.height, 4),
            "frames": {"valley": self.valley_frame, "peak": self.peak_frame,
                       "end": self.end_frame},
        }


@dataclass
class RepAnalysis:
    times: np.ndarray
    raw: np.ndarray
    displacement: np.ndarray
    head_margin: np.ndarray
    baseline: float
    amplitude: float
    reps: list[Rep] = field(default_factory=list)
    counts_by_source_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    on_bar: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    @property
    def off_bar_frames(self) -> int:
        return int((~self.on_bar).sum()) if len(self.on_bar) else 0

    @property
    def count(self) -> int:
        return len(self.reps)

    def count_at(self, frame: int) -> int:
        """Reps completed by *frame*, for the progressive panel."""
        counts = self.counts_by_source_frame
        return int(counts[frame]) if frame < len(counts) else self.count

    def summary(self) -> dict:
        if not self.reps:
            return {"reps": 0}
        total = [r.total_seconds for r in self.reps]
        return {
            "reps": len(self.reps),
            "baseline_signal": round(self.baseline, 4),
            "peak_displacement": round(self.amplitude, 4),
            "mean_ascent_seconds": round(float(np.mean([r.ascent_seconds for r in self.reps])), 3),
            "mean_descent_seconds": round(float(np.mean([r.descent_seconds for r in self.reps])), 3),
            "mean_rep_seconds": round(float(np.mean(total)), 3),
            "fastest_rep_seconds": round(float(np.min(total)), 3),
            "slowest_rep_seconds": round(float(np.max(total)), 3),
            "reps_detail": [r.as_dict() for r in self.reps],
        }

    def series(self) -> dict:
        """The full per-frame signal, for reps.json."""
        return {
            "time": [round(float(v), 4) for v in self.times],
            "displacement": [round(float(v), 5) for v in self.displacement],
            "head_margin": [round(float(v), 5) for v in self.head_margin],
        }


def signal_config(cfg) -> dict:
    """The settings that shaped the signal, snapshotted into reps.json."""
    return {
        "torso_joints": list(cfg.TORSO_JOINTS),
        "hand_joints": list(cfg.HAND_JOINTS),
        "head_joints": list(cfg.HEAD_JOINTS),
        "referenced_to_hands": cfg.REFERENCE_TO_HANDS,
        "smooth_median_frames": cfg.SMOOTH_MEDIAN_FRAMES,
        "smooth_mean_frames": cfg.SMOOTH_MEAN_FRAMES,
        "baseline_percentile": cfg.BASELINE_PERCENTILE,
        "peak_fraction": cfg.REP_PEAK_FRACTION,
        "reset_fraction": cfg.REP_RESET_FRACTION,
        "require_head_above_hands": cfg.REQUIRE_HEAD_ABOVE_HANDS,
    }


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Kill isolated spikes without rounding off the peaks."""
    if k < 3:
        return x
    k += (k + 1) % 2   # force odd
    pad = np.pad(x, k // 2, mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(x))])


def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average. Centered and edge-padded, so it does not shift peak times."""
    if k < 2:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _fill_gaps(x: np.ndarray) -> np.ndarray:
    """Interpolate frames where the joints were missing."""
    valid = np.isfinite(x)
    if valid.all():
        return x
    if not valid.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[valid], x[valid])


def joint_y(frames: list[dict], names: tuple[str, ...]) -> np.ndarray:
    """Mean y of the named joints on the first tracked person, per returned frame.

    (0, 0) is the model's "not visible" sentinel, so it is excluded rather than
    averaged in as a real point at the top-left corner.
    """
    out = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        persons = frame.get("persons") or []
        if not persons:
            continue
        kpts = persons[0].get("kpts_xy") or []
        vals = [
            kpts[_INDEX[n]][1]
            for n in names
            if _INDEX[n] < len(kpts) and tuple(kpts[_INDEX[n]]) != (0, 0)
        ]
        if vals:
            out[i] = float(np.mean(vals))
    return out


def _onset_index(displacement, valley_i: int, peak_i: int, *, amplitude: float, cfg) -> int:
    """Where the pull begins: the foot of the rise, not the lowest point.

    ``threshold`` reads the signal at the bottom, where the body is nearly still
    and the noise floor is the whole signal. Push it low enough to sit at the base
    and a dead hang's jitter crosses it early. ``velocity`` walks back from the
    fastest point of the rise instead, reading the signal where the movement is
    large, which is what makes it steadier exactly where the threshold is not.
    """
    if peak_i <= valley_i:
        return valley_i

    if cfg.REP_ONSET_METHOD == "threshold":
        rise = np.where(
            displacement[valley_i:peak_i + 1]
            > displacement[valley_i] + cfg.REP_ONSET_FRACTION * amplitude
        )[0]
        return valley_i + int(rise[0]) if len(rise) else valley_i

    segment = displacement[valley_i:peak_i + 1]
    if len(segment) < 3:
        return valley_i
    velocity = np.gradient(segment)
    fastest = int(np.argmax(velocity))
    if velocity[fastest] <= 0:
        return valley_i

    floor = cfg.REP_ONSET_VELOCITY_FRACTION * velocity[fastest]
    i = fastest
    while i > 0 and velocity[i] > floor:
        i -= 1
    return valley_i + i


def _robust_amplitude(displacement: np.ndarray, valid: np.ndarray) -> float:
    """Typical rep height, as the median of the peaks rather than the max.

    ``max()`` is one frame's opinion, and every threshold downstream is a
    fraction of this number, so one bad pose would move all of them.
    """
    usable = displacement[valid]
    if not len(usable):
        return 0.0
    top = float(np.max(usable))
    if top <= 0:
        return 0.0

    peaks = [
        displacement[i]
        for i in range(1, len(displacement) - 1)
        if valid[i]
        and displacement[i] >= displacement[i - 1]
        and displacement[i] > displacement[i + 1]
        and displacement[i] > 0.3 * top
    ]
    if len(peaks) >= 3:
        return float(np.median(peaks))
    return float(np.percentile(usable, 95))


def analyze(frames: list[dict], fps: float, n_source_frames: int, *, cfg) -> RepAnalysis:
    """Build the displacement signal and count reps off it."""
    indices = np.array([f["index"] for f in frames], dtype=int)
    times = indices / fps

    torso = _fill_gaps(joint_y(frames, cfg.TORSO_JOINTS))
    hands = _fill_gaps(joint_y(frames, cfg.HAND_JOINTS))
    head = _fill_gaps(joint_y(frames, cfg.HEAD_JOINTS))

    # Body height relative to the hands on the bar, falling back to screen
    # position if the wrists were never found.
    on_bar = np.ones(len(torso), dtype=bool)
    if cfg.REFERENCE_TO_HANDS and np.isfinite(hands).any():
        raw = hands - torso
        # The signal assumes the hands are a fixed point. Let go and that is
        # simply false. The wrists drop to the hips and the "displacement"
        # spikes to nearly double a real rep, so those frames are excluded.
        on_bar = np.abs(hands - np.median(hands)) <= cfg.HAND_OFF_BAR_TOLERANCE
    else:
        raw = -torso

    smooth = _box_filter(_median_filter(raw, cfg.SMOOTH_MEDIAN_FRAMES), cfg.SMOOTH_MEAN_FRAMES)

    # Relative zero is the hanging plateau, taken as a low percentile rather
    # than the minimum so one bad frame cannot define it.
    usable = smooth[on_bar] if on_bar.any() else smooth
    baseline = float(np.percentile(usable, cfg.BASELINE_PERCENTILE))
    displacement = smooth - baseline
    amplitude = _robust_amplitude(displacement, on_bar)

    # Positive when the eyes are above the hands, the chin-over-bar criterion.
    head_margin = hands - head

    reps = _count(displacement, head_margin, times, indices, amplitude, on_bar, cfg=cfg)

    counts = np.zeros(max(n_source_frames, 1), dtype=int)
    for rep in reps:
        if rep.peak_frame < len(counts):
            counts[rep.peak_frame:] = rep.number
    return RepAnalysis(times, raw, displacement, head_margin, baseline, amplitude,
                       reps, counts, on_bar)


def _count(displacement, head_margin, times, indices, amplitude, on_bar, *, cfg) -> list[Rep]:
    """Hysteresis state machine: armed at a valley, confirmed at a peak, closed
    when the body returns near the baseline."""
    if amplitude <= 0 or len(displacement) < 3:
        return []

    high = cfg.REP_PEAK_FRACTION * amplitude
    low = cfg.REP_RESET_FRACTION * amplitude

    reps: list[Rep] = []
    state, valley_i, peak_i, peak_v = "down", 0, 0, -np.inf
    last_on_bar = 0

    def close(end_i: int) -> None:
        reps.append(_make_rep(
            len(reps) + 1, valley_i, peak_i, end_i, times, indices, displacement,
            onset_i=_onset_index(displacement, valley_i, peak_i,
                                 amplitude=amplitude, cfg=cfg)))

    for i, d in enumerate(displacement):
        if not on_bar[i]:
            continue   # hands off the bar: this frame says nothing about reps
        last_on_bar = i
        gate = (not cfg.REQUIRE_HEAD_ABOVE_HANDS) or bool(head_margin[i] > 0)
        if state == "down":
            if d < displacement[valley_i]:
                valley_i = i
            if d > high and gate:
                state, peak_i, peak_v = "up", i, d
        else:
            if d > peak_v:
                peak_i, peak_v = i, d
            if d < low:
                close(i)
                state, valley_i = "down", i

    if state == "up":
        # The clip ended mid-rep. Close on the last frame the hands were still
        # on the bar: after that the descent is a release, and timing it to the
        # clip's end would invent a lowering phase that never happened.
        close(last_on_bar)
    return reps


def _make_rep(number, valley_i, peak_i, end_i, times, indices, displacement,
              *, onset_i: int | None = None) -> Rep:
    onset_i = valley_i if onset_i is None else onset_i
    return Rep(
        number=number,
        valley_frame=int(indices[valley_i]),
        peak_frame=int(indices[peak_i]),
        end_frame=int(indices[end_i]),
        valley_time=float(times[valley_i]),
        peak_time=float(times[peak_i]),
        end_time=float(times[end_i]),
        ascent_seconds=float(times[peak_i] - times[valley_i]),
        descent_seconds=float(times[end_i] - times[peak_i]),
        total_seconds=float(times[end_i] - times[valley_i]),
        height=float(displacement[peak_i]),
        onset_time=float(times[onset_i]),
        moving_seconds=float(times[peak_i] - times[onset_i]),
    )


def summary_text(analysis: RepAnalysis, *, video_seconds: float, source: str = "",
                 metric: str = "ascent") -> str:
    """Human-readable rep report, saved alongside the machine-readable JSON."""
    lines = ["Chin-up summary", "=" * 58]
    if source:
        lines.append(f"{'source':<22}{source}")
    lines += [
        f"{'clip length':<22}{video_seconds:.2f}s",
        f"{'total reps':<22}{analysis.count}",
    ]

    if not analysis.reps:
        lines.append("\nNo reps detected. Check displacement.png.")
        return "\n".join(lines) + "\n"

    s = analysis.summary()
    values = [rep_value(r, metric) for r in analysis.reps]
    totals = [r.total_seconds for r in analysis.reps]
    work = float(np.sum(totals))
    fast_i, slow_i = extremes(values)

    lines += [
        f"{'reported phase':<22}{phase_label(metric)}",
        f"{'mean':<22}{float(np.mean(values)):.2f}s",
        f"{'fastest':<22}{min(values):.2f}s   (rep {fast_i + 1})",
        f"{'slowest':<22}{max(values):.2f}s   (rep {slow_i + 1})",
        f"{'spread':<22}{max(values) - min(values):.2f}s",
        "",
        f"{'mean ascent':<22}{s['mean_ascent_seconds']:.2f}s   (valley to peak)",
        f"{'mean moving':<22}"
        f"{float(np.mean([r.moving_seconds for r in analysis.reps])):.2f}s   (onset to peak)",
        f"{'mean descent':<22}{s['mean_descent_seconds']:.2f}s",
        f"{'mean full cycle':<22}{float(np.mean(totals)):.2f}s",
        f"{'time under tension':<22}{work:.2f}s of {video_seconds:.2f}s "
        f"({work / video_seconds * 100:.0f}%)",
        f"{'pace':<22}{analysis.count / video_seconds * 60:.1f} reps/min",
        "",
        "Per rep",
        "-" * 58,
        f"{'rep':>3} {'valley':>8} {'onset':>8} {'peak':>8} {'up':>7} {'moving':>7} "
        f"{'down':>7} {'total':>7}",
    ]
    for r in analysis.reps:
        lines.append(
            f"{r.number:>3} {r.valley_time:>7.2f}s {r.onset_time:>7.2f}s {r.peak_time:>7.2f}s "
            f"{r.ascent_seconds:>6.2f}s {r.moving_seconds:>6.2f}s "
            f"{r.descent_seconds:>6.2f}s {r.total_seconds:>6.2f}s"
        )

    rests = [
        analysis.reps[i + 1].valley_time - analysis.reps[i].end_time
        for i in range(len(analysis.reps) - 1)
    ]
    if rests:
        lines += [
            "",
            "Rest between reps",
            "-" * 58,
            "  " + "  ".join(f"{r:.2f}s" for r in rests)
            + f"   (mean {float(np.mean(rests)):.2f}s)",
        ]

    lines += [
        "",
        "`up` is valley to peak; `moving` excludes the dead hang before the pull",
        "starts. Ascent lengthening while range of motion holds is fatigue.",
    ]
    return "\n".join(lines) + "\n"
