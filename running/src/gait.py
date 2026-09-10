"""Ankle height, foot strikes and cadence.

One signal per leg: how far that **ankle** sits above its own stance level, in
units of the runner's leg length. It is near zero while the foot is planted and
rises to a peak at mid-swing, so a run reads as two interleaved waves — which is
what the panel plots and what every number here is measured off.

The ankle and not the foot, because COCO-17 has no foot: the gateway's ViTPose
returns 17 keypoints and the ankles are the lowest two, with no heel and no toes.
Two consequences run through this whole module. The zero has to be *calibrated*
per leg rather than being the floor, since an ankle joint stands several
centimetres above the ground even with the foot flat; and the angle of the foot
at touchdown is invisible, so the strike landmark carries an offset that depends
on how the runner lands. cadence-explained.md has the detail — it is one of the
reasons intervals are timed on mid-stance instead.

See cadence-explained.md for the method, and config.py for what each default is
doing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .skeleton import KPT_NAMES

_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}

FEET = ("left", "right")


# ── signal helpers ───────────────────────────────────────────────────────────
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


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """Fill NaN gaps by linear interpolation, leaving an all-NaN array alone."""
    valid = np.isfinite(x)
    if valid.all() or not valid.any():
        return x
    idx = np.arange(len(x))
    out = x.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return out


def joint_series(frames: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` of one joint on the first tracked person, per returned frame.

    (0, 0) is the model's "not visible" sentinel, so it becomes NaN rather than
    a real point at the top-left corner.
    """
    j = _INDEX[name]
    xs = np.full(len(frames), np.nan)
    ys = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        persons = frame.get("persons") or []
        if not persons:
            continue
        kpts = persons[0].get("kpts_xy") or []
        if j >= len(kpts) or tuple(kpts[j]) == (0, 0):
            continue
        xs[i], ys[i] = float(kpts[j][0]), float(kpts[j][1])
    return xs, ys


def _densify(values: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    """Put a per-returned-frame series onto a per-source-frame grid.

    The panel is redrawn every source frame, so it wants to index a signal by
    frame number rather than by position in the response. Resampling once here
    is what lets every drawing routine be a plain array lookup.
    """
    out = np.full(n, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    grid = np.arange(n)
    out[:] = np.interp(grid, indices[finite], values[finite],
                       left=np.nan, right=np.nan)
    return out


def _robust_peak(signal: np.ndarray) -> float:
    """Typical swing height, as the median of the peaks rather than the max.

    Every threshold below is a fraction of this number, so one badly placed
    ankle must not be allowed to move all of them.
    """
    finite = signal[np.isfinite(signal)]
    if not finite.size:
        return 0.0
    top = float(np.max(finite))
    if top <= 0:
        return 0.0

    peaks = [
        signal[i]
        for i in range(1, len(signal) - 1)
        if np.isfinite(signal[i - 1:i + 2]).all()
        and signal[i] >= signal[i - 1]
        and signal[i] > signal[i + 1]
        and signal[i] > 0.4 * top
    ]
    if len(peaks) >= 3:
        return float(np.median(peaks))
    return float(np.percentile(finite, 95))


def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs as ``(start, end_exclusive, label)``."""
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, int(labels[start])))
            start = i
    return runs


def _cross_time(signal: np.ndarray, before: int, after: int, level: float,
                fps: float) -> float:
    """Sub-frame time at which *signal* crosses *level* between two frames.

    Foot strike is timed on the crossing rather than on the frame that follows
    it, because at 30 fps a whole frame is 33 ms against a ~350 ms step: taking
    the frame index alone quantizes every interval to that grid and the cadence
    inherits the staircase. The signal is steep at the crossing, which is
    exactly where linear interpolation is worth doing.
    """
    v0, v1 = signal[before], signal[after]
    if not (np.isfinite(v0) and np.isfinite(v1)) or v1 == v0:
        return after / fps
    frac = float(np.clip((level - v0) / (v1 - v0), 0.0, 1.0))
    return (before + frac) / fps


def _joint_xy(frames: list[dict], name: str, indices: np.ndarray,
              n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One joint's ``(x, y)`` on the per-source-frame grid, plus an observed mask.

    Gaps are interpolated so the signal stays continuous, and **not** smoothed.
    An earlier version smoothed here to stop the knee landmark being picked as
    the noisiest low frame, which the parabola fit in _vertex now handles
    properly — and smoothing on top of it did real damage: it flattened the
    extremum the fit was locating (reading 20.9 degrees where the raw fit reads
    17.5) and it put the panel's label a few degrees away from the limb drawn
    beside it, since the number came from one version of the data and the
    picture from the other.

    The mask marks frames the model actually returned. An interpolated frame is
    not evidence and must not be chosen as a landmark, which is why it is kept
    rather than inferred from NaN after the fill.
    """
    xs, ys = joint_series(frames, name)
    dense_x = _densify(xs, indices, n)
    dense_y = _densify(ys, indices, n)
    observed = np.isfinite(dense_x) & np.isfinite(dense_y)
    return _interp_nan(dense_x), _interp_nan(dense_y), observed


def _leg_length(limb: dict) -> float:
    """The leg, hip to ankle, as the two rigid segments it is made of.

    Measured as median thigh + median shank rather than as any average of the
    hip-to-ankle distance, because the knee is bent for most of a running clip
    and an average of that distance is really an average of how bent it
    happened to be. On this clip it ran 8-11% short, which is the kind of error
    that shows: the two segments then summed to 1.09-1.14 of the "leg" they are
    the two halves of.

    A segment length is only a length at all once x and y are in the same unit
    (see analyze), and once they are it does not depend on the limb's
    orientation — which is what makes a median of it meaningful rather than a
    mixture over the stride.

    One figure for the clip, so the two legs are averaged. They differ by about
    5% here and that is the camera: the near leg is closer to the lens and
    projects larger. Neither is more correct, and using each foot's own would
    make the two legs' numbers incomparable, which is the opposite of what the
    unit is for.
    """
    per_leg = []
    for joints in limb.values():
        total = 0.0
        for a_, b_ in (("hip", "knee"), ("knee", "ankle")):
            d = np.hypot(joints[b_][0] - joints[a_][0], joints[b_][1] - joints[a_][1])
            d = d[np.isfinite(d)]
            if not d.size:
                total = np.nan
                break
            total += float(np.median(d))
        if np.isfinite(total):
            per_leg.append(total)
    return float(np.mean(per_leg)) if per_leg else 0.0


def _knee_flexion(hip, knee, ankle) -> np.ndarray:
    """Knee flexion in degrees, per frame. 0 is a straight leg.

    Sampled for the panel at its own local minimum near each strike rather than
    at the strike itself; see the note in analyze().

    The interior angle at the knee, between the thigh (knee -> hip) and the
    shank (knee -> ankle), subtracted from 180. A fully extended leg puts those
    two vectors 180 apart and reads 0 flexion; a bent knee reads positive.

    This is the projected angle in the image plane, not the anatomical one. Seen
    from the side that is very nearly the sagittal knee angle, which is the one
    worth having — but a leg swinging toward or away from the camera foreshortens
    and reads flatter than it is.

    The coordinates must already be in one unit on both axes; analyze() puts
    them there. Fed the model's raw normalized x, this measures the angle in a
    space stretched by the frame's aspect ratio, which inflates every deviation
    from vertical — and flexion is exactly such a deviation, bounded below at
    zero, so it can only read high. On this clip that was worth +10 degrees at
    foot strike and +25 at the swing peak.
    """
    v1 = np.vstack([hip[0] - knee[0], hip[1] - knee[1]])
    v2 = np.vstack([ankle[0] - knee[0], ankle[1] - knee[1]])
    n1, n2 = np.hypot(*v1), np.hypot(*v2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1 * v2).sum(axis=0) / (n1 * n2)
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _heading(limb: dict, planted: dict, nose_x: np.ndarray) -> tuple[float, float]:
    """Which way the runner is *travelling*, in image x. +1 is towards +x.

    Returned as ``(heading, facing)``, both -1, +1 or 0 when undecidable.

    Measured off the planted foot, not off the runner's displacement: on a
    treadmill they do not go anywhere, so displacement is zero and would decide
    nothing. What holds either way is that a foot in contact with the ground is
    the part of the runner that is *not* moving with them — so relative to the
    hips it travels backwards through every stance, on a belt or on a road.
    Averaged over every planted frame of both feet, the sign of that is the
    direction of travel reversed.

    `facing` is the cross-check, from the nose against the mid-hip. It is
    anatomy rather than motion, so it is only the fallback and the disagreement
    flag: a runner backwards on a belt, or a clip where the model has the body
    turned around, is exactly the case where the two differ.
    """
    drift = []
    for foot, joints in limb.items():
        ankle_x, hip_x = joints["ankle"][0], joints["hip"][0]
        rel = np.asarray(ankle_x) - np.asarray(hip_x)
        stance = np.asarray(planted[foot])
        # Both ends of a step have to be in contact for the difference to be a
        # stance movement rather than the jump into or out of one.
        keep = stance[1:] & stance[:-1]
        d = np.diff(rel)[keep]
        drift += [v for v in d if np.isfinite(v)]

    heading = -float(np.sign(np.mean(drift))) if drift else 0.0

    hips = [np.asarray(joints["hip"][0]) for joints in limb.values()]
    mid_hip_x = np.nanmean(np.vstack(hips), axis=0) if hips else np.zeros(0)
    lead = np.asarray(nose_x) - mid_hip_x
    lead = lead[np.isfinite(lead)]
    facing = float(np.sign(np.mean(lead))) if lead.size else 0.0
    return heading or facing, facing


def _vertex(series: np.ndarray, centre: int, reach: int) -> tuple[float, float]:
    """Sub-frame minimum of *series* near *centre*, as ``(frame, value)``.

    A least-squares parabola over the window, taking its vertex. Every point in
    the window contributes, so the noise averages down instead of being selected
    for — which is the whole problem with taking the lowest frame.

    The two obvious alternatives are both biased, in opposite directions:

    * ``argmin`` of the raw series picks the noisiest low sample. With this
      clip's 4.8 deg of per-frame flexion jitter that is worth about -5 deg.
    * ``argmin`` of a smoothed series removes that, but box-smoothing flattens
      an extremum, so it reads about +3 deg high.

    Measured on this clip the three give 13.8 / 19.5 / 17.5 degrees for the same
    left knee. The parabola is the one with no thumb on the scale.
    """
    lo = max(0, centre - reach)
    hi = min(len(series), centre + reach + 1)
    window = series[lo:hi]
    if len(window) < 5 or not np.isfinite(window).all():
        finite = [i for i in range(lo, hi) if np.isfinite(series[i])]
        if not finite:
            return float(centre), float("nan")
        best = min(finite, key=lambda i: series[i])
        return float(best), float(series[best])

    t = np.arange(len(window)) - (len(window) - 1) / 2.0
    a2, a1, a0 = np.polyfit(t, window, 2)
    if a2 <= 0:                      # not a minimum: fall back to the lowest
        j = int(np.argmin(window))
        return float(lo + j), float(window[j])
    peak = -a1 / (2.0 * a2)
    peak = float(np.clip(peak, t[0], t[-1]))
    return (lo + (len(window) - 1) / 2.0 + peak,
            float(a2 * peak ** 2 + a1 * peak + a0))


def _segment_angle(a, b) -> np.ndarray:
    """Direction from *a* to *b* in radians, in image coordinates (y down)."""
    return np.arctan2(b[1] - a[1], b[0] - a[0])


def _circular_mean(angles: np.ndarray) -> float:
    """Mean of angles, via the unit vectors — so 359 and 1 average to 0."""
    finite = angles[np.isfinite(angles)]
    if not finite.size:
        return float("nan")
    return float(np.arctan2(np.mean(np.sin(finite)), np.mean(np.cos(finite))))


def _circular_sd(angles: np.ndarray) -> float:
    """Spread of angles in radians, from the length of the mean resultant.

    ``sqrt(-2 ln R)``, which for a tight cluster agrees with the ordinary
    standard deviation and, unlike it, cannot be thrown by the wrap at ±pi.
    """
    finite = angles[np.isfinite(angles)]
    if finite.size < 2:
        return 0.0
    r = np.hypot(np.mean(np.cos(finite)), np.mean(np.sin(finite)))
    return float(np.sqrt(-2.0 * np.log(max(r, 1e-12))))


@dataclass
class KneeShape:
    """The mean hip-knee-ankle geometry at foot strike, for one leg.

    Held as two segment *directions* plus two lengths rather than as three
    averaged points. Averaging the points directly shortens the limb whenever
    the orientation varies — the mean of two opposite positions is the midpoint
    — which would draw a leg that shrinks the more inconsistent the runner is.
    Averaging directions and taking the median lengths keeps the limb the right
    size and puts all the variation where it belongs, in the angles.
    """

    foot: str
    n: int
    thigh_angle: float      # radians, image coordinates
    shank_angle: float
    thigh_length: float     # leg lengths
    shank_length: float
    flexion: float          # degrees, 0 = straight
    flexion_sd: float       # spread of the knee angle across those strikes
    thigh_sd: float = 0.0   # spread of the whole limb's orientation, radians
    bend_sign: float = 1.0  # which way the shank turns off the thigh

    def as_dict(self) -> dict:
        return {
            "foot": self.foot, "strikes_averaged": self.n,
            "flexion_deg": _round(self.flexion, 2),
            "flexion_sd_deg": _round(self.flexion_sd, 2),
            "thigh_deg": _round(np.degrees(self.thigh_angle), 2),
            "shank_deg": _round(np.degrees(self.shank_angle), 2),
            "thigh_deg_sd": _round(np.degrees(self.thigh_sd), 2),
            "thigh_length_legs": _round(self.thigh_length, 4),
            "shank_length_legs": _round(self.shank_length, 4),
        }


# ── events ───────────────────────────────────────────────────────────────────
@dataclass
class Step:
    """One foot strike, and the swing that preceded it.

    Two landmarks, because no single one does both jobs. ``time`` is the strike
    — the instant the foot lands, which is what the overlay flashes on and what
    the panel dots. ``mid_stance`` is the centre of the same stance, which is
    what every *interval* is measured between. See cadence-explained.md; the
    short version is that a threshold crossing is where the signal is steep and
    therefore precise against noise, but it also slides when the threshold is
    slightly wrong — and it slides by different amounts on the near and the far
    leg, which a side-on camera guarantees. The centre of a stance run does not
    move when the threshold is wrong, because the run grows at both ends.
    """

    number: int
    foot: str
    time: float                     # foot strike, seconds
    mid_stance: float               # centre of the stance; intervals are timed here
    frame: int                      # nearest source frame, for the overlay flash
    peak_time: float                # mid-swing, the highest the foot got
    peak_clearance: float           # in leg lengths
    contact_seconds: float          # strike to toe-off; a proxy, see the docs
    knee_frame: int = 0                  # nearest frame, for gating and reporting
    knee_time: float = 0.0               # sub-frame instant the shape is read at
    knee_offset: float = 0.0             # seconds from the strike to that instant
    knee_flexion: float = float("nan")   # degrees there
    interval: float = float("nan")  # since the previous step, either foot
    stride: float = float("nan")    # since the previous step, same foot

    def as_dict(self) -> dict:
        return {
            "number": self.number,
            "foot": self.foot,
            "time": round(self.time, 4),
            "mid_stance": round(self.mid_stance, 4),
            "peak_time": round(self.peak_time, 4),
            "peak_clearance": round(self.peak_clearance, 4),
            "contact_seconds": round(self.contact_seconds, 4),
            "knee_flexion_deg": None if not np.isfinite(self.knee_flexion)
            else round(self.knee_flexion, 2),
            "knee_sample_offset_ms": round(self.knee_offset * 1000.0, 1),
            "step_interval": None if not np.isfinite(self.interval)
            else round(self.interval, 4),
            "stride_seconds": None if not np.isfinite(self.stride)
            else round(self.stride, 4),
        }


@dataclass
class GaitAnalysis:
    """Everything the panel, the plots and the report read from."""

    fps: float
    n_frames: int
    clearance: dict[str, np.ndarray]      # per foot, per source frame, leg lengths
    raw: dict[str, np.ndarray]            # the same before smoothing
    amplitude: dict[str, float]           # typical swing height per foot
    contact_level: float                  # the threshold a strike is timed on
    leg_length: float                     # normalized frame units
    ground: dict[str, float]              # each foot's own belt line, normalized y
    hip_height: np.ndarray                # mid-hip, leg lengths above ground
    knee_flexion: dict = field(default_factory=dict)   # per foot, per frame, degrees
    limb: dict = field(default_factory=dict)           # per foot: hip/knee/ankle x,y
    steps: list[Step] = field(default_factory=list)
    cadence_live: np.ndarray = field(default_factory=lambda: np.zeros(0))
    foot_cadence_live: dict = field(default_factory=dict)   # per foot, per frame
    _smoothing: float = 0.2      # EMA weight the live trace was built with
    _foot_window: int = 2        # trailing window the per-foot series used
    warnings: dict = field(default_factory=dict)   # silent failure modes, counted
    airborne: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    # Direction of travel in image x, +1 towards +x. See _heading.
    heading: float = 0.0
    facing: float = 0.0
    # The source frame's width / height. `limb` is already scaled by it, so
    # nothing downstream needs to apply it again; it is here to be recorded.
    frame_aspect: float = 1.0

    # ── counts and headline numbers ──────────────────────────────────────────
    @property
    def count(self) -> int:
        return len(self.steps)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / (self.fps or 30.0)

    @property
    def intervals(self) -> list[float]:
        return [s.interval for s in self.steps if np.isfinite(s.interval)]

    def mean_cadence_by(self, until: float | None = None) -> float:
        """Steps per minute over the strides completed by *until*.

        Derived from the stride rather than from the step, because a stride is
        one foot's landmark to that same foot's next one: any offset in *where*
        on the stance the landmark sits cancels exactly, appearing at both ends
        of the subtraction. Two steps to a stride, hence the 120.

        Timing it step to step looks equivalent and is not — the residual
        left/right phase offset (~15 ms here, and mostly the camera rather than
        the runner) alternates, so a clip opening on one foot and closing on the
        other keeps half of it. That alone moved a synthetic clip of known
        cadence by 0.6 spm.

        Not ``steps / elapsed`` either: a clip opens and closes mid-stride, so
        dividing by its full duration counts time no step was taken in and
        reports a cadence a few percent low.
        """
        strides = [s.stride for s in self.steps
                   if np.isfinite(s.stride)
                   and (until is None or s.mid_stance <= until)]
        if strides:
            return 120.0 / float(np.mean(strides))

        # Under two strides there is no same-foot pair to measure, so fall back
        # to first landmark to last over the steps between them.
        marks = [s.mid_stance for s in self.steps
                 if np.isfinite(s.mid_stance)
                 and (until is None or s.mid_stance <= until)]
        if len(marks) < 2:
            return float("nan")
        return 60.0 * (len(marks) - 1) / (marks[-1] - marks[0])

    @property
    def mean_cadence(self) -> float:
        """Steps per minute over the whole clip."""
        return self.mean_cadence_by(None)

    @property
    def mean_stride(self) -> float:
        strides = [s.stride for s in self.steps if np.isfinite(s.stride)]
        return float(np.mean(strides)) if strides else float("nan")

    @property
    def flight_ratio(self) -> float:
        """Share of the run with neither foot planted — the airborne phase.

        This is what separates running from walking: a walk always has a foot
        down, so it would read 0 here.
        """
        return float(np.mean(self.airborne)) if len(self.airborne) else float("nan")

    def stride_spread(self, until: float | None = None) -> float:
        """Stride-to-stride variability, in seconds. How metronomic the run was.

        Every standard deviation in this module is a *sample* one (``ddof=1``).
        They were mixed before — this one corrected, the step spread and the
        knee spread not — which made a 22-sample figure read about 2% low
        against its neighbour for no reason a reader could see.

        Measured on the stride rather than on the step, and that is the whole
        point: a stride is one foot's landmark to that same foot's next one, so
        any per-foot offset in where the landmark sits cancels in the
        subtraction. The step interval does not have that property — the
        intervals alternate long/short, and on this clip that alternation alone
        contributes 7.4 ms of the 10.8 ms step spread, none of it the runner.

        Stride spread comes out 10.7 ms here, and the two feet agree on it
        (10.2 and 11.3 ms), which is what an uncontaminated measure looks like.
        """
        strides = [s.stride for s in self.steps
                   if np.isfinite(s.stride)
                   and (until is None or s.mid_stance <= until)]
        return float(np.std(strides, ddof=1)) if len(strides) > 1 else float("nan")

    def step_spread(self) -> float:
        """Standard deviation of the step interval — how metronomic the run was.

        Worth reporting only because the timing landmark is a stable one: on the
        strike crossing this clip reads 34 ms, on mid-stance 12 ms, and the
        difference is measurement rather than running. See cadence-explained.md.
        """
        intervals = self.intervals
        return float(np.std(intervals, ddof=1)) if len(intervals) > 1 else float("nan")

    def step_interval(self, foot: str, until: float | None = None) -> float:
        """Mean interval of the steps that *land* on this foot, in seconds.

        The step landing on the left foot is timed from the right foot's
        landmark to the left's. That is the quantity asymmetry lives in: whether
        one foot comes down halfway between the other's, or early, or late.

        Deliberately not the per-foot *stride*. A stride is one foot's landmark
        to that same foot's next one, so both feet complete the same cycles over
        the same clip and their rates are pinned together by arithmetic — 179.4
        against 179.6 here, and that would hold however unevenly someone ran.

        *until* bounds it to steps already landed, for the live panel.
        """
        values = [s.interval for s in self.steps
                  if s.foot == foot and np.isfinite(s.interval)
                  and (until is None or s.mid_stance <= until)]
        return float(np.mean(values)) if values else float("nan")

    def foot_stride_rate(self, foot: str, until: float | None = None) -> float:
        """How often this foot actually lands, as a steps-per-minute equivalent.

        This is the *only* honest per-foot rate, and it is the same for both
        feet by construction — the feet alternate, so over any window each takes
        half the steps. On this clip 179.37 against 179.56, a difference of
        0.18 spm at t = 0.20. If these two ever differ meaningfully, something is
        wrong with the detection, not with the runner.
        """
        strides = [s.stride for s in self.steps_for(foot)
                   if np.isfinite(s.stride)
                   and (until is None or s.mid_stance <= until)]
        return 120.0 / float(np.mean(strides)) if strides else float("nan")

    def phase_split(self, until: float | None = None) -> tuple[float, float]:
        """The stride split into its two halves, as percentages summing to 100.

        This is what a left/right difference in this data really is, and it is
        one number rather than two: the interval before a left landing plus the
        interval before a right landing *is* the stride, so if one half grows
        the other shrinks by exactly as much.

        Reporting the two halves as two step rates in spm — which an earlier
        version of the panel did — reads as "the right foot takes more steps per
        minute than the left". That cannot happen. 50/50 means each foot lands
        exactly halfway between the other's landings.
        """
        left = self.step_interval("left", until)
        right = self.step_interval("right", until)
        total = left + right
        if not np.isfinite(total) or total <= 0:
            return float("nan"), float("nan")
        return 100.0 * left / total, 100.0 * right / total

    def symmetry(self, until: float | None = None) -> float:
        """How even the stride's two halves are, as a percentage.

        **Not resolvable from a single side-on camera. Kept for the record, and
        deliberately absent from the panel.** Read cadence-explained.md before
        using it for anything.

        A *phase* measure, not a rate one. Both feet land at the same rate — see
        foot_stride_rate — so this asks whether each foot comes down halfway
        between the other's landings, or early, or late.

        The reason it does not survive: a real phase shift is a constant time
        offset, so it must read the same wherever in the gait cycle it is
        measured. Swept around the cycle on this clip it runs from -41 ms to
        +69 ms and changes sign — 110 ms of spread on a 15 ms result.

        And the strikes are in fact evenly spaced. A foot strike is a physical
        event, so no camera distance can move it; what moves is the *landmark*,
        which precedes touchdown by (threshold - touchdown height) / descent
        speed, and the two legs descend at different speeds. Sweeping the
        threshold, the alternation passes through zero at about 23% of swing
        amplitude, where the two feet come out even to 0.2 +/- 2.9 ms. One free
        parameter erases the whole asymmetry, so it carried no information about
        the runner. The threshold is not moved there because at that height the
        airborne fraction collapses to ~20% and then 8%, which no runner does;
        see cadence-explained.md.

        The shorter over the longer, so 100 means the feet land exactly evenly
        spaced and there is no direction to the number — which is right, since
        "the left leads by 4%" and "the right trails by 4%" describe one thing.

        **Read the floor before reading the value.** A single side-on camera
        cannot resolve this well: the two ankles descend through the strike
        threshold at different rates (0.80 leg lengths a second on the right,
        0.53 on the left), so a fixed landmark sits at a different point in each
        leg's cycle. Whether that difference is the runner or the view of them
        is not separable from one camera — but either way the *timing* reading
        is invalid, which is what the landmark sweep in cadence-explained.md
        shows.
        """
        left = self.step_interval("left", until)
        right = self.step_interval("right", until)
        if not (np.isfinite(left) and np.isfinite(right)) or min(left, right) <= 0:
            return float("nan")
        return 100.0 * min(left, right) / max(left, right)

    def knee_shape(self, foot: str, until: float | None = None,
                   window: int | None = None) -> KneeShape | None:
        """Mean hip-knee-ankle geometry at this foot's strikes, or None.

        Averaged over the strikes that have already landed, so the shape fills
        in as the run goes rather than being the whole clip's answer shown from
        frame one.

        ``window`` averages only the most recent N strikes. A mean over *every*
        strike converges after a few strides and then sits perfectly still —
        1.0 degrees of movement across this clip — which reads as a frozen
        readout rather than a steady one. A trailing window keeps moving because
        it keeps forgetting: eight strikes gives 1.7 degrees of drift and is
        still an average, just over a stated span. The panel uses a window; the
        figures in gait.json and summary.txt use every strike.

        Every length is in leg lengths, so the drawing is the same size whatever
        the camera distance or export resolution. See KneeShape for why the
        directions are averaged rather than the joint positions.
        """
        if foot not in self.limb:
            return None
        # Gated on the frame the shape is *sampled* at, not on the strike. The
        # landmark can sit up to KNEE_SEARCH_SECONDS after the strike — on this
        # clip 15 of 45 do, by as much as 67 ms — so filtering on the strike let
        # the live panel average in geometry from frames that had not played
        # yet. Everything else on the panel is strictly causal; this was not.
        # Gated on the instant the shape is *sampled* at, not on the strike: the
        # landmark can sit after its own strike, so filtering on the strike let
        # the live panel average in geometry from frames that had not played yet.
        cut = np.inf if until is None else float(until)
        landed = [s for s in self.steps_for(foot)
                  if s.knee_time <= cut and np.isfinite(s.knee_flexion)]
        if window:
            landed = landed[-int(window):]
        if not landed:
            return None

        hip, knee, ankle = (self.limb[foot][j] for j in ("hip", "knee", "ankle"))
        grid = np.arange(self.n_frames)
        at = np.array([s.knee_time * (self.fps or 30.0) for s in landed])
        # Read at the sub-frame instant the vertex found, rather than snapped to
        # the nearest frame: the knee moves 150-420 deg/s through here, so half
        # a frame is several degrees of limb geometry.
        pick = lambda j: (np.interp(at, grid, j[0]),
                          np.interp(at, grid, j[1]))   # noqa: E731
        h, k, a = pick(hip), pick(knee), pick(ankle)

        thigh = _segment_angle(h, k)
        shank = _segment_angle(k, a)
        good = np.isfinite(thigh) & np.isfinite(shank)
        if not good.any():
            return None

        flex = np.array([s.knee_flexion for s in landed])
        flex = flex[np.isfinite(flex)]

        # The shank direction is *derived* from the reported flexion rather
        # than averaged independently, so the limb drawn and the number beside
        # it cannot disagree. Averaging both separately left them 1-2 degrees
        # apart: the label is a mean of fitted vertex values, the drawing was a
        # mean of measured directions, and nothing forces two different
        # averages of the same thing to coincide. A panel whose picture and
        # caption differ is worse than either.
        #
        # The thigh direction and both lengths are still measured; only the
        # combination is reconstructed. A straight leg (0 flexion) puts the
        # shank along the thigh, which is what collinear hip-knee-ankle means.
        thigh_mean = _circular_mean(thigh[good])
        shank_mean = _circular_mean(shank[good])
        flexion = float(np.mean(flex)) if flex.size else float("nan")
        bend_sign = 1.0
        if np.isfinite(flexion) and np.isfinite(thigh_mean):
            turn = np.radians(flexion)
            options = ((thigh_mean + turn, 1.0), (thigh_mean - turn, -1.0))
            shank_mean, bend_sign = min(options, key=lambda c: abs(np.arctan2(
                np.sin(c[0] - shank_mean), np.cos(c[0] - shank_mean))))

        return KneeShape(
            foot=foot,
            n=int(good.sum()),
            thigh_angle=thigh_mean,
            shank_angle=shank_mean,
            thigh_length=float(np.median(np.hypot(k[0] - h[0], k[1] - h[1])[good]))
            / self.leg_length,
            shank_length=float(np.median(np.hypot(a[0] - k[0], a[1] - k[1])[good]))
            / self.leg_length,
            flexion=flexion,
            flexion_sd=float(np.std(flex, ddof=1)) if flex.size > 1 else float("nan"),
            thigh_sd=_circular_sd(thigh[good]),
            bend_sign=bend_sign,
        )

    def segment_ratio(self, foot: str) -> float:
        """Measured shank length over thigh length, at foot strike.

        A keypoint-quality check. Hip-to-knee and knee-to-ankle are near-equal
        in humans — both about 0.245 of standing height — so this should read
        near 1.0. On this clip it gives 1.04 left and 1.04 right: both sane, and
        equal, so neither knee is misplaced relative to the other.

        This is the number that caught the coordinate space being wrong. While
        an x distance and a y distance were still in different units (see
        analyze) it read 0.93 and 0.91 at contact and 1.6-1.9 through mid-swing,
        and the swing figure was written off as the shank foreshortening as it
        points toward or away from the camera. It was mostly not: in one unit it
        reads 1.00 and 1.13 there. A shank cannot change length, so a "ratio"
        that moves through the stride is the measurement moving, not the leg.

        Measured **at the strike frames** rather than over a low-flexion
        percentile of the whole clip. That percentile was the first attempt and
        it was wrong: running has two knee-extension minima per stride, one at
        contact and one near toe-off, so the filter mixed two different phases
        and reported 0.93 against 1.33 — a difference that was entirely which
        frames each leg happened to contribute.

        Still read it near contact by preference. The residual through mid-swing
        (1.00 left, 1.13 right) is real projection — the swinging shank does
        turn out of the image plane — and near contact both legs are close to
        side-on, which is the phase this is a check on.
        """
        if foot not in self.limb:
            return float("nan")
        L = self.limb[foot]
        thigh = np.hypot(L["knee"][0] - L["hip"][0], L["knee"][1] - L["hip"][1])
        shank = np.hypot(L["ankle"][0] - L["knee"][0], L["ankle"][1] - L["knee"][1])
        at = np.array([s.knee_time * (self.fps or 30.0)
                       for s in self.steps_for(foot)])
        if not at.size:
            return float("nan")
        grid = np.arange(self.n_frames)
        ratio = np.interp(at, grid, shank / thigh)
        ratio = ratio[np.isfinite(ratio)]
        return float(np.median(ratio)) if ratio.size else float("nan")

    def peak_flexion(self, foot: str) -> float:
        """The most the knee bends anywhere in the cycle, in degrees.

        Mid-swing, not at contact — the heel comes up toward the glute — so it
        is a different measurement from the strike figure and much larger.
        """
        series = self.knee_flexion.get(foot)
        if series is None or not len(series):
            return float("nan")
        finite = series[np.isfinite(series)]
        return float(np.percentile(finite, 98)) if finite.size else float("nan")

    def steps_for(self, foot: str) -> list[Step]:
        return [s for s in self.steps if s.foot == foot]

    def contact_mean(self, foot: str) -> float:
        values = [s.contact_seconds for s in self.steps_for(foot)
                  if np.isfinite(s.contact_seconds)]
        return float(np.mean(values)) if values else float("nan")

    def balance(self) -> tuple[float, float]:
        """Left / right share of total contact time, as percentages.

        Even at a real asymmetry the two numbers sit within a point or two of
        50, which is why it is reported to one decimal: the interesting range is
        narrow, and rounding to whole percent throws the finding away.
        """
        left, right = self.contact_mean("left"), self.contact_mean("right")
        if not (np.isfinite(left) and np.isfinite(right)) or left + right <= 0:
            return float("nan"), float("nan")
        total = left + right
        return 100.0 * left / total, 100.0 * right / total

    def vertical_oscillation(self, until: float | None = None) -> float:
        """Median hip rise and fall within a stride, in leg lengths.

        Per stride rather than over the clip, because a runner drifting a little
        forward or back on the belt would otherwise be measured as bounce.

        *until* bounds it to strides already completed, which is what the live
        panel needs: the figure has to be the median of what has happened, not
        of the whole clip.
        """
        strides = [(a.mid_stance, b.mid_stance)
                   for foot in FEET
                   for a, b in zip(self.steps_for(foot), self.steps_for(foot)[1:])
                   if np.isfinite(a.mid_stance) and np.isfinite(b.mid_stance)
                   and (until is None or b.mid_stance <= until)]
        if not strides:
            return float("nan")
        fps = self.fps or 30.0
        spans = []
        for t0, t1 in strides:
            i0, i1 = int(round(t0 * fps)), int(round(t1 * fps))
            window = self.hip_height[max(0, i0):min(self.n_frames, i1 + 1)]
            window = window[np.isfinite(window)]
            if window.size > 2:
                spans.append(float(window.max() - window.min()))
        return float(np.median(spans)) if spans else float("nan")

    def cadence_at(self, frame: int) -> float:
        if not len(self.cadence_live) or frame >= len(self.cadence_live):
            return float("nan")
        return float(self.cadence_live[frame])

    def mean_samples(self) -> list[tuple[float, float]]:
        """The mean trace's vertices: one reading per stride. See
        _running_mean_samples."""
        return _running_mean_samples(self.steps)

    def running_cadence_series(self, foot: str | None = None) -> np.ndarray:
        """Per-frame running mean cadence, over every step landed so far.

        The alternative shape for the panel's traces. Where the trailing-window
        series answers "what is the cadence right now", this answers "what has
        the cadence averaged so far" — it converges instead of varying, so it
        settles into near-flat lines by mid-clip. Which one a viewer wants is a
        judgement call, so PANEL_TRACE_MODE picks.
        """
        samples = (_running_mean_samples(self.steps) if foot is None
                   else _foot_window_samples(self.steps,
                                             self._foot_window, foot))
        return _densify_samples(samples, self.fps, self.n_frames)

    def foot_cadence_at(self, foot: str, frame: int) -> float:
        """That foot's step rate at *frame*, over its own trailing window."""
        series = self.foot_cadence_live.get(foot)
        if series is None or not len(series) or frame >= len(series):
            return float("nan")
        return float(series[frame])

    def steps_by(self, frame: int) -> int:
        """Strikes landed at or before *frame*, for the live counter."""
        t = frame / (self.fps or 30.0)
        return sum(1 for s in self.steps if s.time <= t)

    # ── serialization ────────────────────────────────────────────────────────
    def summary(self) -> dict:
        left_pct, right_pct = self.balance()
        intervals = self.intervals
        return {
            "steps": self.count,
            "cadence_spm": _round(self.mean_cadence, 1),
            "mean_step_seconds": _round(float(np.mean(intervals)) if intervals
                                        else float("nan"), 4),
            # Step spread is inflated by the long/short alternation; stride
            # spread is the clean one. See stride_spread.
            "step_seconds_sd": _round(self.step_spread(), 4),
            "stride_seconds_sd": _round(self.stride_spread(), 4),
            "mean_stride_seconds": _round(self.mean_stride, 4),
            "step_seconds_per_foot": {foot: _round(self.step_interval(foot), 4)
                                      for foot in FEET},
            # The real per-foot rate, which is equal for both by construction.
            "stride_rate_spm_per_foot": {
                foot: _round(self.foot_stride_rate(foot), 2) for foot in FEET},
            "phase_split_percent": dict(zip(
                FEET, [_round(v, 1) for v in self.phase_split()])),
            "step_symmetry_percent": _round(self.symmetry(), 1),
            "contact_seconds": {foot: _round(self.contact_mean(foot), 4)
                                for foot in FEET},
            "contact_balance_percent": {"left": _round(left_pct, 1),
                                        "right": _round(right_pct, 1)},
            "flight_ratio": _round(self.flight_ratio, 4),
            # Image-space direction of travel, and the anatomical cross-check.
            # See _heading; the trails stream against `heading`.
            "heading_x": self.heading,
            "facing_x": self.facing,
            "peak_clearance_legs": {
                foot: _round(float(np.mean([s.peak_clearance
                                            for s in self.steps_for(foot)]))
                             if self.steps_for(foot) else float("nan"), 4)
                for foot in FEET
            },
            "vertical_oscillation_legs": _round(self.vertical_oscillation(), 4),
            "knee_flexion_at_strike_deg": {
                foot: {"mean": _round(shape.flexion, 2),
                       "sd": _round(shape.flexion_sd, 2),
                       "strikes": shape.n}
                for foot in FEET
                if (shape := self.knee_shape(foot)) is not None
            },
            "knee_peak_flexion_deg": {foot: _round(self.peak_flexion(foot), 2)
                                      for foot in FEET},
            # Keypoint-quality check: should be near 1.0. See segment_ratio.
            "shank_thigh_ratio_at_contact": {
                foot: _round(self.segment_ratio(foot), 3) for foot in FEET},
            "knee_shape_at_strike": {
                foot: shape.as_dict() for foot in FEET
                if (shape := self.knee_shape(foot)) is not None
            },
            "steps_per_foot": {foot: len(self.steps_for(foot)) for foot in FEET},
            "quality": self.warnings,
            "steps_detail": [s.as_dict() for s in self.steps],
        }

    def series(self) -> dict:
        """The full per-frame signals, for gait.json."""
        return {
            "time": [round(float(v), 4) for v in self.times],
            "clearance": {foot: [_round(float(v), 5) for v in self.clearance[foot]]
                          for foot in FEET},
            "hip_height": [_round(float(v), 5) for v in self.hip_height],
            "knee_flexion_deg": {
                foot: [_round(float(v), 3) for v in self.knee_flexion[foot]]
                for foot in FEET if foot in self.knee_flexion
            },
            "cadence_spm": [_round(float(v), 2) for v in self.cadence_live],
            # NOT that foot's cadence. It is 60 / the interval of the step that
            # *lands* on it, which is one half-stride and cannot be a rate --
            # the feet alternate, so both land equally often. It is the exact
            # quantity taken off the panel for implying otherwise, kept here
            # because the phase split is derived from it. See phase_split.
            "half_stride_rate_spm_by_foot": {
                foot: [_round(float(v), 2) for v in self.foot_cadence_live[foot]]
                for foot in FEET if foot in self.foot_cadence_live
            },
            "airborne": [bool(v) for v in self.airborne],
        }


def _round(value: float, places: int):
    """JSON has no NaN. None is what a missing measurement actually is."""
    return None if value is None or not np.isfinite(value) else round(float(value), places)


def signal_config(cfg) -> dict:
    """The settings that shaped the signal, snapshotted into gait.json."""
    return {
        "foot_reference": cfg.FOOT_REFERENCE,
        "per_foot_ground": cfg.PER_FOOT_GROUND,
        "ground_percentile": cfg.GROUND_PERCENTILE,
        "smooth_median_frames": cfg.SMOOTH_MEDIAN_FRAMES,
        "smooth_mean_frames": cfg.SMOOTH_MEAN_FRAMES,
        "swing_peak_fraction": cfg.SWING_PEAK_FRACTION,
        "contact_fraction": cfg.CONTACT_FRACTION,
        "min_contact_seconds": cfg.MIN_CONTACT_SECONDS,
        "cadence_smoothing_alpha": cfg.CADENCE_SMOOTHING,
        "cadence_foot_window_steps": cfg.CADENCE_FOOT_WINDOW_STEPS,
    }


# ── the analysis ─────────────────────────────────────────────────────────────
def analyze(frames: list[dict], fps: float, n_source_frames: int, *, cfg,
            aspect: float) -> GaitAnalysis:
    """Build both feet's clearance signals and find every foot strike.

    *aspect* is the source frame's width / height, and it is required rather
    than defaulted because there is no safe default: the model normalizes x by
    the frame width and y by its height, so on anything but a square frame an
    x distance and a y distance arrive in different units. Left mixed, every
    angle and every length in here is measured in a stretched space. See the
    note on `limb` below for what that was worth on this clip.
    """
    indices = np.array([f["index"] for f in frames], dtype=float)
    n = max(n_source_frames, 1)

    # Hip, knee and ankle, both coordinates, per leg. The clearance signal only
    # needs the y values, but the knee shape needs the whole limb.
    raw_limb = {
        foot: {joint: _joint_xy(frames, f"{foot}_{joint}", indices, n)
               for joint in ("hip", "knee", "ankle")}
        for foot in FEET
    }

    # x, put into the same unit as y — fractions of the frame *height* — so
    # that hypot and arctan2 mean something. Everything downstream reads the
    # limb from here (the knee flexion, the knee shape, the segment ratio, the
    # panel's knee block and its ankle view, knee.png), so this is the one
    # place it has to be right.
    #
    # It was wrong here for a while, and the test that settles it is that a
    # thigh and a shank are rigid: their length cannot depend on which way the
    # limb points. Measured on this clip, in the mixed space a tilted shank
    # came out 45% longer than an upright one and the length varied by 18-20%
    # across the clip; in this one it is 1-2% and 2-4%. That is not a
    # calibration question, it is the difference between measuring geometry and
    # measuring the frame's shape.
    scale_x = float(aspect)
    limb = {foot: {j: (v[0] * scale_x, v[1]) for j, v in joints.items()}
            for foot, joints in raw_limb.items()}
    # A frame counts as observed only if the whole limb was seen on it.
    observed = {
        foot: np.logical_and.reduce([raw_limb[foot][j][2]
                                     for j in ("hip", "knee", "ankle")])
        for foot in FEET
    }

    # The clearance signal keeps its own path from the *unsmoothed* ankle, so
    # none of the strike thresholds shift under this change.
    ankle_y = {foot: _densify(joint_series(frames, f"{foot}_ankle")[1], indices, n)
               for foot in FEET}
    hip_y = {foot: _densify(joint_series(frames, f"{foot}_hip")[1], indices, n)
             for foot in FEET}

    # One scale for the whole clip: the runner's own leg. Every number
    # downstream is expressed in it, so nothing depends on how far away the
    # camera stood or what the export height is.
    leg_length = _leg_length(limb)
    if leg_length <= 1e-6:
        raise RuntimeError(
            "Could not measure a leg length: the hips or ankles were never "
            "detected. Check poses.json, and that the runner is in frame."
        )

    # The belt line, as a high percentile of ankle height rather than its
    # maximum: the lowest single frame of the clip is a bad pose as often as it
    # is the ground.
    #
    # Per foot by default, and that is not a detail. The two ankles do not
    # project to the same height while standing on the same belt: measured on
    # this clip the right leg is the near one (it projects 3-6% larger) and the
    # left ankle's stance level sits 1.0% of a leg length *higher* in frame.
    # Share one line between them and the lower foot spends longer under the
    # strike threshold for no reason to do with the runner, which lands as a
    # 42/58 contact balance. Zeroing each foot on its own stance costs nothing
    # and is what makes the two comparable at all; PER_FOOT_GROUND = False
    # shares one line, right only if both ankles are the same distance away.
    if cfg.PER_FOOT_GROUND:
        ground = {foot: float(np.percentile(ankle_y[foot][np.isfinite(ankle_y[foot])],
                                            cfg.GROUND_PERCENTILE))
                  for foot in FEET}
    else:
        stacked = np.concatenate([ankle_y[foot][np.isfinite(ankle_y[foot])]
                                  for foot in FEET])
        shared = float(np.percentile(stacked, cfg.GROUND_PERCENTILE))
        ground = {foot: shared for foot in FEET}

    mid_hip = np.nanmean(np.vstack([hip_y[foot] for foot in FEET]), axis=0)

    raw, clearance, amplitude = {}, {}, {}
    for foot in FEET:
        # y grows downward, so subtracting from the ground line turns image
        # coordinates into height above it.
        series = (ground[foot] - ankle_y[foot]) / leg_length
        if cfg.FOOT_REFERENCE == "hip":
            # For a camera that moves: measure the foot against this frame's own
            # hip instead of a fixed line, then re-zero on the planted foot.
            series = (mid_hip - ankle_y[foot]) / leg_length
            floor = np.nanpercentile(series, 100 - cfg.GROUND_PERCENTILE)
            series = series - floor
        raw[foot] = series
        filled = _interp_nan(series)
        smooth = _box_filter(_median_filter(filled, cfg.SMOOTH_MEDIAN_FRAMES),
                             cfg.SMOOTH_MEAN_FRAMES)
        smooth[~np.isfinite(series) & ~np.isfinite(filled)] = np.nan
        clearance[foot] = smooth
        amplitude[foot] = _robust_peak(smooth)

    # One threshold for both feet, set on the shorter swing of the two. The
    # *zero* is per foot, because the two ankles sit at different heights for
    # reasons of perspective; the threshold above it is shared, because a strike
    # should mean the same clearance on either leg. Give each foot its own
    # threshold as well and the contact balance measures the thresholds.
    swing = min(v for v in amplitude.values() if v > 0) if any(
        v > 0 for v in amplitude.values()) else 0.0
    contact_level = cfg.CONTACT_FRACTION * swing
    high = cfg.SWING_PEAK_FRACTION * swing

    strikes: list[Step] = []
    for foot in FEET:
        strikes += _strikes(clearance[foot], foot, fps, high=high, low=contact_level,
                            min_frames=max(1, int(round(
                                cfg.MIN_CONTACT_SECONDS * fps))))
    # Sorted on the strike, which every step has. Sorting on mid_stance put a
    # NaN-timed step at an arbitrary position; the two orders are otherwise
    # identical here, since the per-leg mid-stance offsets are ~25 ms against a
    # 334 ms step.
    strikes.sort(key=lambda s: s.time)

    last_by_foot: dict[str, float] = {}
    for i, step in enumerate(strikes):
        step.number = i + 1
        step.frame = int(min(n - 1, round(step.time * fps)))
        if not np.isfinite(step.mid_stance):
            continue          # untimed: no interval out of it, and none into it
        if i and np.isfinite(strikes[i - 1].mid_stance):
            step.interval = step.mid_stance - strikes[i - 1].mid_stance
        if step.foot in last_by_foot:
            step.stride = step.mid_stance - last_by_foot[step.foot]
        last_by_foot[step.foot] = step.mid_stance

    planted = {foot: np.isfinite(clearance[foot]) & (clearance[foot] < contact_level)
               for foot in FEET}
    airborne = np.ones(n, dtype=bool)
    for foot in FEET:
        airborne &= ~planted[foot]

    # Which way the runner is going, so the trails can stream the way the air
    # would. Derived per clip; nothing downstream may hardcode a side.
    heading, facing = _heading(limb, planted,
                               _densify(joint_series(frames, "nose")[0], indices, n))
    # Not an error to correct, and not corrected: a disagreement is the one
    # case where motion and anatomy genuinely differ -- a runner backwards on a
    # belt, or a body the model has turned around -- and the reader should know
    # which of the two the drawing followed. It follows the motion.
    heading_conflict = bool(heading and facing and heading != facing)

    knee_flexion = {
        foot: _knee_flexion(limb[foot]["hip"], limb[foot]["knee"],
                            limb[foot]["ankle"])
        for foot in FEET
    }

    knee_flexion_raw = knee_flexion
    # Where to sample the knee. Not at the strike frame, which is what this
    # first did and it was wrong for a measurable reason: the knee reaches its
    # most extended point within a frame of contact and then flexes hard for the
    # loading response, at 250-450 deg/s. Sampling on that *slope* means one
    # frame of timing costs 8-15 deg, and the two legs happened to land on
    # opposite sides of their own turning point (left 20 ms past it, right 30 ms
    # before), so the bias did not even cancel between them.
    #
    # The turning point itself is an extremum, so the rate is zero there and a
    # frame of timing error costs almost nothing. Same reasoning that put the
    # step intervals on mid-stance: prefer the landmark whose value is flat
    # where you measure it.
    reach = max(1, int(round(cfg.KNEE_SEARCH_SECONDS * fps)))
    edge_hits = 0
    for step in strikes:
        series = knee_flexion[step.foot]
        seen = observed[step.foot]
        step.knee_frame = step.frame
        step.knee_time = step.frame / fps
        step.knee_offset = 0.0

        # Skip a window that leans on interpolated frames: an interpolated frame
        # is not evidence and must not shape a landmark.
        lo = max(0, step.frame - reach)
        hi = min(len(series), step.frame + reach + 1)
        if not seen[lo:hi].all():
            continue

        at, value = _vertex(series, step.frame, reach)
        # A vertex pinned to the window edge means the real minimum is probably
        # outside it, which is a window too narrow rather than a measurement.
        if abs(at - step.frame) >= reach - 1e-6:
            edge_hits += 1
        step.knee_time = at / fps
        step.knee_frame = int(np.clip(round(at), 0, n - 1))
        step.knee_offset = (at - step.frame) / fps
        step.knee_flexion = value

    # Two failure modes that would otherwise pass silently.
    repeats = sum(1 for a_, b_ in zip(strikes, strikes[1:]) if a_.foot == b_.foot)
    warnings = {
        "untimed_strikes": sum(1 for s in strikes
                               if not np.isfinite(s.mid_stance)),
        "consecutive_same_foot_strikes": repeats,
        "knee_landmark_on_window_edge": edge_hits,
        "implausible_stance_flexion_frames": _implausible_stance(
            knee_flexion_raw, strikes, fps, cfg.KNEE_STANCE_LIMIT_DEG),
        "frames_interpolated": {
            foot: int(n - int(observed[foot].sum())) for foot in FEET},
        "heading_disagrees_with_facing": int(heading_conflict),
    }

    return GaitAnalysis(
        fps=fps, n_frames=n,
        clearance=clearance, raw=raw, amplitude=amplitude,
        contact_level=contact_level, leg_length=leg_length, ground=ground,
        # The hips get the mean of the two lines. There is only one belt, and
        # the per-foot split above is a correction for where each ankle was
        # seen from, not two different floors.
        hip_height=(float(np.mean(list(ground.values()))) - mid_hip) / leg_length,
        knee_flexion=knee_flexion,
        limb=limb,
        steps=strikes,
        cadence_live=_densify_samples(
            _stride_ema_samples(strikes, cfg.CADENCE_SMOOTHING), fps, n),
        foot_cadence_live={
            foot: _densify_samples(
                _foot_window_samples(strikes, cfg.CADENCE_FOOT_WINDOW_STEPS, foot),
                fps, n)
            for foot in FEET
        },
        _smoothing=cfg.CADENCE_SMOOTHING,
        _foot_window=cfg.CADENCE_FOOT_WINDOW_STEPS,
        warnings=warnings,
        airborne=airborne,
        heading=heading,
        facing=facing,
        frame_aspect=scale_x,
    )


def _strikes(signal: np.ndarray, foot: str, fps: float, *, high: float, low: float,
             min_frames: int) -> list[Step]:
    """Every foot strike on one foot's clearance signal.

    Stance and swing are found as runs either side of one threshold, then two
    passes clean up what a plain threshold gets wrong:

    * a "swing" that never reaches *high* was the ankle wobbling at the bottom
      of stance, so it is absorbed back into the stance around it;
    * a "stance" shorter than *min_frames* was the signal clipping the
      threshold mid-swing, so it is absorbed into the swing.

    Without the first pass a single planted foot reports two or three strikes;
    without the second, one swing reports two.
    """
    if high <= 0 or len(signal) < 3:
        return []

    finite = np.isfinite(signal)
    labels = (finite & (signal < low)).astype(int)   # 1 = planted

    for _ in range(2):
        changed = False
        for start, end, label in _segments(labels):
            window = signal[start:end]
            window = window[np.isfinite(window)]
            if label == 0 and (not window.size or float(window.max()) < high):
                labels[start:end] = 1
                changed = True
            elif label == 1 and end - start < min_frames:
                labels[start:end] = 0
                changed = True
        if not changed:
            break

    steps: list[Step] = []
    stances = [(a, b) for a, b, label in _segments(labels) if label == 1]
    for start, end in stances:
        if start == 0:
            continue   # the clip opened with this foot already down; no strike to time
        strike = _cross_time(signal, start - 1, start, low, fps)
        toe_off = (_cross_time(signal, end - 1, end, low, fps)
                   if end < len(signal) else float("nan"))

        # The swing that led into this strike, for the peak clearance the panel
        # reports. Walking back from the strike rather than forward from the
        # previous toe-off, so a clip that opens mid-swing still gets one.
        prev_end = max((b for a, b in stances if b <= start), default=0)
        window = signal[prev_end:start]
        window = window[np.isfinite(window)]
        peak = float(window.max()) if window.size else float("nan")
        peak_i = prev_end + int(np.nanargmax(signal[prev_end:start])) \
            if window.size else start

        # Mid-stance from the two *interpolated* crossings, not from the run's
        # integer bounds. Averaging is what makes it robust — a threshold set
        # slightly too high moves the entry earlier and the exit later, so the
        # error largely cancels in the mean — and interpolating first is what
        # keeps it off the frame grid. The integer midpoint has the same
        # robustness but quantizes to half a frame, which on a 0.33s step is
        # 16 ms of avoidable spread.
        #
        # With no toe-off there is no mid-stance. This used to fall back to the
        # integer midpoint of the run, which for the clip's final stance is the
        # midpoint of a stance *cut short by the recording* — a landmark placed
        # early by however much of the stance was never filmed, and it fed
        # straight into the last stride and step interval. NaN instead, and
        # every timing consumer skips it: the strike still happened and still
        # counts, it just cannot be timed against.
        mid = (strike + toe_off) / 2 if np.isfinite(toe_off) else float("nan")

        steps.append(Step(
            number=0, foot=foot, time=strike, frame=0,
            mid_stance=mid,
            peak_time=peak_i / fps, peak_clearance=peak,
            contact_seconds=toe_off - strike if np.isfinite(toe_off) else float("nan"),
        ))
    return steps


def _implausible_stance(knee_flexion: dict, steps: list, fps: float,
                        limit: float) -> int:
    """Frames inside a detected ground contact where the knee is impossibly bent.

    Run on the **unsmoothed** flexion. A planted knee does not exceed about 70
    degrees; anything near the swing peak (100-120) while the foot is supposed
    to be on the ground is the pose, not the runner. Smoothing would defeat the
    check -- it pulled this clip's 113-degree spike down to 76. On this clip it catches the stretch where ViTPose
    has swapped the left and right leg keypoints as the legs scissor past each
    other in the image.

    This replaced a frame-to-frame swap detector, which was the textbook test
    and did not work here: it compares each frame's assignment against the
    previous frame's, so a swap *sustained* over several frames looks perfectly
    continuous from the inside, and at the transition the two legs are close
    enough together that both assignments cost about the same. It returned zero
    on a clip with a known swap, which is worse than not testing -- a detector
    that cannot fail is indistinguishable from a clean result.

    Counted, not repaired: rewriting a model's output on a heuristic risks
    replacing a visible problem with an invisible one.
    """
    bad = 0
    for step in steps:
        if not np.isfinite(step.contact_seconds):
            continue
        series = knee_flexion.get(step.foot)
        if series is None:
            continue
        lo = int(round(step.time * fps))
        hi = int(round((step.time + step.contact_seconds) * fps))
        window = series[max(0, lo):min(len(series), hi + 1)]
        bad += int(np.sum(np.isfinite(window) & (window > limit)))
    return bad


def _stride_ema_samples(steps: list[Step], alpha: float) -> list[tuple[float, float]]:
    """The live cadence reading, once per step. ``(time, spm)``.

    Two choices here, and both are about how the trace *looks* without lying
    about what was measured.

    **Overlapping strides, not steps.** The interval measured at step i is
    ``marks[i] - marks[i-2]`` — a whole stride, ending at that step. A stride is
    phase-balanced by construction, so the left/right alternation cancels in
    every single sample rather than having to be averaged out over an even
    window. It still updates once a step, so nothing is lost in resolution.

    **An exponential average, not a boxcar.** A trailing window of k samples
    lurches twice for every reading — once when a value enters and again when it
    leaves k steps later — and that second lurch is pure rendering artifact,
    since nothing happened at that moment. On this clip a 6-step boxcar left
    0.93 spm of step-to-step jitter, which on the panel's axis is a 32px zigzag
    across 11px of horizontal spacing. The same effective smoothing as an
    exponential average gives 0.44 spm, because it responds on entry and then
    decays smoothly.

    ``alpha`` is the weight on the newest stride; the effective window is about
    ``2 / alpha - 1`` steps.
    """
    marks = [s.mid_stance for s in steps]
    if len(marks) < 3:
        return []
    out: list[tuple[float, float]] = []
    average = None
    for i in range(2, len(marks)):
        if not np.isfinite(marks[i - 2:i + 1]).all():
            continue      # an untimed step in the window: no stride to read
        stride = marks[i] - marks[i - 2]
        average = stride if average is None else \
            alpha * stride + (1.0 - alpha) * average
        out.append((marks[i], 120.0 / average))
    return out


def _running_mean_samples(steps: list[Step]) -> list[tuple[float, float]]:
    """The mean cadence so far, once per stride. ``(time, spm)``.

    Sampled only at an **even** number of step intervals. Two steps make one
    stride, so an even count is a whole number of strides and the left/right
    phase offset cancels; an odd count carries half of it. Skipping that put a
    7 spm dive in the first second of the trace — a mean over a single
    half-stride — and a small odd/even wobble on it thereafter.
    """
    marks = [s.mid_stance for s in steps]
    out: list[tuple[float, float]] = []
    for i in range(2, len(marks), 2):
        if not (np.isfinite(marks[i]) and np.isfinite(marks[0])):
            continue
        out.append((marks[i], 60.0 * i / (marks[i] - marks[0])))
    return out


def _foot_window_samples(steps: list[Step], window: int,
                         foot: str) -> list[tuple[float, float]]:
    """One foot's step rate over a trailing window of its own steps.

    Kept for `gait.json` and `clearance.png` only — it is not on the panel,
    because the feet alternate and neither can have a higher rate than the
    other. See GaitAnalysis.phase_split.
    """
    marks = [s.mid_stance for s in steps]
    out: list[tuple[float, float]] = []
    intervals: list[float] = []
    for i in range(1, len(marks)):
        if steps[i].foot != foot:
            continue
        if not (np.isfinite(marks[i]) and np.isfinite(marks[i - 1])):
            continue
        intervals.append(marks[i] - marks[i - 1])
        k = min(int(window), len(intervals))
        out.append((marks[i], 60.0 / float(np.mean(intervals[-k:]))))
    return out


def _densify_samples(samples, fps: float, n: int) -> np.ndarray:
    """Hold each reading from its own moment until the next one.

    A step function, because nothing is known about the cadence between two
    steps. The panel smooths this for *display* by curving through the readings;
    the array itself stays honest, which is what gait.json records.
    """
    out = np.full(n, np.nan)
    for t, value in samples:
        start = int(min(n, np.ceil(t * (fps or 30.0))))
        out[start:] = value
    return out


def summary_text(analysis: GaitAnalysis, *, video_seconds: float, source: str = "") -> str:
    """Human-readable gait report, saved alongside the machine-readable JSON."""
    lines = ["Running gait summary", "=" * 58]
    if source:
        lines.append(f"{'source':<24}{source}")
    # Which way the runner is going, because the trails are raked against it
    # and a clip where the detector got it wrong should say so here rather than
    # only in gait.json. See _heading.
    side = {1.0: "towards +x (screen right)", -1.0: "towards -x (screen left)"}
    heading = side.get(analysis.heading, "undecided")
    if analysis.facing and analysis.heading != analysis.facing:
        heading += "   <- DISAGREES with the runner's facing; see quality"
    lines += [
        f"{'clip length':<24}{video_seconds:.2f}s",
        f"{'foot strikes':<24}{analysis.count}"
        f"   ({len(analysis.steps_for('left'))} left,"
        f" {len(analysis.steps_for('right'))} right)",
        f"{'direction of travel':<24}{heading}",
    ]

    if analysis.count < 2:
        lines.append("\nToo few foot strikes to measure a cadence. Check clearance.png.")
        return "\n".join(lines) + "\n"

    intervals = analysis.intervals
    left_pct, right_pct = analysis.balance()
    vo = analysis.vertical_oscillation()

    lines += [
        f"{'cadence':<24}{analysis.mean_cadence:.1f} steps/min",
        f"{'step time':<24}{float(np.mean(intervals)):.3f}s",
        f"{'stride variability':<24}{analysis.stride_spread() * 1000:.1f} ms"
        f"   ({analysis.stride_spread() / analysis.mean_stride:.1%} of the stride)",
        f"{'stride time':<24}{analysis.mean_stride:.3f}s   (one full cycle, per foot)",
        "",
        "Left / right",
        "-" * 58,
        f"{'strikes':<24}{len(analysis.steps_for('left'))} left,"
        f" {len(analysis.steps_for('right'))} right",
        f"{'stride rate':<24}{analysis.foot_stride_rate('left'):.2f} spm left,"
        f" {analysis.foot_stride_rate('right'):.2f} spm right"
        "   <- equal by construction",
        f"{'half-stride':<24}{analysis.step_interval('left'):.3f}s before a left,"
        f" {analysis.step_interval('right'):.3f}s before a right",
        f"{'phase split':<24}"
        f"{analysis.phase_split()[0]:.1f}% / {analysis.phase_split()[1]:.1f}%",
        f"{'step symmetry':<24}{analysis.symmetry():.1f}%"
        "   <- NOT MEASURABLE HERE; see below",
        f"{'contact time':<24}{analysis.contact_mean('left'):.3f}s left,"
        f" {analysis.contact_mean('right'):.3f}s right",
        f"{'contact balance':<24}{left_pct:.1f}% / {right_pct:.1f}%"
        "   <- see the note below",
        f"{'peak clearance':<24}"
        f"{float(np.mean([s.peak_clearance for s in analysis.steps_for('left')])):.3f}"
        f" / "
        f"{float(np.mean([s.peak_clearance for s in analysis.steps_for('right')])):.3f}"
        "  leg lengths",
        "",
        "Knee",
        "-" * 58,
        f"{'flexion at contact':<24}"
        + ", ".join(
            f"{analysis.knee_shape(f).flexion:.1f} +/- "
            f"{analysis.knee_shape(f).flexion_sd:.1f} deg {f}"
            for f in FEET if analysis.knee_shape(f) is not None)
        + "   <- see the note below",
        f"{'peak flexion in cycle':<24}"
        + ", ".join(f"{analysis.peak_flexion(f):.1f} deg {f}" for f in FEET),
        f"{'shank/thigh at contact':<24}"
        + ", ".join(f"{analysis.segment_ratio(f):.2f} {f}" for f in FEET)
        + "   <- should be ~1.0",
        f"{'':<24}(0 deg is a straight leg; mid-swing is the peak, not contact)",
        f"{'':<24}(sampled at the knee's own most-extended point near the",
        f"{'':<24} strike, where the rate is zero -- on the slope just after,",
        f"{'':<24} one frame of timing error costs 5-14 deg)",
        "",
        "Whole body",
        "-" * 58,
        f"{'airborne':<24}{analysis.flight_ratio * 100:.0f}% of the clip"
        "   (neither foot planted)",
        f"{'vertical oscillation':<24}"
        + (f"{vo:.3f} leg lengths per stride" if np.isfinite(vo) else "—"),
        f"{'leg length':<24}{analysis.leg_length:.3f} of the frame height",
        "",
        "Per step",
        "-" * 58,
        f"{'#':>3} {'foot':<6} {'strike':>8} {'step':>7} {'stride':>7} "
        f"{'contact':>8} {'clearance':>10}",
    ]
    def cell(value: float, width: int, places: int, unit: str = "") -> str:
        """One right-aligned number, or an em dash where there is no measurement."""
        if not np.isfinite(value):
            return f"{'—':>{width}}"
        return f"{value:>{width - len(unit)}.{places}f}{unit}"

    for s in analysis.steps:
        lines.append(
            f"{s.number:>3} {s.foot:<6} {cell(s.time, 8, 3, 's')} "
            f"{cell(s.interval, 7, 3, 's')} {cell(s.stride, 7, 3, 's')} "
            f"{cell(s.contact_seconds, 8, 3, 's')} {cell(s.peak_clearance, 10, 3)}"
        )

    lines += [
        "",
        "First, what a left/right difference can and cannot be here. The feet",
        "alternate, so each lands the same number of times and `stride rate` is",
        "equal for both by construction -- neither foot can take more steps per",
        "minute than the other. The two half-strides sum to one stride, so they",
        "are one measurement with one degree of freedom, not two rates. That",
        "measurement is the phase split, and `step symmetry` is its evenness.",
        "",
        "Everything left/right above then has the same limit, and it is worth",
        "knowing before reading any of it.",
        "",
        "The two ankles descend through the strike threshold at different rates",
        "-- 0.80 leg lengths a second on the right, 0.53 on the left -- so a",
        "fixed landmark sits at a different point in each leg's cycle, and every",
        "left-versus-right number inherits that. Whether the difference is the",
        "runner or the single side-on view of them is not separable here; either",
        "way it is not a difference in when the feet land.",
        "",
        "`step symmetry` does not survive this at all, which is why it is not on",
        "the panel. A foot strike is a physical event, so nothing about the",
        "camera can move it -- but the *landmark* is not the event. It precedes",
        "touchdown by (threshold - touchdown height) / descent speed, and the",
        "two legs descend at different speeds (0.80 leg lengths a second on the",
        "right, 0.53 on the left). Sweep the threshold and the alternation runs",
        "from +24 ms to -31 ms, passing through zero around 23% of swing, where",
        "the two feet come out even to within 0.2 ms. One free parameter erases",
        "the whole asymmetry, so it says nothing about the runner. The threshold",
        "stays at 18% because the airborne fraction reads a physiological 30%",
        "there and collapses to 8% by 29%.",
        "",
        "The cadence is untouched by any of it: across that whole sweep it moves",
        "from 179.40 to 179.71 spm, because it is timed stride by stride on one",
        "foot and a per-foot displacement cancels in that subtraction.",
        "",
        "`contact` is weaker still, on two counts. It is time under the threshold",
        "-- a proxy for ground contact, since an ankle keypoint cannot see the",
        "moment a shoe loads or leaves -- and it is the quantity the near/far",
        "asymmetry hits hardest, because the foot crossing more slowly registers",
        "the longer stance. Per-foot ground lines took the split from 42/58 to",
        "45/55; the rest is geometry. Two cameras, or a force plate, would settle",
        "it.",
        "",
        "`flexion at contact` is a projected angle, in the image plane rather",
        "than anatomical. It now agrees with anatomy at the phases anatomy has",
        "a number for: 17.5 and 22.5 degrees at contact against a textbook",
        "20-25, and a stance peak of 36 and 44 against a typical 40-45. Both of",
        "those read 10-25 degrees high until the coordinate space was fixed (an",
        "x distance and a y distance were in different units; see gait.analyze).",
        "",
        "What still does not agree is the peak in swing, 82 and 92 degrees",
        "against a published 100-125 for faster running. Two candidates, and",
        "this clip cannot separate them: the shank turns furthest out of the",
        "image plane exactly there, which a projected angle can only under-read,",
        "and published peaks are mostly measured at paces above this one. So",
        "read contact and stance as calibrated, and the swing peak as a floor.",
        "",
        "`shank/thigh at contact` is the keypoint check: near-equal segments in",
        "a real leg, so it should read about 1.0. Both legs pass, and they agree",
        "with each other, so neither knee is misplaced relative to the other.",
        "It is also the check that caught the coordinate space: a shank cannot",
        "change length, so a ratio that moved through the stride (1.6-1.9 in",
        "mid-swing, against 1.00-1.13 now) was the measurement moving.",
        "",
        "What separates the two legs is noise, not geometry. The right leg is",
        "the near one and measurably the cleaner: 20-58% less keypoint jitter",
        "and 5.8% of range stride to stride against the left's 8.7%. So the",
        "right knee is the better one to read if only one is wanted --",
        "PANEL_KNEE_FEET does that. What the gap between them means is still not",
        "answerable from one camera, since the two legs are viewed from slightly",
        "different angles. Each leg against itself over time is the comparison",
        "that survives either way.",
        "",
        "The cadence inherits none of this. It is timed stride by stride on one",
        "foot at a time, so a per-foot bias cancels exactly: every landmark tried",
        "on this clip agreed on it within 0.5 spm.",
    ]
    return "\n".join(lines) + "\n"
