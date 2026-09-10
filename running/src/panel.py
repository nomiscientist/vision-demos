"""The side panel: cadence, big, over a cadence graph and the average knee.

Built with OpenCV and Pillow rather than matplotlib, because this panel changes
every frame — 455 matplotlib figures would cost more than the whole render. The
parts that never move (titles, axes, tick labels, gridlines) are drawn once into
a background image, and each frame copies that and draws only what moves.

Three blocks under the headline, evenly spaced:

* **Average cadence over time**, spanning the whole clip with a playhead. One
  trace: the mean so far, which is what the title says and what removes the need
  for any key. The live reading has the headline to itself. There is deliberately
  no per-foot trace either — the feet alternate, so each lands the same number of
  times and neither can have a higher rate than the other.

  This strip used to be square, on the argument that a slope on screen should
  then be a slope in the data. A wide box compresses slope, and that is a real
  cost paid deliberately: the trace is a *running mean*, so it settles within a
  few strides and then drifts by about a spm. What it is there to show is the
  settling and the drift, not a rate to read off by eye — the live figure is the
  headline and every underlying reading is in `gait.json`. The y axis keeps its
  fixed minimum span (PANEL_Y_MIN_SPAN), so a flat-looking trace still cannot be
  a rescaled wobble.

* **The average knee at foot strike**, one leg either side. This is what the
  contact events buy: they give a *phase* to average at, so every strike can be
  stacked on the same instant of the cycle. The limb is drawn at its measured
  orientation, and the fan around the shank is one standard deviation of the
  knee angle across those strikes — so a consistent runner draws a narrow wedge
  and an inconsistent one a wide fan.

* **The ankles, live**, lifted out of the pose and drawn on their own. Two dots
  and their comet trails in the space they actually move through, with a dim
  cloud of everywhere each ankle has already been. It is the same two joints the
  whole analysis is built on, with the body taken away: the loop each one traces,
  how far apart the two are, and where the belt line sits under them.

Nothing here is drawn before it happened. Every trace, the headline, the knee
average and the ankle view read only frames at or before the one being written,
so the panel is a recording being made rather than a finished chart with a cursor
on it. That is also why the ankle view has no full-clip path drawn faintly behind
the live one — the shape of the loop is the answer, and showing it from frame one
would be showing the future.
"""

from __future__ import annotations

import cv2
import numpy as np

BG = (18, 18, 20)          # BGR, near-black
GRID = (40, 40, 46)
AXIS = (90, 90, 98)
DIM = (150, 150, 158)
INK = (245, 245, 245)
MUTED = (110, 110, 118)

from . import text as textmod          # noqa: E402
from .comet import Comet, catmull_rom, wind_vector  # noqa: E402
from .gait import FEET                 # noqa: E402

# The feet keep the colors their limbs already have in the overlay, so the knee
# drawn on the panel and the leg it was measured from are the same color.
FOOT_COLORS = {
    "left": (255, 200, 0),     # cyan
    "right": (0, 140, 255),    # orange
}
# The cadence trace, BGR. It sits on a near-black ground, so a saturated blue
# rather than a genuinely dark one -- below about this lightness a 3px line
# stops reading at all.
MEAN_COLOR = (235, 111, 31)     # #1f6fea, deep blue


class _Box:
    """A plot rectangle plus the data range it maps, in one place."""

    def __init__(self, px0: int, px1: int, py0: int, py1: int):
        self.px0, self.px1, self.py0, self.py1 = px0, px1, py0, py1
        self.x0 = self.x1 = self.y0 = self.y1 = 0.0

    @property
    def width(self) -> int:
        return self.px1 - self.px0

    @property
    def height(self) -> int:
        return self.py1 - self.py0

    def sx(self, x: float) -> int:
        span = max(self.x1 - self.x0, 1e-9)
        t = (float(x) - self.x0) / span
        return int(round(self.px0 + self.width * float(np.clip(t, 0.0, 1.0))))

    def sy(self, y: float) -> int:
        span = max(self.y1 - self.y0, 1e-9)
        t = (float(y) - self.y0) / span
        return int(round(self.py1 - self.height * float(np.clip(t, 0.0, 1.0))))


def _nice_step(span: float, target: int, candidates) -> float:
    """The smallest candidate step that keeps *span* under *target* ticks."""
    for value in candidates:
        if span / value <= target:
            return float(value)
    return float(candidates[-1])


class GaitPanel:
    """Draws the panel for a given frame. One instance per render."""

    def __init__(self, analysis, width: int, height: int, *, cfg, font=None):
        self.a = analysis
        self.w, self.h = width, height
        self.cfg = cfg
        self.font = font
        self.u = width / 720.0          # everything is sized at a 720px-wide panel
        self.fps = analysis.fps or 30.0

        # ── type sizes ───────────────────────────────────────────────────────
        self.size_title = self._s(cfg.PANEL_TITLE_SIZE)
        self.size_number = self._s(cfg.PANEL_NUMBER_SIZE)
        self.size_unit = self._s(cfg.PANEL_UNIT_SIZE)
        self.size_sub = self._s(cfg.PANEL_SUB_SIZE)
        self.size_graph_title = self._s(cfg.PANEL_GRAPH_TITLE_SIZE)
        self.size_axis_label = self._s(cfg.PANEL_AXIS_LABEL_SIZE)
        self.size_tick = self._s(cfg.PANEL_TICK_SIZE)
        self.size_knee_key = self._s(cfg.PANEL_KNEE_KEY_SIZE)
        self.size_knee_value = self._s(cfg.PANEL_KNEE_VALUE_SIZE)

        # ── horizontal extent ────────────────────────────────────────────────
        # Placed from the ink, not from the plot rectangle: the left gutter
        # carries the tick numbers and the rotated y label, and a box centred
        # without counting them sits visibly right of everything above it.
        side_margin = self._s(cfg.PANEL_SIDE_MARGIN)
        self.y_label_dx = self._s(52)
        gutter = side_margin + self.y_label_dx + self.size_axis_label // 2
        avail_w = (width - side_margin) - gutter

        # The two drawing blocks get the full ink column instead, centred on
        # the panel: neither carries tick numbers or a rotated label, so
        # aligning them to the *plot rectangle* left them sitting visibly right
        # of every title above them, by exactly the gutter's half-width.
        draw_x0, draw_x1 = side_margin, width - side_margin

        # Both elastic blocks are shapes rather than free rectangles, so the
        # ankle path's own proportions are needed before the height is split.
        self.ankle_series, ankle_shape = self._ankle_series()

        # ── vertical layout ──────────────────────────────────────────────────
        # One content block, measured from its own type sizes and then centred
        # as a whole. Every break between blocks is the same `gap_section`, and
        # every title sits the same `gap_plot` above its own drawing, so the
        # rhythm down the panel is even by construction rather than by eye.
        gap_number = self._s(14)
        gap_sub = self._s(10)
        gap_section = self._s(cfg.PANEL_SECTION_GAP)
        gap_plot = self._s(14)
        below_axis = self.size_tick + self._s(10) + self.size_axis_label
        knee_label_h = self.size_knee_value + self._s(9)

        # The credit line owns the bottom-right corner and is placed from the
        # panel edge, not from this block, so it never moves.
        credit = (self._s(cfg.ATTRIBUTION_SIZE) + 2 * self._s(cfg.ATTRIBUTION_MARGIN)
                  if cfg.ATTRIBUTION else self._s(14))

        # Top and bottom air, reserved before the plots get their share. This
        # used to be a bare 8px floor: the block took every pixel left after the
        # fixed elements, so whatever it did not need collapsed to nothing and
        # the title sat 8px off the panel edge. Reserving the margins first and
        # letting the two plots shrink into what remains is the way round that
        # keeps the air fixed and the drawings elastic.
        margin_top = self._s(cfg.PANEL_TOP_MARGIN)
        margin_bottom = self._s(cfg.PANEL_BOTTOM_MARGIN)

        # Everything that is not one of the elastic drawings. The knee label is
        # deliberately absent: it is drawn *inside* the knee box, and that box
        # already carves it out of its own height, so counting it here as well
        # reserved it twice and left a strip of dead space below the block that
        # nothing could ever occupy.
        fixed = (self.size_title + gap_number + self.size_number + gap_sub
                 + self.size_sub
                 + gap_section + self.size_graph_title + gap_plot + below_axis
                 + gap_section + self.size_graph_title + gap_plot
                 + gap_section + self.size_graph_title + gap_plot)
        room = height - credit - fixed - margin_top - margin_bottom

        # The cadence strip is now a fixed *shape* rather than a square: it
        # spans the whole column and its height follows from the aspect ratio,
        # which is what frees the height the ankle view lives in. Capped at half
        # the room, so a small aspect number cannot squeeze the two drawings
        # below it down to their floors.
        graph_h = max(self._s(50),
                      min(int(round(avail_w / max(float(cfg.PANEL_GRAPH_ASPECT), 0.2))),
                          int(room * 0.5)))
        rest = room - graph_h

        # The ankle view is a shape too, and the shape is the *data's*: at the
        # column's width, this is exactly the height its path needs at equal
        # scale on both axes. Sizing it any taller would only add a dead band
        # under the drawing, since an equal-scale fit cannot stretch to fill.
        # Capped, so a clip with a tall path (a runner crossing the frame, or a
        # high knee drill) cannot crowd the knee block out.
        span_x, span_y = ankle_shape
        ideal = int(round((draw_x1 - draw_x0) * span_y / max(span_x, 1e-9)))
        ankle_h = max(self._s(60),
                      min(ideal, int(rest * cfg.PANEL_ANKLE_MAX_SHARE)))
        knee_h = max(self._s(60), rest - ankle_h)

        px0, px1 = gutter, gutter + avail_w

        # Placed at the reserved top margin rather than centred in the leftover,
        # so the air above the title is the number in the config and not a
        # remainder.
        top = margin_top

        y = top
        self.y_title = y
        y += self.size_title + gap_number
        self.y_number = y + self.size_number // 2      # drawn centred, not topped
        y += self.size_number + gap_sub
        self.y_sub = y
        y += self.size_sub + gap_section

        self.y_graph_title = y
        y += self.size_graph_title + gap_plot
        self.plot = _Box(px0, px1, y, y + graph_h)
        y += graph_h + below_axis + gap_section

        self.y_knee_title = y
        y += self.size_graph_title + gap_plot
        # The knee block spans the whole ink column, so each leg gets an honest
        # half of the width.
        self.knee = _Box(draw_x0, draw_x1, y, y + knee_h)
        self.knee_label_h = knee_label_h
        y += knee_h + gap_section

        self.y_ankle_title = y
        y += self.size_graph_title + gap_plot
        self.ankle = _Box(draw_x0, draw_x1, y, y + ankle_h)

        # ── data ranges ──────────────────────────────────────────────────────
        self.plot.x0 = 0.0
        self.plot.x1 = analysis.n_frames / self.fps

        # One trace: the mean so far. The live reading is the headline's own
        # figure and used to be drawn here too, which needed a colour key to
        # tell the two apart — a key for a distinction the title can just make.
        #
        # Held as per-step vertices rather than a per-frame array, because a
        # reading per step is all that was ever measured. The drawing curves
        # through them; see comet.catmull_rom.
        self.mean_samples = analysis.mean_samples()
        finite = np.array([v for _, v in self.mean_samples if np.isfinite(v)])

        # A minimum span, and it is half of why this trace reads as a line
        # rather than a seismograph. A steady runner varies by about a spm, so
        # an axis auto-fitted to the data alone spans ~15 spm across the plot —
        # 34 px per spm — and magnifies whatever wobble is left. With the floor
        # and the smoothing in src/gait._stride_ema_samples together, the live
        # trace went from a 22px lurch across 11px of spacing (63°) to 8px
        # across the same 11px (37°), against dance_sync's ~28°.
        if finite.size:
            span = max(float(np.ptp(finite)) * 1.25, float(cfg.PANEL_Y_MIN_SPAN))
            mid = 0.5 * (finite.min() + finite.max())
            # Bounds exactly at mid +/- span/2, with the ticks falling on round
            # numbers *inside* them. Snapping the bounds themselves to tick
            # multiples inflated a 30 spm span to 40.
            lo, hi = mid - span / 2, mid + span / 2
            step = _nice_step(span, 4, (5, 10, 20, 25, 50))
        else:
            step, lo, hi = 20.0, 140.0, 200.0
        self.plot.y0, self.plot.y1 = float(lo), float(hi)
        self.y_step = step

        self.knee_track = self._knee_track()
        self.ankle_track = self._ankle_window()
        self.background = self._background()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _s(self, px: float) -> int:
        return max(1, int(round(px * self.u)))

    def _knee_track(self) -> dict:
        """Per-frame knee geometry, eased so it drifts instead of stepping.

        The average only changes when a strike lands, so drawn straight it is a
        static shape that jumps once a stride — and a figure that never moves
        reads as broken rather than as steady. A one-pole filter towards the
        current average gives continuous drift: the value on screen is always
        heading for a real measurement, it just takes about half a second to
        get there.

        Causal by construction. The target at frame i is the average over
        strikes sampled at or before i, and the filter only ever reads its own
        previous output, so nothing here can show a measurement early.

        Precomputed rather than eased inside draw(), which also takes the
        per-frame cost of knee_shape() out of the render loop.
        """
        feet = [f for f in FEET if f in self.cfg.PANEL_KNEE_FEET] or list(FEET)
        alpha = float(self.cfg.PANEL_KNEE_EASING)
        keys = ("thigh", "shank", "thigh_len", "shank_len", "flexion",
                "flexion_sd", "thigh_sd")
        track = {f: {k: np.full(self.a.n_frames, np.nan) for k in keys}
                 for f in feet}
        for foot in feet:
            state = None
            for i in range(self.a.n_frames):
                shape = self.a.knee_shape(foot, until=i / self.fps,
                                          window=self.cfg.PANEL_KNEE_WINDOW)
                if shape is None or not np.isfinite(shape.flexion):
                    continue
                target = {
                    "thigh": shape.thigh_angle, "shank": shape.shank_angle,
                    "thigh_len": shape.thigh_length,
                    "shank_len": shape.shank_length,
                    "flexion": shape.flexion,
                    # Left as NaN on a single strike, which has no spread. It
                    # used to be coerced to zero, and the label then printed
                    # "± 0.0°" -- claiming perfect consistency from one sample.
                    "flexion_sd": shape.flexion_sd,
                    "thigh_sd": shape.thigh_sd,
                }
                # The first reading is taken as-is; easing up from zero would
                # animate a shape that was never measured.
                if state is None:
                    state = dict(target)
                else:
                    # NaN must not be eased *from*: the spread is NaN on the
                    # first strike, and `nan + alpha * (x - nan)` is nan, so a
                    # naive filter kept it NaN for the whole clip and the label
                    # never regained its ± or its shaded band. A field adopts
                    # its first finite reading outright and eases after that.
                    nxt = {}
                    for k in keys:
                        goal, prev = target[k], state[k]
                        if not np.isfinite(goal) or not np.isfinite(prev):
                            nxt[k] = goal
                        else:
                            nxt[k] = prev + alpha * (goal - prev)
                    state = nxt
                for k in keys:
                    track[foot][k][i] = state[k]
            track[foot]["bend"] = 1.0
            shape = self.a.knee_shape(foot)
            if shape is not None:
                track[foot]["bend"] = shape.bend_sign
        return track

    def _ankle_series(self) -> tuple[dict, tuple[float, float]]:
        """Both ankles' per-frame position, and the shape of the space they use.

        Returns the per-foot ``(x, y)`` arrays in leg lengths, plus the padded
        ``(x_span, y_span)`` the layout sizes the block from — so the drawing
        gets a box its own proportions rather than a rectangle it has to sit
        inside.

        Two frames of reference, and which one is right is the same question
        FOOT_REFERENCE already answers for the signal, so "auto" follows it:

        * **image** — the ankle where it is in the shot. Right for a fixed
          camera, and the only mode where the belt line is a fixed line, so it
          is the only one that draws it. A pan smears the loops, which is
          honest: the ankle really did move across the frame.
        * **hip** — the ankle against the mid-hip, which cancels a camera that
          moves with the runner and centres the loops on the body.

        Units are leg lengths, and y is negated so up on the panel is up in the
        world; image y grows downward. Both axes are already in the same unit
        when they arrive — gait.analyze scales x by the frame's aspect ratio,
        since the model normalizes each axis by a different number — so the
        loop drawn here keeps its real proportions rather than being stretched
        by the shape of the frame.
        """
        cfg = self.cfg
        feet = [f for f in FEET if f in cfg.PANEL_ANKLE_FEET] or list(FEET)
        mode = str(cfg.PANEL_ANKLE_FRAME).lower()
        if mode == "auto":
            mode = "hip" if cfg.FOOT_REFERENCE == "hip" else "image"
        self.ankle_mode = mode

        leg = self.a.leg_length or 1e-6
        hip_x = hip_y = 0.0
        if mode == "hip":
            hip_x = np.nanmean(np.vstack([self.a.limb[f]["hip"][0] for f in FEET]),
                               axis=0)
            hip_y = np.nanmean(np.vstack([self.a.limb[f]["hip"][1] for f in FEET]),
                               axis=0)

        track = {}
        for foot in feet:
            ax, ay = self.a.limb[foot]["ankle"]
            track[foot] = ((np.asarray(ax, float) - hip_x) / leg,
                           -(np.asarray(ay, float) - hip_y) / leg)

        # The belt line, in the same units. Fixed in the frame, so it only
        # means anything in image mode; one line for the two feet, because
        # there is one belt and the per-foot split in the signal is a correction
        # for where each ankle was seen from, not two floors.
        self.ankle_ground = (-float(np.mean(list(self.a.ground.values()))) / leg
                             if mode == "image" and cfg.PANEL_ANKLE_GROUND else None)

        xs = np.concatenate([t[0] for t in track.values()]) if track else np.zeros(0)
        ys = np.concatenate([t[1] for t in track.values()]) if track else np.zeros(0)
        xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
        if not xs.size or not ys.size:
            xs = ys = np.array([0.0, 1.0])
        lo_y, hi_y = float(ys.min()), float(ys.max())
        if self.ankle_ground is not None:
            lo_y, hi_y = min(lo_y, self.ankle_ground), max(hi_y, self.ankle_ground)

        pad = 1.0 + 2 * float(cfg.PANEL_ANKLE_PAD)
        self._ankle_bounds = (float(xs.min()), float(xs.max()), lo_y, hi_y)

        # A second copy of the same path, eased rather than exact -- for the
        # *trail* to draw through, not for the dot. A near-180 degree reversal
        # at toe-off is real, one or two frames wide, and a curve required to
        # pass exactly through every sample still shows it as a point no
        # matter how densely it is subdivided: rounding a genuine reversal
        # means moving the drawn position off the exact sample near it, which
        # only an approximating filter does, not an interpolating spline.
        # Causal for the same reason every live trace on this panel is: state
        # at frame i must not depend on frame i+1.
        alpha = float(cfg.PANEL_ANKLE_PATH_EASE)
        self.ankle_track_eased = {}
        for foot, (fx, fy) in track.items():
            ex, ey = np.array(fx, dtype=float), np.array(fy, dtype=float)
            for i in range(1, len(ex)):
                if not (np.isfinite(ex[i]) and np.isfinite(ey[i])):
                    continue
                if np.isfinite(ex[i - 1]) and np.isfinite(ey[i - 1]):
                    ex[i] = alpha * ex[i] + (1 - alpha) * ex[i - 1]
                    ey[i] = alpha * ey[i] + (1 - alpha) * ey[i - 1]
            self.ankle_track_eased[foot] = (ex, ey)

        return track, (max(float(xs.max() - xs.min()), 1e-6) * pad,
                       max(hi_y - lo_y, 1e-6) * pad)

    def _ankle_window(self) -> dict:
        """Fit the ankle box to the path, and build what the drawing reuses.

        A fixed window, like the cadence plot's axes: what is *drawn* stays
        causal, but a window that grew with the data would slide the whole
        picture every time the runner drifted on the belt, and nothing in it
        could then be compared with anything else.
        """
        cfg, box = self.cfg, self.ankle
        track = self.ankle_series
        x_lo, x_hi, y_lo, y_hi = self._ankle_bounds
        pad = 1.0 + 2 * float(cfg.PANEL_ANKLE_PAD)
        x_span = max(x_hi - x_lo, 1e-6) * pad
        y_span = max(y_hi - y_lo, 1e-6) * pad

        # Equal scale on both axes, so the path keeps its real shape and the
        # window opens up around it rather than stretching it to fit. The block
        # was sized from the same two spans, so what opens up is a few px.
        px_per_unit = min(box.width / x_span, box.height / y_span)
        half_x = box.width / (2 * px_per_unit)
        half_y = box.height / (2 * px_per_unit)
        x_mid, y_mid = 0.5 * (x_hi + x_lo), 0.5 * (y_hi + y_lo)
        box.x0, box.x1 = x_mid - half_x, x_mid + half_x
        box.y0, box.y1 = y_mid - half_y, y_mid + half_y

        # The cloud's curve, fit once over the whole clip rather than
        # re-fit from a small trailing window every frame: catmull_rom's
        # tangents are clamped at whichever points are at the *edge* of
        # whatever it is handed, and a window that moves every frame puts a
        # fresh clamp at a fresh place every frame, which showed up as a
        # slightly different little kink in each frame's segment rather than
        # one continuous curve. Fit once, sliced as the clip plays, the clamp
        # only ever falls at the true ends of the eased path.
        #
        # Dense point `d` sits at continuous index `d / K` along the eased
        # per-frame path (see comet.catmull_rom): real frame `i`'s span is
        # always dense indices ((i-1)*K, i*K], which is what lets the cloud
        # reveal it a frame at a time without recomputing anything.
        self.ankle_dense_step = K = max(1, int(cfg.PANEL_ANKLE_TRAIL_SMOOTH))
        self.ankle_dense = {}
        for foot, (ex, ey) in self.ankle_track_eased.items():
            pts = list(zip(ex.tolist(), ey.tolist()))
            finite = all(np.isfinite(x) and np.isfinite(y) for x, y in pts)
            self.ankle_dense[foot] = catmull_rom(pts, K) if finite and K > 1 else pts

        # One trail, drawn into the box and nothing outside it: with a wind
        # rake the tail leaves the data window, and it must clip against this
        # block rather than paint over the knee above or the credit below.
        self.ankle_comet = Comet(
            width=max(2, self._s(cfg.PANEL_ANKLE_TRAIL_WIDTH)),
            taper=cfg.PANEL_ANKLE_TRAIL_TAPER,
            opacity=cfg.PANEL_ANKLE_TRAIL_OPACITY,
            fade=cfg.PANEL_ANKLE_TRAIL_FADE,
            glow=cfg.PANEL_ANKLE_GLOW,
            glow_opacity=cfg.PANEL_ANKLE_GLOW_OPACITY,
            softness=cfg.PANEL_ANKLE_SOFTNESS,
            wind=wind_vector(cfg.TRAIL_WIND, self.a.heading,
                             speed=cfg.PANEL_ANKLE_WIND_SPEED * self.u,
                             fps=self.fps),
            # This trail is wide and lives a while, unlike the overlay's, so a
            # frame-to-frame turn -- the top of the swing, a foot strike --
            # reads as a visible elbow rather than the arc it actually is.
            # Rounded off, not rerouted: catmull_rom still passes through every
            # measured position, at PANEL_ANKLE_TRAIL_SMOOTH sub-points per
            # frame.
            smooth=max(1, int(cfg.PANEL_ANKLE_TRAIL_SMOOTH)),
        )
        self.ankle_trail = max(2, round(cfg.PANEL_ANKLE_TRAIL_SECONDS * self.fps))
        self.ankle_settle_frames = max(1, round(cfg.PANEL_ANKLE_PATH_SETTLE_SECONDS * self.fps))

        # The cloud's state, kept between frames: it is state, and the frame
        # it is state *as of* is tracked with it, so a draw() out of order
        # rebuilds rather than showing history from the wrong point in the
        # run. `_heat` is coverage, 0-1, combined by maximum rather than by a
        # direct alpha composite -- see _visited_cloud for why that is what
        # keeps each frame's join to the last one from reading as its own
        # bright spot.
        halflife = max(float(cfg.PANEL_ANKLE_MEMORY_HALFLIFE_SECONDS), 1e-3)
        self._memory_decay = 0.5 ** (1.0 / max(halflife * self.fps, 1e-6))
        self._visited_upto = -1
        self._visited_pen = np.zeros((box.height, box.width), np.uint8)
        self._heat = None
        return track

    # ── the parts that never change ──────────────────────────────────────────
    def _background(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        f, cfg, box = self.font, self.cfg, self.plot

        textmod.draw(img, cfg.PANEL_GRAPH_TITLE, f, size=self.size_graph_title,
                     xy=(self.w // 2, self.y_graph_title), color=DIM, anchor="ct")

        # y-axis tick labels, no ruled gridlines behind them -- the trace
        # itself and the two axes are enough scaffolding for a plot this
        # size, and a dashed rule at every tick was competing with the two
        # traces that actually cross it.
        first = np.ceil(box.y0 / self.y_step) * self.y_step
        for value in np.arange(first, box.y1 + 1e-9, self.y_step):
            y = box.sy(value)
            textmod.draw(img, f"{int(round(value))}", f, size=self.size_tick,
                         xy=(box.px0 - self._s(9), y), color=MUTED, anchor="rm")

        cv2.line(img, (box.px0, box.py0), (box.px0, box.py1), AXIS, 1, cv2.LINE_AA)
        cv2.line(img, (box.px0, box.py1), (box.px1, box.py1), AXIS, 1, cv2.LINE_AA)

        x_step = _nice_step(box.x1 - box.x0, 7, (1, 2, 5, 10, 15, 30, 60))
        for sec in np.arange(0, box.x1 + 1e-9, x_step):
            textmod.draw(img, f"{int(sec)}", f, size=self.size_tick,
                         xy=(box.sx(sec), box.py1 + self._s(8)),
                         color=MUTED, anchor="ct")

        textmod.draw(img, cfg.PANEL_X_LABEL, f, size=self.size_axis_label,
                     xy=(self.w // 2, box.py1 + self._s(10) + self.size_tick),
                     color=DIM, anchor="ct")
        textmod.draw(img, cfg.PANEL_Y_LABEL, f, size=self.size_axis_label,
                     xy=(box.px0 - self.y_label_dx, (box.py0 + box.py1) // 2),
                     color=DIM, anchor="cm", rotate=90)

        textmod.draw(img, cfg.PANEL_KNEE_TITLE, f, size=self.size_graph_title,
                     xy=(self.w // 2, self.y_knee_title), color=DIM, anchor="ct")

        # A hairline between the legs, so each reads as its own column rather
        # than as one wide drawing with two limbs in it. Only when there are two.
        feet = [f for f in FEET if f in cfg.PANEL_KNEE_FEET] or list(FEET)
        if len(feet) > 1:
            mid_x = (self.knee.px0 + self.knee.px1) // 2
            for y in range(self.knee.py0, self.knee.py1 - self.knee_label_h,
                           self._s(10)):
                cv2.line(img, (mid_x, y),
                         (mid_x, min(y + self._s(5), self.knee.py1)), GRID, 1,
                         cv2.LINE_AA)

        textmod.draw(img, cfg.PANEL_ANKLE_TITLE, f, size=self.size_graph_title,
                     xy=(self.w // 2, self.y_ankle_title), color=DIM, anchor="ct")

        box = self.ankle

        # A small legend, top-right of the block: which colour is which foot
        # is the one thing that stopped being obvious once the skeleton (and
        # its own colour-coded limbs) was taken out of the picture. Reuses the
        # knee block's own LEFT/RIGHT strings, so the wording agrees with it.
        legend_gap = self._s(6)
        legend_row = self.size_tick + self._s(7)
        ly = box.py0 + self._s(8) + self.size_tick // 2
        for i, foot in enumerate(("left", "right")):
            name = cfg.PANEL_KNEE_LEFT if foot == "left" else cfg.PANEL_KNEE_RIGHT
            ty = ly + i * legend_row
            tx = box.px1 - self._s(6)
            tw = textmod.measure(name, f, self.size_tick)[0]
            textmod.draw(img, name, f, size=self.size_tick, xy=(tx, ty),
                         color=MUTED, anchor="rm")
            r = self._s(3)
            cv2.circle(img, (tx - tw - legend_gap - r, ty), r, FOOT_COLORS[foot],
                      -1, cv2.LINE_AA)

        if self.ankle_ground is not None:
            # The belt line the strikes are timed against, so the loops sit on
            # a floor instead of floating. Dashed and dim: it is a reference,
            # and the ankles are the subject.
            gy = box.sy(self.ankle_ground)
            for x in range(box.px0, box.px1, self._s(12)):
                cv2.line(img, (x, gy), (min(x + self._s(5), box.px1), gy), AXIS, 1,
                         cv2.LINE_AA)
        elif self.ankle_mode == "hip" and box.x0 < 0.0 < box.x1:
            # Hip-anchored, so the reference is the body — but the hip itself is
            # a leg length above the ankles and including it would shrink the
            # loops to fit a mostly empty box. The line under it is the part
            # worth having anyway: it says where each ankle is relative to
            # directly beneath the runner. Drawn only when it is genuinely in
            # view, never clamped to an edge, which would put a reference mark
            # somewhere the reference is not.
            hx = box.sx(0.0)
            for y in range(box.py0, box.py1, self._s(12)):
                cv2.line(img, (hx, y), (hx, min(y + self._s(5), box.py1)), AXIS, 1,
                         cv2.LINE_AA)
            textmod.draw(img, cfg.PANEL_ANKLE_HIP_LABEL, f, size=self.size_tick,
                         xy=(hx + self._s(6), box.py0), color=MUTED, anchor="lt")

        if cfg.ATTRIBUTION:
            textmod.draw(img, cfg.ATTRIBUTION, f, size=self._s(cfg.ATTRIBUTION_SIZE),
                         xy=(self.w - self._s(cfg.ATTRIBUTION_MARGIN),
                             self.h - self._s(cfg.ATTRIBUTION_MARGIN)),
                         color=INK, anchor="rb", opacity=cfg.ATTRIBUTION_OPACITY)
        return img

    # ── per-frame content ────────────────────────────────────────────────────
    def _headline(self, img, frame: int) -> None:
        f = self.font
        textmod.draw(img, self.cfg.PANEL_TITLE, f, size=self.size_title,
                     xy=(self.w // 2, self.y_title), color=INK, anchor="ct")

        live = self.a.cadence_at(frame)
        defined = bool(np.isfinite(live))
        text = f"{live:.0f}" if defined else "—"

        # The number and its unit are laid out as one line and centred together,
        # so the panel's centre line runs through the pair rather than through
        # the digits with the unit hanging off the side. Before the third step
        # there is no cadence yet and the dash stands alone: "— spm" reads as a
        # unit waiting for a value, which is more distracting than the gap.
        unit = self.cfg.PANEL_UNIT if defined else ""
        wide, _ = textmod.measure(text, f, self.size_number)
        unit_w = textmod.measure(unit, f, self.size_unit)[0] if unit else 0
        pad = self._s(9) if unit else 0
        left = self.w // 2 - (wide + pad + unit_w) // 2
        textmod.draw(img, text, f, size=self.size_number,
                     xy=(left, self.y_number), color=INK, anchor="lm")
        if unit:
            textmod.draw(img, unit, f, size=self.size_unit,
                         xy=(left + wide + pad,
                             self.y_number + self.size_number // 2),
                         color=MUTED, anchor="lb")

        landed = self.a.steps_by(frame)
        template = (self.cfg.PANEL_STEPS_LABEL_ONE if landed == 1
                    else self.cfg.PANEL_STEPS_LABEL)
        textmod.draw(img, template.format(steps=landed), f, size=self.size_sub,
                     xy=(self.w // 2, self.y_sub), color=MUTED, anchor="ct")

    def _trace(self, img, samples, frame: int, color, *, thickness: int) -> None:
        """One cadence series as a smooth curve, up to *frame*.

        Only the readings at or before the current frame are drawn, so the curve
        is a recording being made. Past the last reading it is held flat out to
        the playhead — the cadence has not been measured again since, and a
        sloped continuation would be a guess about the step still in progress.
        """
        box = self.plot
        now = frame / self.fps
        visible = [(t, v) for t, v in samples if t <= now and np.isfinite(v)]
        if not visible:
            return

        pixels = [(box.sx(t), box.sy(v)) for t, v in
                  catmull_rom(visible, max(2, int(self.cfg.PANEL_TRACE_SEGMENTS)))]
        for a_, b_ in zip(pixels[:-1], pixels[1:]):
            cv2.line(img, a_, b_, color, thickness, cv2.LINE_AA)

        end = pixels[-1]
        live = (box.sx(now), end[1])
        if live[0] > end[0]:
            cv2.line(img, end, live, color, thickness, cv2.LINE_AA)

        cv2.circle(img, live, self._s(5), color, -1, cv2.LINE_AA)
        cv2.circle(img, live, self._s(5), BG, self._s(1), cv2.LINE_AA)

    def _graph(self, img, frame: int) -> None:
        box = self.plot

        # A playhead first, so the trace sits on top of it.
        x = box.sx(frame / self.fps)
        cv2.line(img, (x, box.py0), (x, box.py1), (86, 86, 94), 1, cv2.LINE_AA)

        self._trace(img, self.mean_samples, frame, MEAN_COLOR,
                    thickness=self._s(self.cfg.PANEL_TRACE_THICKNESS))

    # ── the average knee ─────────────────────────────────────────────────────
    def _knee(self, img, frame: int) -> None:
        """The mean limb at foot strike, with its spread shaded around it.

        Drawn at the limb's *measured* orientation rather than uprighted, so the
        forward lean of the shank at contact is visible.

        The shading is the limb swept across ±1 sd — of the knee angle *and* of
        the whole limb's orientation together, since both vary and shading only
        the shank understated it. Each sampled limb is laid down translucent, so
        where they overlap the band is denser: the shape reads as a brush stroke
        whose width is the runner's own consistency, without needing a legend to
        say so.
        """
        cfg, box = self.cfg, self.knee
        feet = [f for f in FEET if f in cfg.PANEL_KNEE_FEET] or list(FEET)
        share = box.width // len(feet)
        draw_h = box.height - self.knee_label_h

        for i, foot in enumerate(feet):
            cx = box.px0 + share // 2 + i * share
            track = self.knee_track.get(foot)
            if track is None or not np.isfinite(track["flexion"][frame]):
                continue      # nothing measured yet: draw nothing, say nothing

            g = {k: float(track[k][frame]) for k in
                 ("thigh", "shank", "thigh_len", "shank_len", "flexion",
                  "flexion_sd", "thigh_sd")}
            bend = float(track["bend"])
            color = FOOT_COLORS[foot]

            # Scale on the limb's *total* length, so the drawing keeps one size
            # whatever the knee is doing; scaling on hip-to-ankle would make a
            # bent leg draw bigger than a straight one.
            total = g["thigh_len"] + g["shank_len"]
            scale = (draw_h * cfg.PANEL_KNEE_FILL) / max(total, 1e-6)

            def limb(thigh: float, shank: float):
                knee_off = np.array([np.cos(thigh), np.sin(thigh)]) \
                    * g["thigh_len"] * scale
                ankle_off = knee_off + np.array([np.cos(shank), np.sin(shank)]) \
                    * g["shank_len"] * scale
                return knee_off, ankle_off

            knee_off, ankle_off = limb(g["thigh"], g["shank"])
            # Centre the limb's own bounding box in its column, so a leg at any
            # orientation sits optically centred rather than hanging off its hip.
            pts = np.vstack([[0.0, 0.0], knee_off, ankle_off])
            centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
            origin = np.array([cx, box.py0 + draw_h / 2]) - centre

            def draw_limb(canvas, thigh, shank, thickness):
                k_off, a_off = limb(thigh, shank)
                hip = tuple(np.round(origin).astype(int))
                kn = tuple(np.round(origin + k_off).astype(int))
                an = tuple(np.round(origin + a_off).astype(int))
                cv2.line(canvas, hip, kn, color, thickness, cv2.LINE_AA)
                cv2.line(canvas, kn, an, color, thickness, cv2.LINE_AA)
                return hip, kn, an

            # The band: the same limb at a spread of plausible configurations,
            # accumulated as *coverage* rather than composited in one pass.
            # Compositing once gives every covered pixel the same alpha and the
            # band comes out flat — a uniform slab with hard edges. Counting how
            # many of the sampled limbs cross each pixel and using that as the
            # alpha makes it dense along the middle and feather out at ±1 sd,
            # which is what a spread actually looks like.
            # No spread to sweep on the first strike, so the band is a single
            # limb until there are two to compare.
            sd_flex = g["flexion_sd"] if np.isfinite(g["flexion_sd"]) else 0.0
            spread = max(1, int(cfg.PANEL_KNEE_ENVELOPE_STEPS))
            band = self._s(cfg.PANEL_KNEE_THICKNESS)
            y0, y1 = box.py0, box.py0 + draw_h
            x0, x1 = box.px0 + i * share, box.px0 + (i + 1) * share
            coverage = np.zeros((y1 - y0, x1 - x0), np.float32)
            layer = np.zeros_like(coverage, np.uint8)
            shifted = origin - np.array([x0, y0])
            for k in np.linspace(-cfg.PANEL_KNEE_FAN_SD, cfg.PANEL_KNEE_FAN_SD,
                                 spread):
                thigh_k = g["thigh"] + k * g["thigh_sd"]
                flex_k = g["flexion"] + k * sd_flex
                k_off, a_off = limb(thigh_k,
                                    thigh_k + bend * np.radians(flex_k))
                layer[:] = 0
                pt = [tuple(np.round(shifted + o).astype(int))
                      for o in ([0.0, 0.0], k_off, a_off)]
                cv2.line(layer, pt[0], pt[1], 1, band, cv2.LINE_AA)
                cv2.line(layer, pt[1], pt[2], 1, band, cv2.LINE_AA)
                coverage += layer

            alpha = np.clip(coverage * cfg.PANEL_KNEE_FAN_STEP_OPACITY, 0.0,
                            cfg.PANEL_KNEE_FAN_OPACITY)[..., None]
            region = img[y0:y1, x0:x1].astype(np.float32)
            tint = np.array(color, np.float32)
            img[y0:y1, x0:x1] = (region * (1 - alpha) + tint * alpha).astype(np.uint8)

            hip, kn, an = draw_limb(img, g["thigh"], g["shank"], band)
            for point, r in ((hip, 5), (an, 5), (kn, 7)):
                cv2.circle(img, point, self._s(r), color, -1, cv2.LINE_AA)
                cv2.circle(img, point, self._s(r), BG, self._s(1), cv2.LINE_AA)

            # The label, under its own limb: the foot, then the angle and
            # spread. Read on a slow grid rather than per frame — the shape
            # keeps easing every frame, but a tenth of a degree changing thirty
            # times a second is not a reading, it is flicker. Holding the digits
            # and letting the drawing move is the same split the sibling
            # project's panel uses for its headline.
            hold = max(1, int(round(cfg.PANEL_KNEE_LABEL_HOLD_SECONDS * self.fps)))
            at = (frame // hold) * hold
            if not np.isfinite(track["flexion"][at]):
                at = frame          # before the grid catches up, show what there is
            name = cfg.PANEL_KNEE_LEFT if foot == "left" else cfg.PANEL_KNEE_RIGHT
            places = int(cfg.PANEL_KNEE_DECIMALS)
            sd_at = float(track["flexion_sd"][at])
            value = (cfg.PANEL_KNEE_FORMAT if np.isfinite(sd_at)
                     else cfg.PANEL_KNEE_FORMAT_NO_SD).format(
                deg=f"{float(track['flexion'][at]):.{places}f}",
                sd=f"{sd_at:.{places}f}")
            label_y = box.py1 - self.knee_label_h // 2
            name_w = textmod.measure(name, self.font, self.size_knee_key)[0]
            value_w = textmod.measure(value, self.font, self.size_knee_value)[0]
            gap = self._s(9)
            x = cx - (name_w + gap + value_w) // 2
            textmod.draw(img, name, self.font, size=self.size_knee_key,
                         xy=(x, label_y), color=color, anchor="lm")
            textmod.draw(img, value, self.font, size=self.size_knee_value,
                         xy=(x + name_w + gap, label_y), color=INK, anchor="lm")

    # ── the ankles, on their own ─────────────────────────────────────────────
    def _ankle_point(self, foot: str, i: int):
        """One ankle in box pixels at frame *i*, or None if it was not seen."""
        xs, ys = self.ankle_track[foot]
        if i < 0 or i >= len(xs) or not (np.isfinite(xs[i]) and np.isfinite(ys[i])):
            return None
        return self.ankle.sx(xs[i]), self.ankle.sy(ys[i])

    def _ankle_point_eased(self, foot: str, i: int):
        """Same as `_ankle_point`, off the eased copy of the path.

        For the trail only. The dot and the newest point of any trail segment
        always read `_ankle_point` instead, so the head of the eased line
        still lands exactly on the joint it is trailing off of.
        """
        xs, ys = self.ankle_track_eased[foot]
        if i < 0 or i >= len(xs) or not (np.isfinite(xs[i]) and np.isfinite(ys[i])):
            return None
        return self.ankle.sx(xs[i]), self.ankle.sy(ys[i])

    def _dense_span(self, foot: str, i: int):
        """Box-relative points covering real frame (i-1, i], off the one curve
        fit over the whole clip in `_ankle_window` — see the note there.

        A slice, not a re-fit: frame i always owns dense indices
        ((i-1)*step, i*step], so one frame's span picks up exactly where the
        previous frame's left off, with no seam between them.
        """
        box, dense, step = self.ankle, self.ankle_dense.get(foot), self.ankle_dense_step
        if not dense:
            return None
        lo, hi = max(0, (i - 1) * step), min(len(dense) - 1, i * step)
        if hi <= lo:
            return None
        return [(int(round(box.sx(x) - box.px0)), int(round(box.sy(y) - box.py0)))
                for x, y in dense[lo:hi + 1]]

    def _visited_cloud(self, img, frame: int) -> None:
        """Where each ankle has recently been, fading out as it ages.

        Not the comet — the comet is the live head, the last
        PANEL_ANKLE_TRAIL_SECONDS, drawn fresh every frame. This is the
        *rest* of the history: every earlier pass, left to dim once drawn, so
        a runner well into the clip is not looking at every loop run so far
        stacked on top of each other.

        Held as a *coverage* buffer, valued 0-1, not painted straight onto the
        image. Each frame draws one short stroke for its own span of the
        path, and two consecutive frames' strokes necessarily share their
        join -- a thick line covers a full disk at each of its own ends, so
        both frames' strokes cover that disk, once each. Adding a fresh
        alpha-blend for both, or even adding their coverage together, leaves
        that shared disk with more paint than a singly-covered pixel gets --
        which reads exactly like what it is: a dot sitting at every frame's
        sampled position, tiled along the whole trail. Combining by *maximum*
        instead closes that off completely rather than just shrinking it: two
        overlapping full-coverage strokes give a pixel that is still exactly
        as covered as one, because 1 and 1 combine to 1, not 2. A pixel only
        brightens by being freshly stroked; being stroked *twice* in the same
        instant cannot brighten it further, so there is nothing left for a
        join to stand out with.

        Decay runs first, every frame, so a pixel not restroked keeps fading
        on schedule regardless of how bright an earlier maximum left it.

        Not raked by the wind, unlike the comet: this is where the ankle
        measurably was, not a drawing of motion over it.
        """
        opacity = float(self.cfg.PANEL_ANKLE_MEMORY_OPACITY)
        if opacity <= 0:
            return
        box = self.ankle
        if self._heat is None or frame < self._visited_upto:
            # A draw() that goes backwards is a different clip position, not a
            # continuation, so the trail starts again rather than showing
            # history from somewhere else in the run.
            self._heat = {foot: np.zeros((box.height, box.width), np.float32)
                          for foot in self.ankle_track}
            self._visited_upto = -1

        pen = self._visited_pen
        width = max(1, self._s(self.cfg.PANEL_ANKLE_MEMORY_WIDTH))
        decay = self._memory_decay
        for i in range(self._visited_upto + 1, frame + 1):
            for foot, heat in self._heat.items():
                heat *= decay
                poly = self._dense_span(foot, i)
                if poly is None:
                    continue
                pen[:] = 0
                for p0, p1 in zip(poly[:-1], poly[1:]):
                    cv2.line(pen, p0, p1, 255, width, cv2.LINE_AA)
                np.maximum(heat, pen.astype(np.float32) * (1.0 / 255.0), out=heat)
        self._visited_upto = frame

        # Coloured in one pass, from coverage straight to the background --
        # there is no running composited image to keep between frames any
        # more, so nothing here needs telling to start fresh apart from `_heat`.
        region = self.background[box.py0:box.py1, box.px0:box.px1].astype(np.float32)
        for foot, heat in self._heat.items():
            alpha = heat * opacity
            tint = np.array(FOOT_COLORS[foot], np.float32)
            region = region * (1.0 - alpha[..., None]) + tint * alpha[..., None]
        img[box.py0:box.py1, box.px0:box.px1] = region.astype(np.uint8)

    def _ankles(self, img, frame: int) -> None:
        """The two ankles and their trails, in the space they move through."""
        box = self.ankle
        self._visited_cloud(img, frame)

        # Drawn into a view of the box, so a tail blown out of the window
        # clips against this block instead of painting over the one above it.
        #
        # Every point starts exact and settles onto the eased path over
        # PANEL_ANKLE_PATH_SETTLE_FRAMES, instead of being exact-then-eased
        # outright. The outright version drew a joint's own position two
        # different ways depending only on whether *this* draw call happened
        # to be the one frame it was the head: exact on that one frame, eased
        # on every frame after -- a point already on screen retroactively
        # relocating by however far the two disagree, tens of px on this
        # clip, every single frame, the moment the head moved past it. A
        # short settle instead means a point's drawn position only ever
        # drifts by a small, continuous amount frame to frame, at both ends:
        # zero right at the head, where it must line up with the dot, and
        # zero again once fully settled, where it stops moving at all.
        window = img[box.py0:box.py1, box.px0:box.px1]
        settle = max(1, self.ankle_settle_frames)
        paths = []
        for foot in self.ankle_track:
            points = []
            lo = max(0, frame - self.ankle_trail + 1)
            for i in range(lo, frame + 1):
                age = frame - i
                exact = self._ankle_point(foot, i)
                eased = self._ankle_point_eased(foot, i)
                if exact is None or eased is None:
                    point = None
                elif age >= settle:
                    point = eased
                else:
                    # Raised cosine: 1 at age 0, 0 at age >= settle, and flat
                    # (zero slope) at both ends, so the point's own velocity
                    # never jumps either -- only its curvature does, briefly.
                    w = 0.5 * (1.0 + np.cos(np.pi * age / settle))
                    point = (exact[0] * w + eased[0] * (1.0 - w),
                            exact[1] * w + eased[1] * (1.0 - w))
                points.append(None if point is None
                              else (point[0] - box.px0, point[1] - box.py0))
            paths.append((points, FOOT_COLORS[foot]))
        self.ankle_comet.draw(window, paths)

        # The heads last, on top of both trails: these two dots are the
        # measurement, and they are the same two points the overlay marks.
        for foot in self.ankle_track:
            point = self._ankle_point(foot, frame)
            if point is None:
                continue
            r = self._s(self.cfg.PANEL_ANKLE_DOT)
            cv2.circle(img, point, r, FOOT_COLORS[foot], -1, cv2.LINE_AA)
            cv2.circle(img, point, r, BG, self._s(1), cv2.LINE_AA)

    def draw(self, frame_index: int) -> np.ndarray:
        img = self.background.copy()
        frame = int(np.clip(frame_index, 0, max(self.a.n_frames - 1, 0)))
        self._headline(img, frame)
        self._graph(img, frame)
        self._knee(img, frame)
        self._ankles(img, frame)
        return img
