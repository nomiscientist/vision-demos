"""The fading trail, drawn the same way on both halves of the frame.

A comet is a path plus two ramps along it: the width tapers and the alpha
fades from the head backwards. Both the overlay's ankle trails and the panel's
ankle view are that, over different paths and coordinate systems, so the
compositing lives here once.

The fade is a real alpha composite over a blurred mask, not an additive glow.
Added light blows out over a bright background and disappears over a dark one,
and the overlay is drawn over whatever the runner was filmed against. The color
is laid down wider than the mask, so blurring the mask softens the edge instead
of feathering it towards black.

**The wind is a drawing, not a measurement.** A comet's tail does not point
back along its path — it streams downwind of it, and the trail reads as motion
rather than as a smear precisely because it does. Each point of the tail is
displaced downwind in proportion to its *age*, so the head, at age zero, stays
exactly on the joint it belongs to and only the spent part of the trail drifts.
Read the head as the position and the tail as decay, not as a path: with any
wind at all the tail is no longer where the joint was. `wind=(0, 0)` gives the
path back exactly, and the direction comes from the clip's own detected heading
(see gait._heading), never from a hardcoded side.

**Smoothing is a curve through the same points, not a different path.** One
path entry is one frame, and a joint can turn sharply between two frames — the
top of a swing, a foot strike — which draws as a visible elbow when the points
are only ever joined straight. `smooth` subdivides each frame-to-frame span
with a Catmull-Rom spline (see `catmull_rom`), so the corner rounds off without
moving where the joint actually was: the curve still passes through every
original point, it just stops cutting the corner between them. Age, and so the
wind and the fade, are tracked in real elapsed frames throughout — a
subdivision is a fraction of one frame old, not a step of its own — so
smoothing more finely never changes how fast the trail drifts or dims.
"""

from __future__ import annotations

import cv2
import numpy as np


def catmull_rom(points, per_span: int):
    """Dense points along a Catmull-Rom spline through *points*.

    Passes exactly through every input point, so the curve never moves a
    reading or a measured position -- it only rounds off the corner between
    them. Its tangents want one point either side; the first and last are
    clamped to themselves, so nothing beyond the ends of *points* is read.

    Shared by the cadence trace (panel._trace, over per-step samples) and the
    comet trail (Comet, over per-frame joint positions) -- same math, two
    different kinds of vertex.
    """
    n = len(points)
    if n < 3:
        return list(points)
    out = []
    for i in range(n - 1):
        p0 = points[max(i - 1, 0)]
        p1, p2 = points[i], points[i + 1]
        p3 = points[min(i + 2, n - 1)]
        for k in range(per_span):
            t = k / per_span
            t2, t3 = t * t, t * t * t
            # Uniform Catmull-Rom basis, halved as the standard form requires.
            b0 = -0.5 * t3 + t2 - 0.5 * t
            b1 = 1.5 * t3 - 2.5 * t2 + 1.0
            b2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
            b3 = 0.5 * t3 - 0.5 * t2
            out.append((p0[0] * b0 + p1[0] * b1 + p2[0] * b2 + p3[0] * b3,
                        p0[1] * b0 + p1[1] * b1 + p2[1] * b2 + p3[1] * b3))
    out.append(points[-1])
    return out


class Comet:
    """A reusable trail compositor. One instance per drawing surface.

    The two scratch buffers are held rather than allocated per call: at 1080p a
    fresh pair every frame is 6 MB of zeroing for a few hundred px of trail.
    """

    def __init__(self, *, width: int, taper: float = 0.18, opacity: float = 0.85,
                 fade: float = 1.35, glow: float = 2.6, glow_opacity: float = 0.28,
                 softness: float = 0.6, wind: tuple[float, float] = (0.0, 0.0),
                 smooth: int = 1):
        self.width = max(1, int(width))
        # Displacement per step of age, in px. One step is one entry in the
        # path, which is one frame for both of this project's trails.
        self.wind = (float(wind[0]), float(wind[1]))
        self.taper = float(min(max(taper, 0.0), 1.0))
        self.opacity = float(min(max(opacity, 0.0), 1.0))
        self.fade = max(float(fade), 0.1)
        self.glow = max(float(glow), 1.0)
        self.glow_opacity = float(min(max(glow_opacity, 0.0), 1.0))
        # Blur radius in px, as a fraction of the head width. The mask is the
        # only thing blurred, so this is purely how hard the edge is.
        self.spread = max(0, round(self.width * float(softness)))
        # Sub-points per frame-to-frame span, via catmull_rom. 1 leaves the
        # path exactly as given -- straight segments, one per frame -- which is
        # what the video overlay's trail uses: it is thin and short-lived, and
        # the corners are not the thing a viewer is reading there. The panel's
        # ankle view is a wider, longer-lived trail where the same corner reads
        # as a kink, so it asks for more.
        self.smooth = max(1, int(smooth))
        self._layer = self._mask = self._box = None

    def _segments(self, points):
        """(a, b, head_fraction) per drawable segment, oldest first, blown downwind.

        The fraction is 0 at the tail and 1 at the newest segment, measured
        against however much path was handed over — so a trail fades across its
        full length from the first frame rather than starting as a bright stub.

        A `None` in the path is a gap rather than a point, and the segments
        either side of it are simply not drawn: that breaks the tail where the
        joint was actually lost instead of joining a straight line across it.

        The rake is by *age in real frames*, not by position in the list --
        a point five frames old has drifted the same distance whether the
        trail is fifteen frames long or fifty, and whether `smooth` has cut
        that span into one segment or six. Only the position is denser with
        smoothing on; age still advances by exactly one frame from one original
        point to the next, so the wind speed and the fade curve read the same
        either way -- smoothing changes how the corners look, not how fast the
        trail moves or dims.
        """
        span = max(len(points) - 1, 1)
        last = len(points) - 1
        wx, wy = self.wind

        def blown(point, age: float):
            if not (wx or wy):
                return point
            return (int(round(point[0] + wx * age)), int(round(point[1] + wy * age)))

        # Contiguous runs between gaps, smoothed independently: a spline drawn
        # through a `None` would invent a position for a frame the joint was
        # never seen on.
        i = 0
        n = len(points)
        while i < n:
            if points[i] is None:
                i += 1
                continue
            j = i
            while j + 1 < n and points[j + 1] is not None:
                j += 1
            run = points[i:j + 1]

            if self.smooth <= 1 or len(run) < 3:
                dense, step = run, 1.0
            else:
                dense, step = catmull_rom(run, self.smooth), 1.0 / self.smooth

            # age(k) is linear in position along the run by construction --
            # catmull_rom's k-th dense point sits at continuous index i+k*step
            # of the original run, and age is (last - original index) -- so a
            # plain countdown from the run's oldest age reproduces it exactly,
            # with no need to track each dense point's source index.
            age = last - i
            for k in range(len(dense) - 1):
                a_age, b_age = age, age - step
                f = 1.0 - b_age / span
                yield blown(dense[k], a_age), blown(dense[k + 1], b_age), f
                age = b_age
            i = j + 1

    def draw(self, img, paths):
        """Composite one trail per (points, color) pair in *paths*.

        All of them go into one blend, so trails that cross do not composite
        over each other twice.
        """
        h, w = img.shape[:2]
        if self._layer is None or self._layer.shape[:2] != (h, w):
            self._layer = np.zeros_like(img)
            self._mask = np.zeros((h, w), np.uint8)
        elif self._box is not None:
            x0, y0, x1, y1 = self._box
            self._layer[y0:y1, x0:x1] = 0
            self._mask[y0:y1, x0:x1] = 0
        self._box = None

        # The spline (if smoothing) is fit once per path, not once per pass --
        # the glow and the core are drawn over the same segments below, and
        # refitting it for the second pass would just repeat the first.
        paths = [(list(self._segments(points)), color)
                 for points, color in paths if color is not None]
        xs: list[int] = []
        ys: list[int] = []
        # The halo first and the core over it: both write rather than add, so
        # the brighter core simply replaces the halo it sits inside, and a
        # trail crossing itself does not stack up into a bright knot.
        for scale, opacity in ((self.glow, self.glow_opacity), (1.0, self.opacity)):
            for segments, color in paths:
                for a, b, f in segments:
                    taper = self.taper + (1.0 - self.taper) * f
                    weight = max(1, round(self.width * taper * scale))
                    alpha = round(255 * opacity * f ** self.fade)
                    if alpha <= 0:
                        continue
                    cv2.line(self._layer, a, b, color, weight + 2 * self.spread)
                    cv2.line(self._mask, a, b, alpha, weight, cv2.LINE_AA)
                    xs += [a[0], b[0]]
                    ys += [a[1], b[1]]

        if not xs:
            return img

        pad = max(2, round(self.width * self.glow) + 2 * self.spread)
        x0 = max(0, min(xs) - pad)
        y0 = max(0, min(ys) - pad)
        x1 = min(w, max(xs) + pad)
        y1 = min(h, max(ys) + pad)
        if x1 <= x0 or y1 <= y0:
            return img
        self._box = (x0, y0, x1, y1)

        mask = self._mask[y0:y1, x0:x1]
        if self.spread:
            k = 2 * self.spread + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        # Composited on the box alone: the arithmetic is float, and over a full
        # 1080p frame that would cost more than everything else in the loop.
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        roi = img[y0:y1, x0:x1].astype(np.float32)
        blended = roi * (1.0 - alpha) + self._layer[y0:y1, x0:x1].astype(np.float32) * alpha
        img[y0:y1, x0:x1] = blended.astype(np.uint8)
        return img


def wind_vector(setting: str, heading: float, *, speed: float, fps: float,
                ) -> tuple[float, float]:
    """Downwind drift per frame, in px, for a Comet.

    *setting* is `"auto"` to take it from the clip — downwind is the direction
    of travel reversed — or `"left"` / `"right"` to force a side, or `"none"`
    to switch the rake off and let the tail lie along the real path. *speed* is
    px per second, already scaled to the surface being drawn on.

    An unknown heading (no clip analysed, or a clip with no ground contacts to
    measure one from) resolves to no wind rather than to a guessed side: the
    tail then simply follows the path, which is the honest fallback.
    """
    sign = {"auto": -float(np.sign(heading)), "left": -1.0, "right": 1.0,
            "none": 0.0}.get(str(setting).lower(), -float(np.sign(heading)))
    if not sign or speed <= 0 or fps <= 0:
        return (0.0, 0.0)
    return (sign * float(speed) / float(fps), 0.0)
