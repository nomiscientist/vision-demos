"""COCO-17 skeleton definition and per-frame drawing."""

from __future__ import annotations

from collections import deque

import cv2

from .comet import Comet

# The 17 keypoints, in the order the model returns them.
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

FACE_KPTS = frozenset({0, 1, 2, 3, 4})   # nose, eyes, ears

# (a, b, side); side picks the color. "H" edges all touch a face keypoint, so
# dropping FACE_KPTS removes them together with the ear-to-shoulder lines.
SKELETON = [
    (15, 13, "L"), (13, 11, "L"), (16, 14, "R"), (14, 12, "R"),
    (11, 12, "T"), (5, 11, "L"), (6, 12, "R"), (5, 6, "T"),
    (5, 7, "L"), (7, 9, "L"), (6, 8, "R"), (8, 10, "R"),
    (0, 1, "H"), (0, 2, "H"), (1, 3, "H"), (2, 4, "H"), (3, 5, "H"), (4, 6, "H"),
]

# BGR, because OpenCV.
SIDE_COLORS = {
    "L": (255, 200, 0),    # cyan, left limbs
    "R": (0, 140, 255),    # orange, right limbs
    "T": (80, 220, 80),    # green, torso
    "H": (200, 80, 220),   # magenta, head
}

TRACK_COLORS = [
    (66, 135, 245), (245, 176, 66), (66, 245, 152), (245, 66, 200),
    (245, 245, 66), (140, 66, 245), (66, 245, 245), (245, 66, 66),
]


def track_color(track_id) -> tuple[int, int, int]:
    if track_id is None:
        return (200, 200, 200)
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


def visible_parts(draw_face: bool) -> tuple[list[tuple[int, int, str]], frozenset]:
    """The edges and keypoint indices to draw."""
    if draw_face:
        return SKELETON, frozenset(range(len(KPT_NAMES)))
    edges = [(a, b, s) for a, b, s in SKELETON if a not in FACE_KPTS and b not in FACE_KPTS]
    points = frozenset(i for i in range(len(KPT_NAMES)) if i not in FACE_KPTS)
    return edges, points


def draw_person(
    img,
    person: dict,
    width: int,
    height: int,
    *,
    edges,
    points,
    thickness: int = 3,
    radius: int = 4,
    held: bool = False,
    draw_bbox: bool = True,
    draw_label: bool = False,
    bbox_color: tuple[int, int, int] = (255, 160, 40),
):
    """Draw one person. Coordinates arrive normalized to [0, 1].

    They are normalized against the frame the model was shown (downscaled to a
    2048px long edge), but that scaling is uniform, so multiplying by this
    frame's full width/height is exact rather than approximate.
    """
    pts = [(int(round(x * width)), int(round(y * height))) for x, y in person.get("kpts_xy", [])]

    def ok(i: int) -> bool:
        # (0, 0) is the model's "not visible" sentinel, not a real corner point.
        return 0 <= i < len(pts) and i in points and pts[i] != (0, 0)

    tint = track_color(person.get("track_id"))

    if draw_bbox:
        bx, by, bw, bh = person["bbox_xywh"]
        p0 = (int(bx * width), int(by * height))
        p1 = (int((bx + bw) * width), int((by + bh) * height))
        cv2.rectangle(img, p0, p1, bbox_color, max(1, thickness - 1), cv2.LINE_AA)

        if draw_label:
            tid = person.get("track_id")
            label = f"id {tid}" if tid is not None else "person"
            if held:
                label += " (held)"
            cv2.putText(img, label, (p0[0], max(14, p0[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2, cv2.LINE_AA)

    for a, b, side in edges:
        if ok(a) and ok(b):
            cv2.line(img, pts[a], pts[b], SIDE_COLORS[side], thickness, cv2.LINE_AA)

    for i in range(len(pts)):
        if ok(i):
            cv2.circle(img, pts[i], radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pts[i], radius, tint, 1, cv2.LINE_AA)

    return img


def draw_hud(img, text: str):
    """Text with a dark outline, so it stays readable over any background."""
    for color, weight in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, weight, cv2.LINE_AA)
    return img


# ── running: the two joints the panel is measured from ───────────────────────
ANKLES = {"left": KPT_NAMES.index("left_ankle"), "right": KPT_NAMES.index("right_ankle")}


def _ankle_point(person: dict, foot: str, width: int, height: int):
    """Pixel position of one ankle, or None when the model did not see it."""
    kpts = person.get("kpts_xy") or []
    i = ANKLES[foot]
    if i >= len(kpts) or tuple(kpts[i]) == (0, 0):
        return None
    x, y = kpts[i]
    return int(round(x * width)), int(round(y * height))


def draw_foot_markers(img, person: dict, width: int, height: int, *, colors: dict,
                      radius: int):
    """Emphasize both ankles in the panel's own per-foot colors.

    The skeleton already colors left and right limbs differently, but the ankle
    is one dot among seventeen. Enlarging exactly the two joints the analysis
    reads is what lets a viewer check the trace against the leg it came from.
    """
    for foot, color in colors.items():
        point = _ankle_point(person, foot, width, height)
        if point is None:
            continue
        cv2.circle(img, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, point, radius, (255, 255, 255), max(1, radius // 4), cv2.LINE_AA)
    return img


def draw_contact_flash(img, person: dict, width: int, height: int, *, foot: str,
                       color, progress: float, radius: int, thickness: int):
    """A ring at the ankle that expands and fades over a foot strike.

    *progress* runs 0 at the strike to 1 when the flash is spent, so the ring
    grows and dims together. It marks the same instant the panel's dots do,
    which is what ties the two halves of the frame to one event.
    """
    point = _ankle_point(person, foot, width, height)
    if point is None:
        return img
    t = float(min(max(progress, 0.0), 1.0))
    faded = tuple(int(c * (1.0 - t)) for c in color)
    cv2.circle(img, point, radius + int(radius * 1.8 * t), faded,
               max(1, thickness), cv2.LINE_AA)
    return img


class AnkleTrails:
    """The recent path of each ankle, drawn as a comet tail behind it.

    The overlay says where an ankle *is*; the trail says that it has been
    moving, which is what makes a swing read as a swing rather than as a dot
    that moved between frames. Its head sits exactly on the foot marker and the
    tail thins, fades and streams downwind over the last `length` frames.

    Downwind is the clip's own direction of travel, reversed, so the trails
    behave like the air the runner is moving through — and on a treadmill,
    where the runner goes nowhere, it is still the direction they are running
    *towards* that sets it. See gait._heading for how that is measured and
    comet.Comet for what the rake does to the tail.
    """

    def __init__(self, feet, *, colors, length: int, width: int, taper: float = 0.18,
                 opacity: float = 0.85, fade: float = 1.35, glow: float = 2.6,
                 glow_opacity: float = 0.28, softness: float = 0.6,
                 wind: tuple[float, float] = (0.0, 0.0)):
        self._pts = {foot: deque(maxlen=max(2, length)) for foot in feet}
        self._colors = colors
        self._comet = Comet(width=width, taper=taper, opacity=opacity, fade=fade,
                            glow=glow, glow_opacity=glow_opacity,
                            softness=softness, wind=wind)

    def update(self, person: dict | None, width: int, height: int):
        """Record this frame's ankle positions. `None` for a frame with no pose.

        A frame the model saw nothing on appends `None` rather than nothing at
        all, so the gap breaks the tail where the ankle was actually lost
        instead of joining a straight line across it.
        """
        for foot, history in self._pts.items():
            history.append(None if person is None else _ankle_point(person, foot, width, height))
        return self

    def draw(self, img):
        return self._comet.draw(img, [(list(history), self._colors.get(foot))
                                      for foot, history in self._pts.items()])
