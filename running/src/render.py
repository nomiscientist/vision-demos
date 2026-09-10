"""Drawing the returned poses back onto the frames, beside the live panel."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .comet import wind_vector
from .gait import FEET
from .panel import FOOT_COLORS, GaitPanel
from .skeleton import (
    KPT_NAMES,
    AnkleTrails,
    draw_contact_flash,
    draw_foot_markers,
    draw_hud,
    draw_person,
    visible_parts,
)
from .text import resolve_font
from .video import VideoInfo

_PROGRESS_COLUMNS = (
    TextColumn("[cyan]rendering[/]"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total} frames"),
    TimeElapsedColumn(),
)


def render(
    info: VideoInfo,
    by_index: dict[int, list[dict]],
    n_returned: int,
    dst: Path,
    *,
    cfg,
    console,
    analysis=None,
) -> dict:
    """Write an overlay video and return per-run stats.

    When the clip was decimated (more source frames than the model posed), the
    frames in between have no result of their own. Holding the last pose across
    the gap beats flickering, and those frames are labelled `(held)` so a
    measured frame is never confused with a carried one.
    """
    edges, points = visible_parts(cfg.DRAW_FACE)

    # Stroke widths are pixels, so a fixed value thins out as the export grows.
    # Scaling against the same 720px reference the panel uses keeps the overlay
    # looking identical at any export size.
    stroke = info.width / 720
    thickness = max(1, round(cfg.LINE_THICKNESS * stroke))
    radius = max(1, round(cfg.POINT_RADIUS * stroke))
    foot_radius = max(2, round(cfg.FOOT_MARKER_RADIUS * stroke))
    flash_thickness = max(1, round(cfg.FLASH_THICKNESS * stroke))

    # The trail spans a time, not a frame count, so it covers the same slice of
    # the stride whatever the clip's frame rate is. The wind comes from the
    # clip's own heading, so nothing here knows which way this runner faces.
    trails = None
    if cfg.TRAIL_ON_ANKLES:
        wind = wind_vector(cfg.TRAIL_WIND, analysis.heading if analysis else 0.0,
                           speed=cfg.TRAIL_WIND_SPEED * stroke, fps=info.fps)
        trails = AnkleTrails(
            cfg.TRAIL_FEET,
            colors=FOOT_COLORS,
            length=max(2, round(cfg.TRAIL_SECONDS * info.fps)),
            width=max(2, round(cfg.TRAIL_WIDTH * stroke)),
            taper=cfg.TRAIL_TAPER,
            opacity=cfg.TRAIL_OPACITY,
            fade=cfg.TRAIL_FADE,
            glow=cfg.TRAIL_GLOW,
            glow_opacity=cfg.TRAIL_GLOW_OPACITY,
            softness=cfg.TRAIL_SOFTNESS,
            wind=wind,
        )

    stride = max(1, round(info.n_frames / max(n_returned, 1)))
    hold_limit = stride - 1

    panel = None
    out_w = info.width
    font_name = None
    if cfg.SIDE_PANEL and analysis is not None:
        # Same height, double the width: the clip keeps its native aspect ratio
        # and the panel takes an equal share beside it.
        font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
        font_name = font[2] if font else "OpenCV Hershey"
        panel = GaitPanel(analysis, info.width, info.height, cfg=cfg, font=font)
        out_w = info.width * 2
        console.print(f"  side panel: live, {out_w}x{info.height}, {font_name}")

    # Foot strikes indexed by frame, so the flash is a lookup rather than a scan
    # of every step on every frame.
    flashes: dict[int, list[str]] = {}
    flash_frames = max(1, round(cfg.FLASH_SECONDS * info.fps))
    if analysis is not None and cfg.FLASH_ON_CONTACT:
        for step in analysis.steps:
            flashes.setdefault(step.frame, []).append(step.foot)

    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (out_w, info.height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open a writer for {dst}")

    last_persons: list[dict] | None = None
    held_for = 0
    n_drawn = n_held = n_people = 0

    try:
        with Progress(*_PROGRESS_COLUMNS, console=console, transient=True) as progress:
            task = progress.add_task("render", total=info.n_frames)
            for i in range(info.n_frames):
                ok, frame = cap.read()
                if not ok:
                    break

                if i in by_index:
                    persons, held = by_index[i], False
                    last_persons, held_for = persons, 0
                elif last_persons is not None and held_for < hold_limit:
                    persons, held = last_persons, True
                    held_for += 1
                else:
                    persons, held = [], False

                h, w = frame.shape[:2]

                # Only the tracked subject leaves a trail, and it goes down
                # before the skeleton so a joint is never drawn under its own
                # history. Frames with no pose append a gap rather than
                # nothing, which breaks the tail where the ankle was lost
                # instead of joining a straight line across it.
                if trails is not None:
                    trails.update(persons[0] if persons else None, w, h).draw(frame)

                for person in persons:
                    draw_person(
                        frame, person, w, h,
                        edges=edges, points=points,
                        thickness=thickness, radius=radius,
                        held=held, draw_bbox=cfg.DRAW_BBOX, draw_label=cfg.DRAW_TRACK_LABEL,
                        bbox_color=cfg.BBOX_COLOR,
                    )

                # Only the tracked subject gets foot marks: they are the body the
                # signal was measured on, and marking a bystander's ankles would
                # point at a leg that is not in the graph.
                if persons and cfg.HIGHLIGHT_FEET:
                    draw_foot_markers(frame, persons[0], w, h,
                                      colors=FOOT_COLORS, radius=foot_radius)

                if persons and flashes:
                    for age in range(flash_frames):
                        for foot in flashes.get(i - age, ()):
                            draw_contact_flash(
                                frame, persons[0], w, h, foot=foot,
                                color=FOOT_COLORS[foot],
                                progress=age / flash_frames,
                                radius=foot_radius, thickness=flash_thickness,
                            )

                if persons:
                    n_drawn += 1
                    n_people += len(persons)
                    n_held += bool(held)

                if cfg.DRAW_HUD:
                    hud = f"frame {i}/{info.n_frames}  |  {len(persons)} person(s)"
                    draw_hud(frame, hud + ("  (held)" if held else ""))

                if panel is not None:
                    frame = np.hstack([frame, panel.draw(i)])

                writer.write(frame)
                progress.update(task, advance=1)
    finally:
        cap.release()
        writer.release()

    return {
        "frames_written": info.n_frames,
        "frames_with_pose": n_drawn,
        "frames_held": n_held,
        "mean_people_per_posed_frame": round(n_people / n_drawn, 2) if n_drawn else 0.0,
        "hold_limit": hold_limit,
        "output_width": out_w,
        "side_panel": panel is not None,
        "ankle_trails": trails is not None,
        "trail_wind_px_per_frame": round(wind[0], 3) if trails is not None else None,
        "panel_font": font_name,
    }


def plot_clearance(analysis, dst: Path, *, cfg, title: str = "") -> None:
    """The whole clip's gait, as a still: both ankles, every strike, the cadence.

    This is the plot to open when the panel says something surprising. The
    thresholds are drawn on, so a missed or doubled strike is visible as the
    trace failing to clear one of the two lines rather than as a number being
    wrong.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = analysis.times
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(14, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    colors = {"left": "#12b0d9", "right": "#f2820c"}
    for foot in FEET:
        ax.plot(t, analysis.raw[foot], color=colors[foot], lw=0.8, alpha=0.35,
                zorder=2, label=f"{foot} raw")
        ax.plot(t, analysis.clearance[foot], color=colors[foot], lw=2.0, zorder=3,
                label=f"{foot} smoothed")

    ax.axhline(analysis.contact_level, color="#8a8f98", ls="--", lw=1.2, zorder=1,
               label=f"strike threshold ({analysis.contact_level:.3f})")

    for step in analysis.steps:
        ax.plot(step.time, analysis.contact_level, "o", color=colors[step.foot],
                ms=6, zorder=5, mec="white", mew=0.8)
        ax.axvline(step.time, color=colors[step.foot], lw=0.6, alpha=0.25, zorder=0)

    # Every airborne stretch shaded, so the flight ratio in the report can be
    # checked against the picture rather than taken on trust.
    ax.fill_between(t, *ax.get_ylim(), where=analysis.airborne, color="#adb5bd",
                    alpha=0.10, zorder=0, label="airborne (neither foot down)")

    ax.set_ylabel("ankle height\n(leg lengths above its own stance level)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=6, fontsize=8.5,
              frameon=False, columnspacing=1.4, handlelength=1.6)
    ax.set_title(title or f"{analysis.count} foot strikes", fontsize=13,
                 fontweight="bold", pad=26)

    # The same three series the panel draws, so the still and the video cannot
    # disagree — plus the trailing-window combination, which the panel only
    # shows under PANEL_TRACE_MODE = "window".
    for foot in FEET:
        ax2.plot(t, analysis.running_cadence_series(foot), color=colors[foot], lw=2.0,
                 drawstyle="steps-post", label=f"{foot}, mean so far")
    ax2.plot(t, analysis.running_cadence_series(), color="#212529", lw=2.6,
             drawstyle="steps-post", label="both, mean so far")
    ax2.plot(t, analysis.cadence_live, color="#868e96", lw=1.0, alpha=0.8,
             drawstyle="steps-post",
             label=f"both, live (EMA alpha={cfg.CADENCE_SMOOTHING})")
    if np.isfinite(analysis.mean_cadence):
        ax2.axhline(analysis.mean_cadence, color="#1864ab", ls="--", lw=1.2, alpha=0.7,
                    label=f"clip mean {analysis.mean_cadence:.1f} spm")
    ax2.set_ylabel("cadence\n(steps / min)")
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=3)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.08, hspace=0.12)
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def plot_steps(analysis, dst: Path, *, cfg, title: str = "") -> None:
    """Step time and contact time per step, with the two feet separated.

    The panel reports one balance figure; this is where the two feet can be seen
    apart. A limp shows as two offset rows rather than as a number a point or
    two off fifty.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not analysis.steps:
        return

    colors = {"left": "#12b0d9", "right": "#f2820c"}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

    for foot in FEET:
        steps = analysis.steps_for(foot)
        xs = [s.number for s in steps]
        ax.plot(xs, [s.interval for s in steps], "o-", color=colors[foot], ms=7,
                lw=1.4, label=f"{foot} foot")
        ax2.plot(xs, [s.contact_seconds for s in steps], "o-", color=colors[foot],
                 ms=7, lw=1.4, label=f"{foot} foot")

    intervals = analysis.intervals
    if intervals:
        mean = float(np.mean(intervals))
        ax.axhline(mean, color="#495057", ls="--", lw=1.1,
                   label=f"mean {mean:.3f}s ({analysis.mean_cadence:.0f} spm)")

    ax.set_ylabel("step time (s)\nsince the other foot landed")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper right")
    ax.set_title(title or "Per step", fontsize=13, fontweight="bold")

    left, right = analysis.balance()
    ax2.set_ylabel("contact time (s)\n(proxy, see the docs)")
    ax2.set_xlabel("step number")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, frameon=False, ncol=3, loc="upper right",
               title=f"balance {left:.1f}% / {right:.1f}%", title_fontsize=9)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10, hspace=0.10)
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def plot_joint(frames: list[dict], fps: float, joint: str, dst: Path) -> bool:
    """Plot one joint's height per track. Returns False if it never appeared.

    `track_id` is what makes this meaningful: without it `persons[0]` is a list
    slot, not a person, and the series would jump between people.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j = KPT_NAMES.index(joint)

    tracks: dict = {}
    for f in frames:
        for p in f["persons"]:
            kpts = p.get("kpts_xy", [])
            if j < len(kpts) and tuple(kpts[j]) != (0, 0):
                # y grows downward in image coordinates; flip so "up" reads as up.
                tracks.setdefault(p.get("track_id"), []).append(
                    (f["index"] / fps, 1.0 - kpts[j][1])
                )

    if not tracks:
        return False

    fig, ax = plt.subplots(figsize=(11, 3.5))
    for tid, series in sorted(tracks.items(), key=lambda kv: (kv[0] is None, kv[0])):
        ts, ys = zip(*series)
        ax.plot(ts, ys, lw=1.5, label=f"track {tid}")

    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{joint} height (normalized, up = higher)")
    ax.set_title(f"{joint} over time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst, dpi=140)
    plt.close(fig)
    return True


def plot_knee(analysis, dst: Path, *, cfg, title: str = "") -> bool:
    """Every detected knee, for debugging. Returns False if there is nothing yet.

    Three panels, because three different things go wrong and each one shows up
    in a different view:

    * **At contact** — every strike's limb overlaid on a common hip. A bad pose
      is a limb pointing somewhere the others do not, which no summary statistic
      would tell you about. This is the panel to look at first.
    * **Through stance** — the mean limb at five points from strike to toe-off,
      so the loading and push-off geometry is visible as a sequence rather than
      inferred from one number.
    * **Flexion against stance** — one thin line per strike. A whole cycle that
      went wrong shows as a stray line, and the band shows whether the spread is
      even through the contact or concentrated at one end.

    Everything is hip-anchored and divided by leg length, so the limbs are
    comparable between the legs and across clips.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not analysis.steps or not analysis.limb:
        return False

    colors = {"left": "#12b0d9", "right": "#f2820c"}
    offset = {"left": 0.0, "right": 0.85}      # leg lengths, to sit side by side
    fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1, 1, 1.35]})
    ax_hit, ax_stance, ax_flex = axes

    def limb_at(foot: str, t: float):
        """Hip-anchored (dx, dy) of knee and ankle at time *t*, in leg lengths."""
        L = analysis.limb[foot]
        i = t * (analysis.fps or 30.0)
        grid = np.arange(analysis.n_frames)
        pick = lambda j, c: np.interp(i, grid, L[j][c])   # noqa: E731
        hx, hy = pick("hip", 0), pick("hip", 1)
        out = []
        for j in ("knee", "ankle"):
            out.append(((pick(j, 0) - hx) / analysis.leg_length,
                        (pick(j, 1) - hy) / analysis.leg_length))
        return out

    # ── 1. every strike, overlaid ────────────────────────────────────────────
    for foot in FEET:
        steps = analysis.steps_for(foot)
        if not steps:
            continue
        dx = offset[foot]
        for s in steps:
            (kx, ky), (ax_, ay) = limb_at(foot, s.knee_frame / (analysis.fps or 30.0))
            ax_hit.plot([dx, dx + kx, dx + ax_], [0, ky, ay],
                        color=colors[foot], lw=1.0, alpha=0.35, solid_capstyle="round")
        shape = analysis.knee_shape(foot)
        mx = np.mean([limb_at(foot, s.knee_frame / (analysis.fps or 30.0))
                      for s in steps], axis=0)
        ax_hit.plot([dx, dx + mx[0][0], dx + mx[1][0]], [0, mx[0][1], mx[1][1]],
                    color=colors[foot], lw=4.0, solid_capstyle="round",
                    label=f"{foot}: {shape.flexion:.1f} ± {shape.flexion_sd:.1f}°"
                          f"  (n={len(steps)})")
        ax_hit.plot([dx], [0], "o", color=colors[foot], ms=9, mec="white", mew=1.2)
    ax_hit.set_title("At contact — every strike overlaid\n(thin = one strike, "
                     "bold = the mean)", fontsize=11, fontweight="bold")

    # ── 2. the mean limb through stance ──────────────────────────────────────
    fracs = np.linspace(0.0, 1.0, 5)
    for foot in FEET:
        steps = [s for s in analysis.steps_for(foot)
                 if np.isfinite(s.contact_seconds)]
        if not steps:
            continue
        dx = offset[foot]
        for k, frac in enumerate(fracs):
            pts = np.mean([limb_at(foot, s.time + frac * s.contact_seconds)
                           for s in steps], axis=0)
            fade = 0.30 + 0.70 * frac
            ax_stance.plot([dx, dx + pts[0][0], dx + pts[1][0]],
                           [0, pts[0][1], pts[1][1]],
                           color=colors[foot], lw=2.6, alpha=fade,
                           solid_capstyle="round",
                           label=f"{foot} {frac:.0%} of stance"
                           if k in (0, len(fracs) - 1) else None)
        ax_stance.plot([dx], [0], "o", color=colors[foot], ms=9, mec="white", mew=1.2)
    ax_stance.set_title("Through ground contact — mean limb\n"
                        "(faint = touchdown, solid = toe-off)",
                        fontsize=11, fontweight="bold")

    for ax in (ax_hit, ax_stance):
        ax.set_aspect("equal")
        ax.invert_yaxis()          # image coordinates: y grows downward
        ax.set_xlabel("leg lengths")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8.5, frameon=False, loc="lower center")
    ax_hit.set_ylabel("leg lengths below the hip")

    # ── 3. flexion against stance, one line per strike ───────────────────────
    pct = np.linspace(0, 100, 51)
    for foot in FEET:
        kf = analysis.knee_flexion[foot]
        grid = np.arange(len(kf))
        stack = []
        for s in analysis.steps_for(foot):
            if not np.isfinite(s.contact_seconds):
                continue
            ts = s.time + (pct / 100.0) * s.contact_seconds
            stack.append(np.interp(ts * (analysis.fps or 30.0), grid, kf))
        if not stack:
            continue
        stack = np.array(stack)
        for row in stack:
            ax_flex.plot(pct, row, color=colors[foot], lw=0.8, alpha=0.30)
        m, sd = stack.mean(axis=0), stack.std(axis=0)
        ax_flex.plot(pct, m, color=colors[foot], lw=3.0, label=f"{foot} mean")
        ax_flex.fill_between(pct, m - sd, m + sd, color=colors[foot], alpha=0.15)
    ax_flex.set_xlabel("% of ground contact  (0 = strike, 100 = toe-off)")
    ax_flex.set_ylabel("knee flexion (deg)\n0 = straight leg")
    ax_flex.grid(alpha=0.25)
    ax_flex.legend(fontsize=9, frameon=False, loc="upper left")
    ax_flex.set_title("Flexion through contact — one line per strike\n"
                      "(band = ±1 sd)", fontsize=11, fontweight="bold")

    fig.suptitle(title or "Knee at foot strike", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.11, wspace=0.24)
    fig.savefig(dst, dpi=150)
    plt.close(fig)
    return True
