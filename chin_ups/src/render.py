"""Drawing the returned poses back onto the frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .panel import build_panels
from .skeleton import KPT_NAMES, draw_hud, draw_person, visible_parts
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

    stride = max(1, round(info.n_frames / max(n_returned, 1)))
    hold_limit = stride - 1

    panels = None
    out_w = info.width
    if cfg.SIDE_PANEL and analysis is not None:
        # Same height, double the width: the clip keeps its native aspect ratio
        # and the panel takes an equal share beside it.
        panels = build_panels(analysis, info.width, info.height, cfg=cfg)
        out_w = info.width * 2
        console.print(f"  side panel: {len(panels)} states, output {out_w}x{info.height}")

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
                for person in persons:
                    draw_person(
                        frame, person, w, h,
                        edges=edges, points=points,
                        thickness=thickness, radius=radius,
                        held=held, draw_bbox=cfg.DRAW_BBOX, draw_label=cfg.DRAW_TRACK_LABEL,
                        bbox_color=cfg.BBOX_COLOR,
                    )

                if persons:
                    n_drawn += 1
                    n_people += len(persons)
                    n_held += bool(held)

                if cfg.DRAW_HUD:
                    hud = f"frame {i}/{info.n_frames}  |  {len(persons)} person(s)"
                    draw_hud(frame, hud + ("  (held)" if held else ""))

                if panels is not None:
                    frame = np.hstack([frame, panels[min(analysis.count_at(i), len(panels) - 1)]])

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
    }


def plot_displacement(analysis, dst: Path, *, cfg, title: str = "") -> None:
    """The displacement curve, with the measured interval marked on every rep.

    The shaded span is the one the panel reports (``cfg.REP_METRIC``), not a
    fixed valley-to-valley window, otherwise the graph and the video quote two
    different numbers for the same rep.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .reps import phase_label, rep_value

    metric = cfg.REP_METRIC

    def span(rep):
        """(start, end) of the interval this rep's number actually measures."""
        if metric == "moving":
            return rep.onset_time, rep.peak_time
        if metric == "total":
            return rep.valley_time, rep.end_time
        return rep.valley_time, rep.peak_time

    t = analysis.times
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax.plot(t, analysis.raw - analysis.baseline, color="#c9ced6", lw=1,
            label="raw (torso mean)", zorder=1)
    ax.plot(t, analysis.displacement, color="#2f6df6", lw=2.2, label="smoothed", zorder=3)
    ax.axhline(0, color="#8a8f98", ls="--", lw=1.2, zorder=2,
               label="relative zero (hanging)")

    top = float(np.max(analysis.displacement)) if len(analysis.displacement) else 1.0
    bar = top * 1.16          # one height for every bracket, so they read as a row
    ax.set_ylim(bottom=-top * 0.22, top=bar * 1.14)   # room for the valley timestamps

    for i, rep in enumerate(analysis.reps):
        a, b = span(rep)
        value = rep_value(rep, metric)

        ax.axvspan(a, b, color="#2f6df6", alpha=0.13, zorder=0,
                   label=phase_label(metric) if i == 0 else None)
        for x in (a, b):
            ax.axvline(x, color="#2f6df6", ls=":", lw=1.1, alpha=0.65, zorder=2)

        # The measured interval, drawn as an interval rather than implied by a
        # number floating near a peak.
        ax.annotate("", xy=(b, bar), xytext=(a, bar),
                    arrowprops=dict(arrowstyle="<|-|>", color="#1864ab", lw=1.6,
                                    shrinkA=0, shrinkB=0))
        ax.text((a + b) / 2, bar + top * 0.025, f"{value:.2f}s", ha="center",
                va="bottom", fontsize=11, fontweight="bold", color="#1864ab")

        # When the span starts at movement onset, draw the dead hang the number
        # deliberately excludes, so the graph shows why the start point is not
        # at the low point the eye goes to first.
        if metric == "moving" and a - rep.valley_time > 1.5 / 30:
            ax.axvspan(rep.valley_time, a, color="#8a8f98", alpha=0.10, zorder=0,
                       label="dead hang (excluded)" if i == 0 else None)
            ax.plot(rep.valley_time,
                    analysis.displacement[int(np.argmin(np.abs(t - rep.valley_time)))],
                    "o", color="#adb5bd", ms=5, zorder=4)
            ax.annotate(f"hang {a - rep.valley_time:.2f}s",
                        ((rep.valley_time + a) / 2, 0), textcoords="offset points",
                        xytext=(0, -30), ha="center", fontsize=8.5, color="#868e96")

        a_y = analysis.displacement[int(np.argmin(np.abs(t - a)))]
        ax.plot(b, rep.height, "o", color="#e8590c", ms=8, zorder=4)
        ax.plot(a, a_y, "o", color="#8a8f98", ms=6, zorder=4)
        ax.annotate(f"{rep.number}", (b, rep.height), textcoords="offset points",
                    xytext=(11, 3), ha="left", fontsize=11, fontweight="bold",
                    color="#e8590c")

        # Both endpoints stamped, so the bracket above is checkable by hand.
        ax.annotate(f"{b:.2f}s", (b, rep.height), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9.5, color="#e8590c")
        ax.annotate(f"{a:.2f}s", (a, a_y), textcoords="offset points",
                    xytext=(0, -17), ha="center", fontsize=9.5, color="#5c6470")

    ax.set_ylabel("vertical displacement\n(fraction of frame height)")
    ax.grid(alpha=0.25)
    # Above the axes, not inside them: any in-plot corner eventually sits on a rep.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=5,
              fontsize=9, frameon=False)
    ax.set_title(title or f"{analysis.count} reps", fontsize=13, fontweight="bold",
                 pad=28)

    ax2.plot(t, analysis.head_margin, color="#2b8a3e", lw=1.6)
    ax2.axhline(0, color="#c92a2a", ls="--", lw=1.2)
    ax2.fill_between(t, 0, analysis.head_margin,
                     where=analysis.head_margin > 0, color="#2b8a3e", alpha=0.18)
    ax2.set_ylabel("eyes above\nhands")
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.25)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.09, hspace=0.12)
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
