"""The side panel: rep count, average pace, and a per-rep timing plot.

Every rep is known before the first frame is written, so the panel is rendered
once per distinct count (seven images for a six-rep clip) and composited, rather
than redrawn 458 times. That also lets the axes be fixed up front, since the x
axis is the rep *number* and the y limits span every rep in the clip. Nothing
rescales as dots appear, which is what makes the reveal read as a graph filling
in rather than a graph thrashing.
"""

from __future__ import annotations

import numpy as np

from .reps import extremes, rep_value
from .text import resolve_font

# One palette. Hex is what matplotlib wants; _rgb() converts for the text
# drawing, so the plot and the bands around it cannot drift apart.
BG = "#14161a"
FG = "#f2f4f7"
MUTED = "#8b9199"
ACCENT = "#28a0ff"
GRID = "#2b3038"
FAST = "#51cf66"
SLOW = "#ff922b"


def _rgb(color: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` -> ``(r, g, b)``."""
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _plot_font(font):
    """Use the overlay's own face for the plot text, when matplotlib can load it.

    ``FontProperties(fname=...)`` reads face 0 of a .ttc. If it cannot be loaded
    we fall back to matplotlib's default rather than failing the render.
    """
    if font is None:
        return None
    try:
        from matplotlib import font_manager

        return font_manager.FontProperties(fname=font[0])
    except Exception:
        return None


def _verdict(reps, metric: str) -> list[tuple[str, tuple[int, int, int]]]:
    """The closing read: fastest, slowest, and the gap between them.

    Each line is colored like the dot it names, so the note and the graph read
    as one thing without a legend. The percentage compares *durations*: the
    slowest rep took N% longer than the fastest. That is why the wording says
    "slower" rather than "decrease", since speed goes as 1/time.
    """
    values = [rep_value(r, metric) for r in reps]
    fast_i, slow_i = extremes(values)
    fastest, slowest = values[fast_i], values[slow_i]

    lines = [
        (f"Fastest: rep {fast_i + 1} ({fastest:.2f}s)", _rgb(FAST)),
        (f"Slowest: rep {slow_i + 1} ({slowest:.2f}s)", _rgb(SLOW)),
    ]
    if len(reps) == 1 or slowest - fastest < 1e-9:
        return [lines[0], ("even pace across the set", _rgb(MUTED))]

    pct = (slowest - fastest) / fastest * 100
    lines.append((f"{pct:.0f}% slower from rep {fast_i + 1} to rep {slow_i + 1}",
                  _rgb(MUTED)))
    return lines


def _plot_image(analysis, shown: int, width: int, height: int, *, progressive: bool,
                font=None, y_ticks: int = 4, metric: str = "ascent",
                title: str = "Rep Duration (s)", ylabel: str = "Duration (s)",
                scale: float = 1.0):
    """Render the timing scatter to a BGR array of exactly (height, width)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fp = _plot_font(font)

    def f(size: int) -> dict:
        """Font kwargs at *size*.

        A FontProperties carries its own size and matplotlib applies it *after*
        a plain ``fontsize=``, silently resetting the text to 10pt. Passing a
        sized copy is the only way both survive.
        """
        if fp is None:
            return {"fontsize": size}
        sized = fp.copy()
        sized.set_size(size)
        return {"fontproperties": sized}

    # Matplotlib sizes text in points, which become pixels via the dpi, so a
    # fixed dpi leaves the labels their 608px size on a 1662px panel. Scaling
    # the dpi makes one point the same fraction of the panel at any size.
    dpi = max(30.0, 100.0 * scale)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_subplot(111, facecolor=BG)

    reps = analysis.reps
    n = len(reps)
    xs = np.arange(1, n + 1)
    ys = np.array([rep_value(r, metric) for r in reps])

    # Limits come from every rep, not just the visible ones, so the axes never
    # move as dots appear.
    if n:
        lo, hi = float(ys.min()), float(ys.max())
        span = max(hi - lo, 0.3)
        ax.set_ylim(lo - span * 0.55, hi + span * 0.55)
        ax.set_xlim(0.4, n + 0.6)
        ax.set_xticks(xs)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks, prune=None))

    k = n if not progressive else max(0, min(shown, n))
    complete = k == n and n > 0

    fast_i, slow_i = extremes(list(ys)) if n else (-1, -1)
    distinct = n > 1 and ys[slow_i] - ys[fast_i] > 1e-9

    if k:
        ax.plot(xs[:k], ys[:k], color=ACCENT, lw=2.4, alpha=0.55, zorder=2)

        # Fastest and slowest only mean something once the set is over; during
        # the reveal "fastest so far" would move from dot to dot and mislead.
        colors = [ACCENT] * k
        if complete and distinct:
            colors[fast_i] = FAST
            colors[slow_i] = SLOW

        ax.scatter(xs[:k], ys[:k], s=200, c=colors, zorder=3,
                   edgecolors=BG, linewidths=2)

        for j, (x, y) in enumerate(zip(xs[:k], ys[:k])):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 17),
                        ha="center", color=FG, **f(17))
            if complete and distinct and j in (fast_i, slow_i):
                tag = "fastest" if j == fast_i else "slowest"
                ax.annotate(tag, (x, y), textcoords="offset points", xytext=(0, -30),
                            ha="center", color=FAST if j == fast_i else SLOW, **f(15))

    ax.set_title(title, color=FG, pad=20, **f(28))
    ax.set_xlabel("rep", color=MUTED, **f(18))
    ax.set_ylabel(ylabel, color=MUTED, **f(18))
    ax.tick_params(colors=MUTED, labelsize=16)
    if fp:
        ticks = fp.copy()
        ticks.set_size(16)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(ticks)
    ax.grid(False)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    fig.subplots_adjust(left=0.22, right=0.96, top=0.88, bottom=0.13)
    fig.canvas.draw()

    # Measured, not guessed: the header aligns to wherever matplotlib actually
    # laid the y tick labels, so the two stay aligned when the tick count, the
    # font, or the panel size changes.
    renderer = fig.canvas.get_renderer()
    fig_h = fig.canvas.get_width_height()[1]

    # The leftmost drawn thing, so the bands above and below share one visual
    # left edge with the axis.
    lefts = [t.get_window_extent(renderer).x0
             for t in ax.get_yticklabels() if t.get_text()]
    lefts.append(ax.yaxis.label.get_window_extent(renderer).x0)
    box = {"left": int(min(lefts))}

    # Matplotlib measures from the bottom, rows count from the top. Flipping
    # here is what lets the caller center text against what is actually *seen*
    # rather than against the figure's outer edge, which is mostly padding.
    box["top"] = int(fig_h - ax.title.get_window_extent(renderer).y1)
    box["bottom"] = int(fig_h - ax.xaxis.label.get_window_extent(renderer).y0)

    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    bgr = rgba[..., 2::-1].copy()
    if bgr.shape[0] != height or bgr.shape[1] != width:
        import cv2
        ratio = width / bgr.shape[1]
        bgr = cv2.resize(bgr, (width, height))
        box = {k: int(v * ratio) for k, v in box.items()}
    return bgr, box


def _draw_lines(img, lines, font, *, px: int, x: int, gap: float = 0.40):
    """Draw ``(text, rgb)`` lines vertically centered in *img*, flush left at *x*.

    ``gap`` is the space between baselines as a multiple of the font size, so it
    scales with the text and the output resolution instead of being a pixel
    count that only looks right at one size. It is clamped so the block cannot
    outgrow its band: a large value spreads the lines to fill the space and then
    stops, rather than silently pushing the first line out of view.

    Shared by the header and the closing note, so the two bands use identical
    metrics with no second implementation to drift out of step.
    """
    import cv2

    h, w = img.shape[:2]
    lines = [(t, c) for t, c in lines if t]
    if h < 20 or not lines:
        return img

    def layout(heights):
        """(gap_px, start_y) for a centered block that fits the band."""
        n = len(heights)
        want = int(px * gap)
        if n > 1:
            room = (h - sum(heights)) / (n - 1)
            want = max(0, min(want, int(room)))
        block = sum(heights) + want * (n - 1)
        return want, max(0, (h - block) // 2)

    if font is None:
        fs = px / 34
        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0] for t, _ in lines]
        gap_px, y = layout([sz[1] for sz in sizes])
        for (text, rgb), (_, th) in zip(lines, sizes):
            y += th
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        rgb[::-1], 2, cv2.LINE_AA)
            y += gap_px
        return img

    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    face = ImageFont.truetype(font[0], px, index=font[1])

    boxes = [draw.textbbox((0, 0), text, font=face) for text, _ in lines]
    heights = [b[3] - b[1] for b in boxes]
    gap_px, y = layout(heights)

    for (text, rgb), box, th in zip(lines, boxes, heights):
        draw.text((x - box[0], y - box[1]), text, font=face, fill=rgb)
        y += th + gap_px

    img[:] = np.array(pil)[..., ::-1]
    return img


def _draw_attribution(panel, font, *, text: str, size: int, margin: int, opacity: float):
    """Credit line, tucked into the bottom-right corner of the export.

    Positioned by the text's *ink* box rather than the font's line box, so the
    gap on the right equals the gap underneath. A line box carries the font's
    own ascent and descent padding, which differs from the glyphs drawn.
    """
    import cv2

    if not text:
        return panel

    h, w = panel.shape[:2]
    px = max(9, int(size * w / 720))
    pad = max(6, int(margin * w / 720))
    level = int(255 * max(0.0, min(1.0, opacity)))

    if font is None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 34, 1)
        cv2.putText(panel, text, (w - pad - tw, h - pad), cv2.FONT_HERSHEY_SIMPLEX,
                    px / 34, (level, level, level), 1, cv2.LINE_AA)
        return panel

    from PIL import Image, ImageDraw, ImageFont

    base = Image.fromarray(panel[..., ::-1]).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    face = ImageFont.truetype(font[0], px, index=font[1])

    left, top, right, bottom = draw.textbbox((0, 0), text, font=face)
    draw.text((w - pad - right, h - pad - bottom), text, font=face,
              fill=(255, 255, 255, level))

    panel[:] = np.array(Image.alpha_composite(base, layer).convert("RGB"))[..., ::-1]
    return panel


def build_panels(analysis, width: int, height: int, *, cfg) -> list:
    """One panel image per possible rep count, index 0..N."""
    font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)

    # One square graph, centered on both axes. Everything else arranges itself
    # around it: the header takes the band above, and the same band is mirrored
    # below so the square sits optically centered too.
    side = int(min(width, height) * cfg.PANEL_PLOT_SCALE)
    y0 = (height - side) // 2
    x0 = (width - side) // 2

    panels = []
    for shown in range(len(analysis.reps) + 1):
        panel = np.zeros((height, width, 3), np.uint8)
        panel[:] = _rgb(BG)[::-1]

        plot, box = _plot_image(
            analysis, shown, side, side,
            progressive=cfg.PANEL_PROGRESSIVE, font=font, y_ticks=cfg.PANEL_Y_TICKS,
            metric=cfg.REP_METRIC, title=cfg.PANEL_GRAPH_TITLE,
            ylabel=cfg.PANEL_Y_LABEL, scale=width / 720)
        panel[y0:y0 + side, x0:x0 + side] = plot

        done = analysis.reps[:shown] if cfg.PANEL_PROGRESSIVE else analysis.reps
        avg = float(np.mean([rep_value(r, cfg.REP_METRIC) for r in done])) if done else None
        count = shown if cfg.PANEL_PROGRESSIVE else len(analysis.reps)
        left = x0 + box["left"]
        # Band edges follow the graph's drawn extent, so equal spacing is equal
        # to the eye: figure padding would otherwise skew it.
        head_band, note_band = y0 + box["top"], y0 + box["bottom"]

        # Band 1: the headline numbers.
        _draw_lines(panel[:head_band], [
            (f"{cfg.REP_LABEL}: {count}", _rgb(FG)),
            (f"{cfg.PANEL_AVG_LABEL}: {avg:.2f}s" if avg is not None
             else f"{cfg.PANEL_AVG_LABEL}: —", _rgb(MUTED)),
        ], font, px=int(cfg.PANEL_HEADER_SIZE * width / 720), x=left,
           gap=cfg.PANEL_HEADER_LINE_GAP)

        # Band 3: the closing note, once the set is actually over.
        if shown == len(analysis.reps) and analysis.reps:
            _draw_lines(panel[note_band:], _verdict(analysis.reps, cfg.REP_METRIC), font,
                        px=int(cfg.PANEL_NOTE_SIZE * width / 720), x=left,
                        gap=cfg.PANEL_NOTE_LINE_GAP)

        _draw_attribution(panel, font, text=cfg.ATTRIBUTION,
                          size=cfg.ATTRIBUTION_SIZE, margin=cfg.ATTRIBUTION_MARGIN,
                          opacity=cfg.ATTRIBUTION_OPACITY)
        panels.append(panel)
    return panels
