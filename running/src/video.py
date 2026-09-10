"""Probing and (cached) conversion of the source clip."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn


@dataclass(frozen=True)
class VideoInfo:
    """Geometry of a decoded clip, as OpenCV will see it."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def __str__(self) -> str:
        return (
            f"{self.width}x{self.height} @ {self.fps:.2f} fps, "
            f"{self.n_frames} frames, {self.duration:.1f}s"
        )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stderr[-3000:]}")
    return proc


def probe_source(path: Path) -> dict:
    """ffprobe the source, including the rotation OpenCV would miss."""
    proc = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames,duration,pix_fmt",
        "-show_entries", "stream_side_data=rotation",
        "-of", "json", str(path),
    ])
    stream = json.loads(proc.stdout)["streams"][0]
    rotation = 0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "pix_fmt": stream.get("pix_fmt"),
        "n_frames": int(stream.get("nb_frames") or 0),
        "duration": float(stream.get("duration") or 0.0),
        "rotation": rotation,
    }


def cache_path(src: Path, cache_dir: Path, *, target_height, trim_seconds, crf) -> Path:
    """Deterministic name for the converted MP4.

    Keyed on the source's identity *and* the settings that shaped the output, so
    a changed height or CRF produces a different file rather than silently
    reusing the previous one.
    """
    stat = src.stat()
    key = json.dumps(
        {
            "name": src.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "target_height": target_height,
            "trim_seconds": trim_seconds,
            "crf": crf,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return cache_dir / f"{src.stem}.{digest}.mp4"


def convert(
    src: Path,
    dst: Path,
    *,
    target_height: int | None,
    trim_seconds: float | None,
    crf: int,
    total_frames: int,
    console,
) -> None:
    """Transcode to H.264 MP4, printing ffmpeg's own frame counter as progress."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")

    cmd = ["ffmpeg", "-y", "-nostats", "-loglevel", "error", "-progress", "pipe:1", "-i", str(src)]
    if trim_seconds:
        cmd += ["-t", str(trim_seconds)]
    if target_height:
        # Width follows the aspect ratio; `-2` keeps it even (H.264 needs that)
        # and `min(...,ih)` means this only ever downscales. `ih` is the
        # *rotated* height. ffmpeg applies the container's rotation before
        # filters run, so a clip stored 2954x1662 scales as the portrait it is.
        cmd += ["-vf", f"scale=-2:'min({target_height},ih)'"]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",   # 10-bit HEVC -> 8-bit H.264, which everything decodes
        "-an",                   # audio is dead weight in the upload
        str(tmp),
    ]

    columns = [
        TextColumn("[cyan]converting[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} frames"),
        TimeElapsedColumn(),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with Progress(*columns, console=console, transient=True) as progress:
            task = progress.add_task("convert", total=total_frames or None)
            for line in proc.stdout:
                key, _, value = line.strip().partition("=")
                if key == "frame" and value.isdigit():
                    progress.update(task, completed=min(int(value), total_frames or int(value)))
            proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()

    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg conversion failed:\n{proc.stderr.read()[-3000:]}")

    tmp.replace(dst)   # atomic: a killed run never leaves a half-file in the cache


def inspect(path: Path) -> VideoInfo:
    """Read geometry the way the renderer will, i.e. through OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    info = VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def encode_h264(src: Path, dst: Path, *, crf: int) -> None:
    """Re-encode the rendered video.

    OpenCV's VideoWriter emits MPEG-4 Part 2, which most players and every
    browser refuse. This is the step that makes the output shareable.
    """
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", str(dst),
    ])
