"""The Gateway request and the shape of what comes back."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

CONTENT_OBJECT_FRAMES = "vitpose_plus.pose.frames"


@dataclass
class PoseResult:
    """Frames of poses, keyed by source-frame index."""

    payload: dict
    frames: list[dict]
    by_index: dict[int, list[dict]]
    elapsed: float
    usage: dict | None

    @property
    def track_ids(self) -> list[int]:
        ids = {p.get("track_id") for f in self.frames for p in f["persons"]}
        return sorted(i for i in ids if i is not None)

    @property
    def n_posed(self) -> int:
        return sum(1 for f in self.frames if f["persons"])


def build_request(video: Path, *, every_frame: bool, fps: float, n_frames: int,
                  video_fps: float, video_max_frames, precision: int) -> tuple[str, dict]:
    """Return ``(base64_video, extra_body)``.

    ``video_fps`` is the *detector* cadence, not a sampling rate; pose runs on
    every decoded frame regardless. It reaches stride 1 (detect on every frame)
    as soon as it is >= the decoded rate. ``video_max_frames`` is what actually
    controls how many frames are decoded, posed and billed.
    """
    if every_frame:
        video_fps, video_max_frames = fps, n_frames

    extra_body = {"method": "pose", "video_fps": video_fps, "precision": precision}
    if video_max_frames is not None:
        extra_body["video_max_frames"] = video_max_frames

    return base64.b64encode(video.read_bytes()).decode("ascii"), extra_body


def unwrap(payload: dict) -> tuple[list[dict], dict[int, list[dict]]]:
    """Pull the frame list out of the response envelope.

    Checked rather than assumed: a wrong `content.object` means an image was
    sent where a video was intended, which is clearer to say here than to let
    it surface as a KeyError during rendering.
    """
    try:
        entry = payload["data"][0]
        content = entry["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response envelope: {json.dumps(payload)[:400]}") from exc

    got = content.get("object")
    if got != CONTENT_OBJECT_FRAMES:
        raise RuntimeError(
            f"Expected a video payload ({CONTENT_OBJECT_FRAMES}), got {got!r}. "
            "Did the request send an image_url instead of a video_url?"
        )

    frames = content["items"]
    return frames, {f["index"]: f["persons"] for f in frames}


def request_poses(client, *, model: str, video_b64: str, extra_body: dict) -> dict:
    """One blocking chat-completions call, returning the parsed JSON payload."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [{
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            }],
        }],
        response_format={"type": "json_object"},
        extra_body=extra_body,
    )
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    return payload, usage


def cache_path(video: Path, cache_dir: Path, *, model: str, extra_body: dict) -> Path:
    """Where this exact request's poses live.

    Keyed on the converted video's identity *and* every field that shapes the
    result, so a cache hit can only mean "same clip, same model, same request".
    """
    stat = video.stat()
    key = json.dumps(
        {
            "video": video.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "model": model,
            "extra_body": extra_body,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return cache_dir / f"poses.{digest}.json"


def save_cache(path: Path, payload: dict, usage: dict | None, timing_dict: dict,
               *, stamp: str) -> None:
    """Store the response plus the timings it was actually measured with."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created": stamp,
        "payload": payload,
        "usage": usage,
        "timing": timing_dict,
    }))


def load_cache(path: Path) -> tuple[dict, dict | None, dict, str] | None:
    """Read a cached response, or None if it is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
        return blob["payload"], blob.get("usage"), blob.get("timing") or {}, blob.get(
            "created", "unknown")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _area(box) -> float:
    return float(box[2] * box[3])


def _iou(a, b) -> float:
    """Intersection over union for two normalized ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def dedupe(frames: list[dict], *, max_persons: int | None, iou: float) -> int:
    """Collapse duplicate boxes per frame. Returns how many were dropped.

    The tracker hands identity over by running the old and new track together
    for a few frames, so one body briefly reports twice at IoU ~0.97. Which of
    the pair survives is decided by the best match to the previous frame,
    falling back to the larger box: size alone flickers between two boxes that
    differ by well under a percent, while continuity keeps one body selected.
    """
    previous = None
    dropped = 0

    for frame in frames:
        persons = frame.get("persons") or []
        if not persons:
            continue

        order = sorted(
            persons,
            key=lambda p: (
                -(_iou(p["bbox_xywh"], previous) if previous is not None else 0.0),
                -_area(p["bbox_xywh"]),
            ),
        )

        kept: list[dict] = []
        for person in order:
            if all(_iou(person["bbox_xywh"], k["bbox_xywh"]) < iou for k in kept):
                kept.append(person)

        if max_persons:
            kept = kept[:max_persons]

        dropped += len(persons) - len(kept)
        frame["persons"] = kept
        if kept:
            previous = kept[0]["bbox_xywh"]

    return dropped
