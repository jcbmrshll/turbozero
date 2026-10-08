"""Turning renderable values into bytes the dashboard can show. Used by the client
for media a run logs directly, and by the server for the episodes it renders."""

import io
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Video:
    """Frames (PIL images) of an episode, shown as an animated gif."""

    frames: list[Any]
    fps: float = 25


def encode_media(value: Any) -> tuple[str, bytes] | None:
    """(content type, bytes) for a value the monitor can display, or None if the
    value isn't media."""
    if isinstance(value, list):
        # a bare list of frames
        is_frames = bool(value) and hasattr(value[0], "save")
        return encode_media(Video(value)) if is_frames else None
    if isinstance(value, Video):
        buf = io.BytesIO()
        value.frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=value.frames[1:],
            duration=round(1000 / value.fps),
            loop=0,
        )
        return "image/gif", buf.getvalue()
    # structured data the dashboard shows as-is
    if isinstance(value, dict):
        return "application/json", json.dumps(value).encode()
    buf = io.BytesIO()
    # matplotlib figure
    if hasattr(value, "savefig"):
        value.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        return "image/png", buf.getvalue()
    # PIL image
    if hasattr(value, "save") and hasattr(value, "mode"):
        value.save(buf, format="PNG")
        return "image/png", buf.getvalue()
    return None
