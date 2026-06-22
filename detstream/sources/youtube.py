from __future__ import annotations
import os
from collections.abc import AsyncIterator
import cv2
import numpy as np
from . import capture_frames, sources

# FFmpeg 8 enforces a stricter HLS check (extension_picky) that rejects segments whose URL
# has no recognized file extension. YouTube live chunks come from googlevideo.com/videoplayback
# URLs that end in query params, not a .ts extension, so ffmpeg stalls on them and the stream
# yields almost no frames. Allowing any segment extension restores reading these live streams.
# Set before any VideoCapture opens, and only if the caller has not set its own options
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "allowed_segment_extensions;ALL")


def resolve_stream(youtube_url: str) -> str:
    import yt_dlp

    # remote_components lets yt-dlp fetch and run its JS challenge solver
    opts = {
        "quiet": True,
        "format": "best[protocol^=m3u8]/best",
        "noplaylist": True,
        "remote_components": ["ejs:github"],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
    return info["url"]


# A YouTube live stream. yt-dlp resolves the page to an HLS manifest, which OpenCV reads
# The manifest URL expires, so each reconnect re-resolves rather than reusing the old one
class YouTubeSource:
    def __init__(self, url: str, interval_s: float):
        self.url = url
        self.interval_s = interval_s

    def frames(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        def open_capture() -> cv2.VideoCapture:
            hls = resolve_stream(self.url)
            return cv2.VideoCapture(hls)

        return capture_frames(open_capture, self.interval_s, label=f"youtube {self.url}")


@sources.register("youtube")
def _build(config: dict) -> YouTubeSource:
    return YouTubeSource(url=config["url"], interval_s=config.get("interval_s", 2.0))
