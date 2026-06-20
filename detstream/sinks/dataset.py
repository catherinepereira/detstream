from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
import cv2
from ..events import SightingEnded, SightingStarted
from . import sinks

log = logging.getLogger(__name__)


# Writes the raw peak-confidence frame of each sighting to disk, one JPEG per sighting under
# a per-feed folder. Unlike the supabase sink it does not draw the detection box, so the
# images are clean for use as a fine-tuning dataset
class DatasetSink:
    def __init__(self, config: dict):
        self.dir = Path(config.get("dir", "dataset"))
        self.quality = int(config.get("quality", 95))

    async def on_sighting_start(self, event: SightingStarted, feed_name: str) -> None:
        pass

    async def on_sighting_end(self, event: SightingEnded) -> None:
        if event.peak_frame is None:
            return
        out_dir = self.dir / event.feed_id
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = out_dir / f"{stamp}_{event.peak_confidence:.2f}.jpg"
        ok = cv2.imwrite(str(path), event.peak_frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            log.warning("dataset sink failed to write %s", path)


@sinks.register("dataset")
def _build(config: dict) -> DatasetSink:
    return DatasetSink(config)
