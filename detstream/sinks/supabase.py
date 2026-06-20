from __future__ import annotations
import logging
import os
import uuid
import cv2
from ..annotate import annotate
from ..detectors import Detection
from ..events import SightingEnded, SightingStarted
from . import sinks

log = logging.getLogger(__name__)


# Writes a sightings row on start and patches ended_at on end. The peak confidence frame is
# downscaled, uploaded to Supabase Storage, and its URL is stored on the row. The website
# subscribes to the table via Realtime
class SupabaseSink:
    def __init__(self, config: dict):
        from supabase import create_client

        url = os.environ.get("DETSTREAM_SUPABASE_URL", "")
        key = os.environ.get("DETSTREAM_SUPABASE_KEY", "")
        if not url or not key:
            raise ValueError(
                "supabase sink needs DETSTREAM_SUPABASE_URL and DETSTREAM_SUPABASE_KEY"
            )
        self.client = create_client(url, key)
        self.bucket = config.get("bucket", "thumbnails")
        self.detector_label = config.get("detector_label", "")
        self.thumbnail_width = int(config.get("thumbnail_width", 960))
        self.thumbnail_quality = int(config.get("thumbnail_quality", 70))
        self.retention_hours = float(config.get("retention_hours", 3))
        self._open_rows: dict[str, str] = {}  # feed_id -> sighting row id

    async def on_sighting_start(self, event: SightingStarted, feed_name: str) -> None:
        # No thumbnail yet, it is uploaded on end from the peak-confidence frame
        row = (
            self.client.table("sightings")
            .insert(
                {
                    "cam_id": event.feed_id,
                    "confidence": event.confidence,
                    "detector": self.detector_label,
                    "label": event.label,
                }
            )
            .execute()
        )
        self._open_rows[event.feed_id] = row.data[0]["id"]

    async def on_sighting_end(self, event: SightingEnded) -> None:
        row_id = self._open_rows.pop(event.feed_id, None)
        if row_id is None:
            return
        thumb_url = self._upload_thumbnail(
            event.feed_id, event.peak_frame, event.peak_confidence, event.peak_box
        )
        self.client.table("sightings").update(
            {
                "ended_at": "now()",
                "confidence": event.peak_confidence,
                "thumbnail": thumb_url,
            }
        ).eq("id", row_id).execute()

    def _upload_thumbnail(self, feed_id: str, frame, confidence, box) -> str | None:
        if frame is None:
            return None
        # The website wants the box drawn on its thumbnail, so annotate here (the runner
        # keeps the captured frame raw). annotate returns the frame unchanged when box is None
        frame = annotate(frame, Detection(True, confidence, box), self.detector_label)
        frame = self._downscale(frame)
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.thumbnail_quality]
        )
        if not ok:
            return None
        path = f"{feed_id}/{uuid.uuid4().hex}.jpg"
        # A storage failure (bucket full/missing) must not lose the row's ended_at patch,
        # so the upload is isolated and the row keeps a null thumbnail
        try:
            self.client.storage.from_(self.bucket).upload(
                path, buf.tobytes(), {"content-type": "image/jpeg"}
            )
            return self.client.storage.from_(self.bucket).get_public_url(path)
        except Exception as e:
            log.warning("thumbnail upload failed for %s: %s", feed_id, e)
            return None

    def _downscale(self, frame):
        h, w = frame.shape[:2]
        if w <= self.thumbnail_width:
            return frame
        new_h = round(h * self.thumbnail_width / w)
        return cv2.resize(frame, (self.thumbnail_width, new_h), interpolation=cv2.INTER_AREA)

    # Delete sightings and their thumbnail objects older than retention_hours
    def cleanup(self) -> None:
        if self.retention_hours <= 0:
            return
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)).isoformat()
        old = (
            self.client.table("sightings")
            .select("id, thumbnail")
            .lt("started_at", cutoff)
            .execute()
            .data
        )
        if not old:
            return

        paths = []
        for r in old:
            url = r.get("thumbnail") or ""
            marker = f"/{self.bucket}/"
            i = url.find(marker)
            if i != -1:
                paths.append(url[i + len(marker) :])
        for i in range(0, len(paths), 100):
            self.client.storage.from_(self.bucket).remove(paths[i : i + 100])

        ids = [r["id"] for r in old]
        for i in range(0, len(ids), 100):
            self.client.table("sightings").delete().in_("id", ids[i : i + 100]).execute()
        log.info("cleanup: removed %d sighting(s) older than %gh", len(ids), self.retention_hours)


@sinks.register("supabase")
def _build(config: dict) -> SupabaseSink:
    return SupabaseSink(config)
