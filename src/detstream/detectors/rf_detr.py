from __future__ import annotations
import hashlib
import os
from pathlib import Path
import httpx
import numpy as np
from . import Detection, detectors

# Model size to the rfdetr class that loads it. The factory's default is "medium",
# matching the rfdetr docs' default example
_MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

_CACHE_DIR = Path.home() / ".cache" / "detstream" / "rfdetr"

# Ceiling on a downloaded checkpoint, so a wrong or hostile `weights` URL cannot fill the
# disk. RF-DETR checkpoints are well under this
_MAX_WEIGHTS_BYTES = 4 * 1024**3


# RF-DETR run locally through the rfdetr package. COCO-pretrained by default, point
# `weights` at a fine-tuned checkpoint to detect your own classes.
#
# A checkpoint is loaded with torch, which unpickles it, so a malicious `weights` file runs
# arbitrary code on load. Only point `weights` at a file you trust. For a URL, set
# `weights_sha256` to pin the artifact, and a download that does not match is rejected
class RfDetrDetector:
    def __init__(
        self,
        confidence_threshold: float,
        model: str = "medium",
        weights: str = "",
        weights_sha256: str = "",
        class_name: str = "",
        class_names: list[str] | None = None,
    ):
        import rfdetr

        try:
            model_cls = getattr(rfdetr, _MODEL_CLASSES[model])
        except KeyError:
            known = ", ".join(_MODEL_CLASSES)
            raise ValueError(f"unknown rf-detr model {model!r}; known: {known}") from None

        self.threshold = confidence_threshold

        weights_path = _resolve_weights(weights, weights_sha256)
        if weights_path:
            self.model = model_cls(pretrain_weights=weights_path)
        else:
            self.model = model_cls()

        # Resolve the class filter to the set of class ids it accepts, once. A fine-tuned
        # model's labels differ from COCO, so an explicit class_names override wins,
        # otherwise fall back to the package's COCO names. With no class_name filter, any
        # class is accepted and the names are never needed
        if class_name:
            self._target_ids = {i for i, n in enumerate(_resolve_names(class_names)) if n == class_name}
        else:
            self._target_ids = None

    def detect(self, frame: np.ndarray) -> Detection:
        # detstream frames are BGR (OpenCV); rfdetr's predict expects RGB for numpy input.
        # The reversed view has a negative channel stride, which torch.from_numpy rejects,
        # so make it contiguous before handing it over
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        result = self.model.predict(rgb, threshold=self.threshold)

        # supervision.Detections leaves class_id None when a model reports no class info
        class_ids = result.class_id if result.class_id is not None else [None] * len(result.xyxy)

        best = 0.0
        best_xyxy = None
        for xyxy, conf, class_id in zip(result.xyxy, result.confidence, class_ids):
            if self._target_ids is not None and (class_id is None or int(class_id) not in self._target_ids):
                continue
            conf = float(conf)
            if conf > best:
                best = conf
                best_xyxy = xyxy
        box = tuple(float(v) for v in best_xyxy) if best_xyxy is not None else None
        return Detection(present=best >= self.threshold, confidence=best, box=box)


# The label set used to resolve a class_name filter to class ids: the explicit override
# if given, else the package's COCO names (imported lazily, only when a filter is set)
def _resolve_names(class_names: list[str] | None) -> list[str]:
    if class_names:
        return list(class_names)
    from rfdetr.assets.coco_classes import COCO_CLASSES

    return list(COCO_CLASSES)


# Accept a local path or an http(s) URL. A URL is downloaded once to the cache, keyed by
# its hash, and reused thereafter. When expected_sha256 is set, the file (freshly fetched
# or already cached) must match it, so a poisoned cache or a swapped download is caught
# before the checkpoint is loaded. Returns "" when no weights are configured, which the
# caller turns into a COCO-pretrained model
def _resolve_weights(weights: str, expected_sha256: str = "") -> str:
    if not weights:
        return ""
    if weights.startswith(("http://", "https://")):
        return str(_download_cached(weights, expected_sha256))
    return weights


def _download_cached(url: str, expected_sha256: str = "") -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _CACHE_DIR / (hashlib.sha256(url.encode()).hexdigest() + ".pth")
    if cached.exists():
        _verify_digest(cached, expected_sha256)
        return cached
    # Stream to a unique temp file in the same dir, then atomically rename, so an
    # interrupted download can never be mistaken for a complete cached file, and two
    # processes fetching the same URL do not clobber each other's temp file
    tmp = cached.parent / f"{cached.name}.{os.getpid()}.partial"
    try:
        with httpx.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            written = 0
            with open(tmp, "wb") as f:
                for chunk in response.iter_bytes():
                    written += len(chunk)
                    if written > _MAX_WEIGHTS_BYTES:
                        raise ValueError(
                            f"weights download exceeded {_MAX_WEIGHTS_BYTES} bytes; aborting"
                        )
                    f.write(chunk)
        _verify_digest(tmp, expected_sha256)
        os.replace(tmp, cached)
    finally:
        tmp.unlink(missing_ok=True)
    return cached


# Reject the file unless its sha256 matches the expected digest. A no-op when no digest is
# configured, so pinning is opt-in but enforced the moment it is set
def _verify_digest(path: Path, expected_sha256: str) -> None:
    if not expected_sha256:
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha256.lower():
        raise ValueError(
            f"weights sha256 mismatch: expected {expected_sha256.lower()}, got {actual}"
        )


@detectors.register("rf-detr")
def _build(config: dict) -> RfDetrDetector:
    return RfDetrDetector(
        confidence_threshold=config.get("confidence_threshold", 0.4),
        model=config.get("model", "medium"),
        weights=config.get("weights", ""),
        weights_sha256=config.get("weights_sha256", ""),
        class_name=config.get("class_name", ""),
        class_names=config.get("class_names"),
    )
