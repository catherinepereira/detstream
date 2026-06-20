import sys
import types
from pathlib import Path
import numpy as np
import pytest


class _Tensor:
    # Minimal stand-in for the torch tensors ultralytics returns: indexing, argmax,
    # float(), and a .cpu() that returns self (no device move in tests)
    def __init__(self, data):
        self._a = np.asarray(data, dtype=np.float32)

    def __getitem__(self, i):
        return _Tensor(self._a[i])

    def __len__(self):
        return len(self._a)

    def __float__(self):
        return float(self._a)

    def argmax(self):
        return int(self._a.argmax())

    def cpu(self):
        return self

    def tolist(self):
        return self._a.tolist()


class _Boxes:
    def __init__(self, confs, xyxy):
        self.conf = _Tensor(confs)
        self.xyxy = _Tensor(xyxy)

    def __len__(self):
        return len(self.conf)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    last_predict_conf = None

    def __init__(self, weights):
        self.weights = weights

    def set_classes(self, classes):
        self.classes = classes

    def predict(self, frame, conf, verbose):
        type(self).last_predict_conf = conf
        return self._results


@pytest.fixture
def fake_ultralytics(monkeypatch):
    module = types.ModuleType("ultralytics")
    module.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return module


def _build(fake_ultralytics, results):
    from detstream.detectors.yolo_world import YoloWorldDetector

    det = YoloWorldDetector(prompt="otter", confidence_threshold=0.4)
    det.model._results = results
    return det


def test_picks_highest_confidence_box(fake_ultralytics):
    boxes = _Boxes([0.2, 0.85, 0.5], [[0, 0, 1, 1], [10, 10, 20, 20], [5, 5, 6, 6]])
    det = _build(fake_ultralytics, [_Result(boxes)])

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert out.present
    assert out.confidence == pytest.approx(0.85, abs=1e-5)
    assert out.box == pytest.approx((10.0, 10.0, 20.0, 20.0))


def test_below_threshold_not_present(fake_ultralytics):
    boxes = _Boxes([0.1, 0.3], [[0, 0, 1, 1], [2, 2, 3, 3]])
    det = _build(fake_ultralytics, [_Result(boxes)])

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert not out.present
    assert out.confidence == pytest.approx(0.3, abs=1e-5)


def test_no_boxes_returns_absent(fake_ultralytics):
    det = _build(fake_ultralytics, [_Result(None), _Result(_Boxes([], np.empty((0, 4))))])

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert not out.present
    assert out.confidence == 0.0
    assert out.box is None


# --- rf-detr detector ---


class _Detections:
    # Stand-in for supervision.Detections: the three parallel arrays detect() reads
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.class_id = np.asarray(class_id, dtype=np.int64)


class _FakeRFDETRMedium:
    last_predict_frame = None
    last_pretrain_weights = "__unset__"

    def __init__(self, pretrain_weights=None):
        type(self).last_pretrain_weights = pretrain_weights

    def predict(self, frame, threshold):
        type(self).last_predict_frame = frame
        return self._detections


@pytest.fixture
def fake_rfdetr(monkeypatch):
    module = types.ModuleType("rfdetr")
    module.RFDETRMedium = _FakeRFDETRMedium
    coco = types.ModuleType("rfdetr.assets.coco_classes")
    # class_id 0 -> "otter", 1 -> "rock", so tests can filter by name
    coco.COCO_CLASSES = ["otter", "rock"]
    assets = types.ModuleType("rfdetr.assets")
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    monkeypatch.setitem(sys.modules, "rfdetr.assets", assets)
    monkeypatch.setitem(sys.modules, "rfdetr.assets.coco_classes", coco)
    # Reset captured state between tests
    _FakeRFDETRMedium.last_predict_frame = None
    _FakeRFDETRMedium.last_pretrain_weights = "__unset__"
    return module


def _build_rfdetr(fake_rfdetr, detections, **kwargs):
    from detstream.detectors.rf_detr import RfDetrDetector

    det = RfDetrDetector(confidence_threshold=kwargs.pop("confidence_threshold", 0.4), **kwargs)
    det.model._detections = detections
    return det


def test_rfdetr_picks_highest_confidence_box(fake_rfdetr):
    dets = _Detections(
        xyxy=[[0, 0, 1, 1], [10, 10, 20, 20], [5, 5, 6, 6]],
        confidence=[0.2, 0.85, 0.5],
        class_id=[0, 0, 1],
    )
    det = _build_rfdetr(fake_rfdetr, dets)

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert out.present
    assert out.confidence == pytest.approx(0.85, abs=1e-5)
    assert out.box == pytest.approx((10.0, 10.0, 20.0, 20.0))


def test_rfdetr_class_name_filters_other_classes(fake_rfdetr):
    # The highest-confidence box is class 1 ("rock"). Filtering to "otter" must skip it
    # and pick the best otter box instead
    dets = _Detections(
        xyxy=[[0, 0, 1, 1], [10, 10, 20, 20]],
        confidence=[0.6, 0.95],
        class_id=[0, 1],
    )
    det = _build_rfdetr(fake_rfdetr, dets, class_name="otter")

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert out.present
    assert out.confidence == pytest.approx(0.6, abs=1e-5)
    assert out.box == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_rfdetr_below_threshold_not_present(fake_rfdetr):
    dets = _Detections(xyxy=[[0, 0, 1, 1]], confidence=[0.3], class_id=[0])
    det = _build_rfdetr(fake_rfdetr, dets, confidence_threshold=0.4)

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert not out.present
    assert out.confidence == pytest.approx(0.3, abs=1e-5)


def test_rfdetr_no_detections_returns_absent(fake_rfdetr):
    dets = _Detections(xyxy=np.empty((0, 4)), confidence=[], class_id=[])
    det = _build_rfdetr(fake_rfdetr, dets)

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert not out.present
    assert out.confidence == 0.0
    assert out.box is None


def test_rfdetr_converts_bgr_to_rgb(fake_rfdetr):
    dets = _Detections(xyxy=np.empty((0, 4)), confidence=[], class_id=[])
    det = _build_rfdetr(fake_rfdetr, dets)
    # A frame whose three channels are distinct, so a BGR->RGB swap is observable
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    frame[0, 0] = [10, 20, 30]  # B, G, R

    det.detect(frame)

    handed = _FakeRFDETRMedium.last_predict_frame
    assert list(handed[0, 0]) == [30, 20, 10]
    # The reversed view has a negative channel stride that the real model would reject,
    # so detect must hand over a contiguous copy
    assert handed.flags["C_CONTIGUOUS"]


def test_rfdetr_no_weights_uses_pretrained(fake_rfdetr):
    _build_rfdetr(fake_rfdetr, _Detections(np.empty((0, 4)), [], []))

    assert _FakeRFDETRMedium.last_pretrain_weights is None


def test_rfdetr_local_weights_passed_through(fake_rfdetr):
    _build_rfdetr(fake_rfdetr, _Detections(np.empty((0, 4)), [], []), weights="./otter.pth")

    assert _FakeRFDETRMedium.last_pretrain_weights == "./otter.pth"


def test_rfdetr_registered():
    from detstream.detectors import detectors

    assert "rf-detr" in detectors._factories


class _FakeStream:
    # Context manager standing in for httpx.stream(...). Yields the body in chunks, and
    # raises like httpx would when an error is configured
    def __init__(self, body=b"", status_error=None, body_error=None):
        self._body = body
        self._status_error = status_error
        self._body_error = body_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def iter_bytes(self):
        if self._body_error is not None:
            raise self._body_error
        yield self._body


def test_rfdetr_url_weights_downloaded_and_cached(fake_rfdetr, monkeypatch, tmp_path):
    from detstream.detectors import rf_detr

    monkeypatch.setattr(rf_detr, "_CACHE_DIR", tmp_path)
    calls = []

    def fake_stream(method, url, **kwargs):
        calls.append(url)
        return _FakeStream(b"fake-weights")

    monkeypatch.setattr(rf_detr.httpx, "stream", fake_stream)

    url = "https://example.com/otter-rfdetr.pth"
    _build_rfdetr(fake_rfdetr, _Detections(np.empty((0, 4)), [], []), weights=url)
    cached = _FakeRFDETRMedium.last_pretrain_weights

    # The model was handed a local cache path, the file holds the downloaded bytes, and
    # no .partial temp file was left behind
    assert cached and Path(cached).parent == tmp_path
    assert Path(cached).read_bytes() == b"fake-weights"
    assert not list(tmp_path.glob("*.partial"))

    # A second build with the same URL reuses the cache without re-downloading
    _build_rfdetr(fake_rfdetr, _Detections(np.empty((0, 4)), [], []), weights=url)
    assert calls == [url]


def test_rfdetr_download_error_propagates_and_cleans_up(fake_rfdetr, monkeypatch, tmp_path):
    from detstream.detectors import rf_detr
    import httpx

    monkeypatch.setattr(rf_detr, "_CACHE_DIR", tmp_path)
    err = httpx.HTTPStatusError("404", request=None, response=None)
    monkeypatch.setattr(
        rf_detr.httpx, "stream", lambda *a, **k: _FakeStream(status_error=err)
    )

    with pytest.raises(httpx.HTTPStatusError):
        _build_rfdetr(
            fake_rfdetr, _Detections(np.empty((0, 4)), [], []),
            weights="https://example.com/missing.pth",
        )
    # No cached file and no leftover .partial after the failed download
    assert not list(tmp_path.iterdir())


def test_rfdetr_download_size_cap(fake_rfdetr, monkeypatch, tmp_path):
    from detstream.detectors import rf_detr

    monkeypatch.setattr(rf_detr, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(rf_detr, "_MAX_WEIGHTS_BYTES", 8)
    monkeypatch.setattr(
        rf_detr.httpx, "stream", lambda *a, **k: _FakeStream(b"way too many bytes")
    )

    with pytest.raises(ValueError, match="exceeded"):
        _build_rfdetr(
            fake_rfdetr, _Detections(np.empty((0, 4)), [], []),
            weights="https://example.com/huge.pth",
        )
    assert not list(tmp_path.iterdir())


def test_rfdetr_sha256_mismatch_rejected(fake_rfdetr, monkeypatch, tmp_path):
    from detstream.detectors import rf_detr

    monkeypatch.setattr(rf_detr, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        rf_detr.httpx, "stream", lambda *a, **k: _FakeStream(b"fake-weights")
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        _build_rfdetr(
            fake_rfdetr, _Detections(np.empty((0, 4)), [], []),
            weights="https://example.com/otter.pth",
            weights_sha256="0" * 64,
        )
    # The download is discarded, not cached, when the digest does not match
    assert not list(tmp_path.iterdir())


def test_rfdetr_sha256_match_accepted(fake_rfdetr, monkeypatch, tmp_path):
    import hashlib
    from detstream.detectors import rf_detr

    body = b"fake-weights"
    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(rf_detr, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(rf_detr.httpx, "stream", lambda *a, **k: _FakeStream(body))

    _build_rfdetr(
        fake_rfdetr, _Detections(np.empty((0, 4)), [], []),
        weights="https://example.com/otter.pth",
        weights_sha256=digest,
    )
    cached = Path(_FakeRFDETRMedium.last_pretrain_weights)
    assert cached.read_bytes() == body


# --- yoloe detector ---


class _BoxesWithCls:
    # Like _Boxes but also carries the per-box class index yoloe reads to label a match
    def __init__(self, confs, xyxy, cls):
        self.conf = _Tensor(confs)
        self.xyxy = _Tensor(xyxy)
        self.cls = _Tensor(cls)

    def __len__(self):
        return len(self.conf)


class _FakeYOLOE:
    last_set_classes = None

    def __init__(self, weights):
        self.weights = weights

    def get_text_pe(self, texts):
        # Stand-in for the text-prompt embeddings; detect() never inspects them
        return [f"pe:{t}" for t in texts]

    def set_classes(self, classes, embeddings):
        type(self).last_set_classes = (list(classes), embeddings)
        self.classes = classes

    def predict(self, frame, conf, verbose):
        return self._results


@pytest.fixture
def fake_yoloe(monkeypatch):
    module = types.ModuleType("ultralytics")
    module.YOLOE = _FakeYOLOE
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    _FakeYOLOE.last_set_classes = None
    return module


def _build_yoloe(fake_yoloe, results, **kwargs):
    from detstream.detectors.yoloe import YoloeDetector

    kwargs.setdefault("prompts", ["deer", "bird", "fox"])
    kwargs.setdefault("confidence_threshold", 0.4)
    det = YoloeDetector(**kwargs)
    det.model._results = results
    return det


def test_yoloe_labels_the_matched_class(fake_yoloe):
    # Top box (0.85) is class 1 -> "bird"
    boxes = _BoxesWithCls([0.2, 0.85, 0.5], [[0, 0, 1, 1], [10, 10, 20, 20], [5, 5, 6, 6]], [0, 1, 2])
    det = _build_yoloe(fake_yoloe, [_Result(boxes)])

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert out.present
    assert out.confidence == pytest.approx(0.85, abs=1e-5)
    assert out.box == pytest.approx((10.0, 10.0, 20.0, 20.0))
    assert out.label == "bird"


def test_yoloe_sets_text_prompt_embeddings_once(fake_yoloe):
    _build_yoloe(fake_yoloe, [_Result(_BoxesWithCls([], np.empty((0, 4)), []))])

    names, embeddings = _FakeYOLOE.last_set_classes
    assert names == ["deer", "bird", "fox"]
    assert embeddings == ["pe:deer", "pe:bird", "pe:fox"]


def test_yoloe_no_boxes_returns_absent_unlabelled(fake_yoloe):
    det = _build_yoloe(fake_yoloe, [_Result(None), _Result(_BoxesWithCls([], np.empty((0, 4)), []))])

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert not out.present
    assert out.confidence == 0.0
    assert out.box is None
    assert out.label is None


def test_yoloe_below_threshold_still_labels(fake_yoloe):
    # Present is False below threshold, but the label/confidence still report what was seen
    boxes = _BoxesWithCls([0.3], [[0, 0, 1, 1]], [2])
    det = _build_yoloe(fake_yoloe, [_Result(boxes)], confidence_threshold=0.4)

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert not out.present
    assert out.label == "fox"


def test_yoloe_out_of_range_class_index_degrades_to_no_label(fake_yoloe):
    # A class index past the prompt list must not raise, just yield no label
    boxes = _BoxesWithCls([0.9], [[0, 0, 1, 1]], [7])
    det = _build_yoloe(fake_yoloe, [_Result(boxes)])

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert out.present
    assert out.label is None


def test_yoloe_single_prompt_accepted(fake_yoloe):
    from detstream.detectors import detectors

    det = detectors.create("yoloe", {"prompt": "deer", "confidence_threshold": 0.4})
    assert det.prompts == ["deer"]


def test_yoloe_no_prompts_raises(fake_yoloe):
    from detstream.detectors.yoloe import YoloeDetector

    with pytest.raises(ValueError):
        YoloeDetector(prompts=[], confidence_threshold=0.4)


def test_yoloe_registered():
    from detstream.detectors import detectors

    assert "yoloe" in detectors._factories
