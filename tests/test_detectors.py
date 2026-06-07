import sys
import types
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
