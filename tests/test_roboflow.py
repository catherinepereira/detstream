import sys
import types
import numpy as np
import pytest


class _FakeClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
        self._outputs = []
        self.last_call = None

    def run_workflow(self, workspace_name, workflow_id, images, parameters, use_cache):
        self.last_call = {
            "workspace_name": workspace_name,
            "workflow_id": workflow_id,
            "images": images,
            "parameters": parameters,
        }
        return self._outputs


@pytest.fixture
def fake_sdk(monkeypatch):
    module = types.ModuleType("inference_sdk")
    module.InferenceHTTPClient = _FakeClient
    monkeypatch.setitem(sys.modules, "inference_sdk", module)
    return module


def _build(fake_sdk, predictions, *, threshold=0.4, class_name="", output_key="predictions", parameters=None):
    from detstream.detectors.roboflow import RoboflowDetector

    det = RoboflowDetector(
        workspace="cats-workspace",
        workflow_id="find-otter",
        api_key="key",
        confidence_threshold=threshold,
        class_name=class_name,
        output_key=output_key,
        parameters=parameters,
    )
    det.client._outputs = [{output_key: {"predictions": predictions}}]
    return det


def _pred(x, y, w, h, conf, cls="otter"):
    return {"x": x, "y": y, "width": w, "height": h, "confidence": conf, "class": cls}


def test_picks_highest_confidence_box(fake_sdk):
    det = _build(fake_sdk, [_pred(5, 5, 2, 2, 0.2), _pred(15, 15, 10, 10, 0.85)])

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert out.present
    assert out.confidence == pytest.approx(0.85)
    assert out.box == pytest.approx((10.0, 10.0, 20.0, 20.0))


def test_below_threshold_not_present(fake_sdk):
    det = _build(fake_sdk, [_pred(1, 1, 2, 2, 0.3)])

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert not out.present
    assert out.confidence == pytest.approx(0.3)


def test_no_predictions_returns_absent(fake_sdk):
    det = _build(fake_sdk, [])

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert not out.present
    assert out.confidence == 0.0
    assert out.box is None


def test_class_name_filters_predictions(fake_sdk):
    det = _build(
        fake_sdk,
        [_pred(5, 5, 2, 2, 0.9, cls="kelp"), _pred(15, 15, 4, 4, 0.5, cls="otter")],
        class_name="otter",
    )

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert out.present
    assert out.confidence == pytest.approx(0.5)
    assert out.box == pytest.approx((13.0, 13.0, 17.0, 17.0))


def test_custom_output_key(fake_sdk):
    det = _build(fake_sdk, [_pred(15, 15, 10, 10, 0.7)], output_key="model_predictions")

    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert out.present
    assert out.confidence == pytest.approx(0.7)


def test_passes_workflow_identifiers(fake_sdk):
    det = _build(fake_sdk, [])

    det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert det.client.last_call["workspace_name"] == "cats-workspace"
    assert det.client.last_call["workflow_id"] == "find-otter"


def test_threshold_sent_as_workflow_confidence(fake_sdk):
    det = _build(fake_sdk, [], threshold=0.3)

    det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert det.client.last_call["parameters"] == {"confidence": 0.3}


def test_extra_parameters_merge_with_confidence(fake_sdk):
    det = _build(fake_sdk, [], threshold=0.3, parameters={"iou_threshold": 0.5})

    det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert det.client.last_call["parameters"] == {"confidence": 0.3, "iou_threshold": 0.5}


def test_explicit_confidence_overrides_threshold(fake_sdk):
    det = _build(fake_sdk, [], threshold=0.4, parameters={"confidence": 0.1})

    det.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert det.client.last_call["parameters"] == {"confidence": 0.1}


def test_missing_workspace_raises(fake_sdk):
    from detstream.detectors.roboflow import RoboflowDetector

    with pytest.raises(ValueError, match="workspace"):
        RoboflowDetector(workspace="", workflow_id="w", api_key="key", confidence_threshold=0.4)


def test_missing_workflow_id_raises(fake_sdk):
    from detstream.detectors.roboflow import RoboflowDetector

    with pytest.raises(ValueError, match="workflow_id"):
        RoboflowDetector(workspace="ws", workflow_id="", api_key="key", confidence_threshold=0.4)


def test_missing_api_key_raises(fake_sdk):
    from detstream.detectors.roboflow import RoboflowDetector

    with pytest.raises(ValueError, match="api_key"):
        RoboflowDetector(workspace="ws", workflow_id="w", api_key="", confidence_threshold=0.4)


def test_registered_under_roboflow(fake_sdk):
    from detstream.detectors import detectors
    from detstream.detectors.roboflow import RoboflowDetector

    det = detectors.create(
        "roboflow",
        {"workspace": "ws", "workflow_id": "w", "api_key": "key", "confidence_threshold": 0.6},
    )
    assert isinstance(det, RoboflowDetector)
    assert det.threshold == 0.6
