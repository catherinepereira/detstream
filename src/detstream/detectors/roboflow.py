from __future__ import annotations
import os
import numpy as np
from . import Detection, detectors

_DEFAULT_API_URL = "https://detect.roboflow.com"


class RoboflowDetector:
    def __init__(
        self,
        workspace: str,
        workflow_id: str,
        api_key: str,
        confidence_threshold: float,
        class_name: str = "",
        output_key: str = "predictions",
        parameters: dict | None = None,
        api_url: str = "",
    ):
        from inference_sdk import InferenceHTTPClient

        if not workspace:
            raise ValueError("roboflow detector needs a workspace")
        if not workflow_id:
            raise ValueError("roboflow detector needs a workflow_id")
        if not api_key:
            raise ValueError("roboflow detector needs an api_key (set ROBOFLOW_API_KEY)")
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.threshold = confidence_threshold
        # Optional: restrict to one class. Empty means accept any predicted class
        self.class_name = class_name
        # Name of the workflow's detection output block, as set when the workflow was built
        self.output_key = output_key
        # Extra workflow inputs, e.g. {"iou_threshold": 0.5}. The threshold is sent as the
        # workflow's `confidence` input so the server filters at the same cutoff detstream
        # uses, unless `parameters` overrides it explicitly
        self.parameters = {"confidence": confidence_threshold, **(parameters or {})}
        self.client = InferenceHTTPClient(
            api_url=api_url or _DEFAULT_API_URL, api_key=api_key
        )

    def detect(self, frame: np.ndarray) -> Detection:
        # run_workflow returns one output dict per input image, and detect sends one frame
        outputs = self.client.run_workflow(
            workspace_name=self.workspace,
            workflow_id=self.workflow_id,
            images={"image": frame},
            parameters=self.parameters,
            use_cache=True,
        )[0]
        # An object-detection block outputs {"predictions": [...], "image": {...}}
        predictions = outputs[self.output_key]["predictions"]

        best = 0.0
        box = None
        for p in predictions:
            if self.class_name and p["class"] != self.class_name:
                continue
            conf = p["confidence"]
            if conf > best:
                best = conf
                box = _to_xyxy(p)
        return Detection(present=best >= self.threshold, confidence=best, box=box)


# Roboflow reports a box as its center (x, y) plus width and height. detstream boxes
# are corner xyxy, matching the YOLO-World detector
def _to_xyxy(p: dict) -> tuple[float, float, float, float]:
    x, y, w, h = p["x"], p["y"], p["width"], p["height"]
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


@detectors.register("roboflow")
def _build(config: dict) -> RoboflowDetector:
    return RoboflowDetector(
        workspace=config.get("workspace", ""),
        workflow_id=config.get("workflow_id", ""),
        api_key=config.get("api_key") or os.environ.get("ROBOFLOW_API_KEY", ""),
        confidence_threshold=config.get("confidence_threshold", 0.4),
        class_name=config.get("class_name", ""),
        output_key=config.get("output_key", "predictions"),
        parameters=config.get("parameters"),
        api_url=config.get("api_url", ""),
    )
