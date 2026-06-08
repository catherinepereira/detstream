# detstream

Modular object detection framework for live video feeds.

## Pipeline

1. **Source**: Where frames come from. `youtube`, `stream` (RTSP/HLS/HTTP), `file` (a video
  file or webcam device)
2. **Detector**: What to look for. `yolo-world`, an open-vocabulary model prompted with a
  word or phrase (`person`, `forklift`, `bird`); `roboflow`, a Workflow you built on Roboflow,
  run through its hosted inference API. Bring your own by registering one (see Plugins).
3. **Tracker**: When a detection counts as a sighting. Hysteresis and cooldown give you one
  alert per sighting instead of one per frame, which is what keeps alerts from becoming spam.
4. **Sinks**: Where alerts go. `console`, `supabase` (rows + thumbnails for a website),
  `discord` (rich embeds).

## Install

```bash
pip install detstream                          # core only
pip install "detstream[yolo,youtube,supabase]" # for example, everything otterwatch uses
```

The core install pulls what every feed needs: `numpy`, `opencv-python-headless`
(decoding and annotation), `pydantic` (config), `pyyaml`, and `httpx`.

| Extra | Enables | Pulls in |
| --- | --- | --- |
| `yolo` | the `yolo-world` detector | `ultralytics>=8.3.0`, `torch>=2.2.0`. ultralytics fetches CLIP on first use of the detector |
| `youtube` | the `youtube` source | `yt-dlp>=2026.03.17` and `deno>=2.8.0`, which ships the deno binary yt-dlp runs to solve YouTube's stream challenge |
| `roboflow` | the `roboflow` detector | `inference-sdk>=1.3.0` |
| `supabase` | the `supabase` sink | `supabase>=2.4.0` |

## Run

```bash
detstream --config examples/otters.yaml
```

A config lists feeds and shared sink settings:

```yaml
feeds:
  - id: monterey-otters
    name: Monterey Sea Otters
    source:   { type: youtube, url: "https://www.youtube.com/watch?v=abbR-Ttd-cA" }
    detector: { type: yolo-world, prompt: otter swimming, confidence_threshold: 0.4 }
    debounce: { enter_frames: 3, exit_frames: 5, cooldown_s: 120, sample_interval_s: 2 }
    sinks: [console, supabase]
sinks:
  supabase: { bucket: thumbnails, detector_label: yolo-world, retention_hours: 3, thumbnail_width: 960 }
```

Credentials and webhook URLs are configured in `.env`: `DETSTREAM_SUPABASE_URL`, `DETSTREAM_SUPABASE_KEY`, and `DETSTREAM_DISCORD_WEBHOOK_URL`.

The `roboflow` detector runs a Workflow you built on Roboflow through its hosted inference API.
Install the extra (`pip install "detstream[roboflow]"`), then give it your workspace and
workflow ID (both shown in the Workflow's deploy snippet):

```yaml
detector:
  type: roboflow
  workspace: cats-workspace-zqd47
  workflow_id: find-otter
  output_key: predictions    # name of the workflow's detection output block
  class_name: otter          # optional: only count this class, omit to accept any
  confidence_threshold: 0.3
```

The API key comes from `ROBOFLOW_API_KEY` in `.env`, or an explicit `api_key:` in the block.
The Workflow must contain an object-detection block whose output is named by `output_key`.
`confidence_threshold` is sent to the Workflow as its `confidence` input and used as
detstream's sighting cutoff, so the server filters at the same level. Set other Workflow
inputs (`iou_threshold`, `max_detections`) with an optional `parameters:` block.

## Plugins

Built-in components register themselves on import. To add your own detector (an HF model, a
cloud API, a fine-tuned ONNX, etc.), register a factory and declare an entry point:

```python
# mypkg/detector.py
from detstream.detectors import detectors, Detection

class MyDetector:
    def detect(self, frame) -> Detection: ...

@detectors.register("my-model")
def _build(config: dict) -> MyDetector:
    return MyDetector(**config)
```

```toml
# mypkg/pyproject.toml
[project.entry-points."detstream.detectors"]
my-model = "mypkg.detector"
```

After `pip install`, reference it in config as `detector: { type: my-model, ... }`. detstream
discovers it through the entry point with no change to detstream itself. The same pattern works
for `detstream.sources` and `detstream.sinks`.

## Layout

```
detstream/
  registry.py     register + create + entry-point discovery
  config.py       FeedConfig / AppConfig, loads YAML
  runner.py       per-feed asyncio loop
  state.py        SightingTracker: hysteresis + cooldown, no I/O
  events.py       SightingStarted / SightingEnded
  sources/        youtube, stream, file_device (+ shared reconnect base)
  detectors/      yolo_world, roboflow
  sinks/          console, supabase, discord
examples/         otters.yaml, eagles.yaml
tests/            config, registry, sources, state, sinks, detectors
```

## License

MIT. See [LICENSE](LICENSE).