# Compatible with:
# label-studio == 1.21.0
# label-studio-ml == 2.0.1.dev0
# ultralytics == 8.3.221

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64
import requests
import os


def load_image(src, context=None):
    # 1) base64
    if src.startswith("data:"):
        b64 = src.split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

    # 2) URL
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")

    # 3) Label Studio local path: /data/upload/...
    if src.startswith("/data/"):

        # ✅ CASE A: context exists
        if context is not None:
            project_path = context.get("project_path")
            if project_path:
                base = os.path.abspath(os.path.join(project_path, ".."))
                real_path = os.path.join(base, src.lstrip("/"))
                return Image.open(real_path).convert("RGB")

        # ✅ CASE B: no context → use LABEL_STUDIO_BASE_DIR
        base = "/Users/wangxp/Library/Application Support/label-studio"
        if base:
            real_path = os.path.join(base, src.lstrip("/"))
            return Image.open(real_path).convert("RGB")

        # ✅ CASE C: fallback (rare)
        local_path = "." + src
        if os.path.exists(local_path):
            return Image.open(local_path).convert("RGB")

        raise FileNotFoundError(f"Cannot resolve {src}. Set LABEL_STUDIO_BASE_DIR or enable context.")

    # 4) local file path
    return Image.open(src).convert("RGB")


class NewModel(LabelStudioMLBase):
    """Minimal YOLO detection backend for LSML 2.x"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Parse LS label config
        cfg = self.parsed_label_config
        self.from_name = list(cfg.keys())[0]
        self.to_name = cfg[self.from_name]["to_name"][0]

        # Load your YOLO model
        # Change to your best.pt
        self.model = YOLO("/Users/wangxp/Jupyter/my_yolo/showcase_200_best.pt")

    def predict(self, tasks, context=None, **kwargs):
        print(tasks)
        responses = []
        path = self.get_local_path(tasks[0]['data']['image'], task_id=tasks[0]['id'])
        task = tasks[0]
        print("-------------------------")
        print(path)
        img = load_image(path, context)
        w, h = img.size

        pred = self.model(img, verbose=False)[0]
        result = []

        for b in pred.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = int(b.cls[0])
            name = self.model.names[cls]

            result.append({
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": {
                    "x": x1 / w * 100,
                    "y": y1 / h * 100,
                    "width": (x2 - x1) / w * 100,
                    "height": (y2 - y1) / h * 100,
                    "rectanglelabels": [name]
                },
                "score": float(b.conf[0])
            })
    
        responses.append({"result": result})
    
        return responses