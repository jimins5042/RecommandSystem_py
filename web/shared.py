import logging
import os

import numpy as np
from PIL import Image
from ultralytics import YOLO
from rembg import remove, new_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR  = os.getenv("MODEL_DIR",  "C:\\Users\\coolc\\PycharmProjects\\recommandSystem-py\\model")
STATIC_DIR = os.getenv("STATIC_DIR", "C:\\Users\\coolc\\PycharmProjects\\recommandSystem-py\\static")

YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))
CLASS_NAMES = ["bag", "sunglasses", "food_drink", "shoes", "clothing"]

YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH)
    logger.info(f"YOLO model loaded from {YOLO_MODEL_PATH}")
else:
    yolo_model = None
    logger.warning(f"YOLO model not found at {YOLO_MODEL_PATH}. Running without object detection.")

# rembg 세션 캐시
_rembg_sessions = {}

def get_rembg_session(model_name: str):
    if model_name not in _rembg_sessions:
        logger.info(f"rembg 새로운 세션 생성: {model_name}")
        _rembg_sessions[model_name] = new_session(model_name)
    return _rembg_sessions[model_name]


def detect_and_crop(
    image: Image.Image,
    use_rembg: bool = False,
    model: str = "u2net",
    alpha_matting: bool = False,
):
    """YOLO로 상품 감지 후 confidence 최고 객체 crop. 전체 감지 목록도 반환."""
    if yolo_model is None:
        return image, None, None, None, []

    processed_image = image
    if use_rembg:
        session = get_rembg_session(model)
        image_rgba = remove(image, session=session, alpha_matting=alpha_matting)
        processed_image = Image.new("RGB", image_rgba.size, (255, 255, 255))
        processed_image.paste(image_rgba, mask=image_rgba.split()[3])

    results = yolo_model.predict(np.array(processed_image), conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return image, None, None, None, []

    w, h = image.size
    all_detections = []
    for box in boxes:
        b = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].item())
        all_detections.append({
            "class": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "unknown",
            "confidence": round(box.conf[0].item(), 4),
            "coordinate": [int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))]
        })

    best_idx  = boxes.conf.argmax().item()
    best_box  = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    best_conf = boxes.conf[best_idx].item()
    best_cls  = int(boxes.cls[best_idx].item())
    class_name = CLASS_NAMES[best_cls] if best_cls < len(CLASS_NAMES) else "unknown"

    x1, y1 = max(0, best_box[0]), max(0, best_box[1])
    x2, y2 = min(w, best_box[2]), min(h, best_box[3])
    cropped = image.crop((x1, y1, x2, y2))

    if cropped.size[0] == 0 or cropped.size[1] == 0:
        return image, None, None, None, all_detections

    return cropped, class_name, best_conf, [int(x1), int(y1), int(x2), int(y2)], all_detections
