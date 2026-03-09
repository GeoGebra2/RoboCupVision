import sys
import json
import base64
import numpy as np
import cv2
from ultralytics import YOLO
model = None
model_path = None
def ensure_model(path):
    global model, model_path
    if model is None or model_path != path:
        model = YOLO(path)
        model_path = path
def decode_image(req):
    w = int(req.get("width", 0))
    h = int(req.get("height", 0))
    is_jpeg = bool(req.get("jpeg", True))
    b64 = req.get("image_b64", "")
    if not b64:
        return None
    data = base64.b64decode(b64)
    if is_jpeg:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    else:
        arr = np.frombuffer(data, dtype=np.uint8)
        if w <= 0 or h <= 0:
            return None
        img = arr.reshape((h, w, 3))
        return img
def run_once(req):
    try:
        path = req.get("model", "")
        if not path:
            return []
        ensure_model(path)
        img = decode_image(req)
        if img is None:
            sys.stderr.write(f"diag: model={path} img=None\n")
            sys.stderr.flush()
            return []
        conf = float(req.get("conf", 0.25))
        iou = float(req.get("iou", 0.5))
        try:
            res = model(img, conf=conf, iou=iou)
        except Exception as e:
            sys.stderr.write(f"diag: model={path} img={img.shape[1]}x{img.shape[0]} conf={conf} iou={iou} error={str(e)}\n")
            sys.stderr.flush()
            return []
        r = res[0]
        out = []
        if hasattr(r, "boxes") and r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            if xyxy.shape[0] == 0:
                sys.stderr.write(f"diag: model={path} img={img.shape[1]}x{img.shape[0]} conf={conf} iou={iou} detections=0\n")
                sys.stderr.flush()
                return []
            for i in range(xyxy.shape[0]):
                x1 = int(xyxy[i,0])
                y1 = int(xyxy[i,1])
                x2 = int(xyxy[i,2])
                y2 = int(xyxy[i,3])
                c = float(confs[i])
                cls = int(clss[i])
                out.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":c,"class_id":cls})
        return out
    except Exception as e:
        sys.stderr.write(f"diag: error={str(e)}\n")
        sys.stderr.flush()
        return []
def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            print("[]")
            sys.stdout.flush()
            continue
        try:
            req = json.loads(line)
        except Exception:
            print("[]")
            sys.stdout.flush()
            continue
        out = run_once(req)
        print(json.dumps(out))
        sys.stdout.flush()
if __name__ == "__main__":
    main()
