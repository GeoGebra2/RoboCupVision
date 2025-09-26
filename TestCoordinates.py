#这个脚本用于测试从像素到坐标的转换关系
import cv2
import numpy as np
from ultralytics import YOLO

# ================== 载入 Homography ==================
try:
    H = np.load("homography.npy")
    print("✅ 已加载 homography.npy")
except:
    raise FileNotFoundError("❌ 请先运行角点选择脚本生成 homography.npy")

# ================== 坐标转换函数 ==================
def image_to_world(u, v, H):
    pt = np.array([[u, v]], dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pt, H)
    return dst[0][0]  # (X, Y)

# ================== YOLO 检测模型 ==================
model = YOLO("runs/detect/train/weights/best.pt")  # 换成你自己的模型路径

# ================== 运行检测 ==================
img_path = "dataset/images/test/00013.png"  # 换成你的测试图
img = cv2.imread(img_path)
results = model(img_path)

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0].item())
        label = model.names[cls]

        # 取检测框中心点 (像素坐标)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # 转换到世界坐标
        X, Y = image_to_world(u, v, H)

        # 绘制检测框和坐标
        color = (0, 255, 0) if label == "blue_bot" else (0, 255, 255) if label == "yellow_bot" else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} ({X:.1f}, {Y:.1f})"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        print(f"{label} 像素=({u},{v}) -> 世界=({X:.1f},{Y:.1f})")

# ================== 保存结果 ==================
cv2.imwrite("RoboIDTestResult/world_coords_result.jpg", img)
print("✅ 结果已保存到 RoboIDTestResult/world_coords_result.jpg")
