#这个文件用于测试车ID的检测情况
import cv2
import numpy as np
from ultralytics import YOLO
import math
import os

# ========== 定义颜色阈值 ==========
COLOR_RANGES = {
    "pink": ((140, 50, 50), (170, 255, 255)),  # 粉色
    "green": ((40, 50, 50), (80, 255, 255)),   # 绿色
}

# ========== ID 映射表（顺序为：上→右→下→左） ==========
ID_MAP = {
    ("pink", "green", "pink", "pink"): 0,
    ("green", "green", "pink", "pink"): 1,
    ("green", "green", "pink", "green"): 2,
    ("pink", "green", "pink", "green"): 3,
    ("pink", "pink", "green", "pink"): 4,
    ("green", "pink", "green", "pink"): 5,
    ("green", "pink", "green", "green"): 6,
    ("pink", "pink", "green", "green"): 7,
    ("green", "green", "green", "green"): 8,
    ("pink", "pink", "pink", "pink"): 9,
    ("pink", "green", "green", "pink"): 10,
    ("green", "pink", "pink", "green"): 11,
    ("green", "green", "green", "pink"): 12,
    ("green", "pink", "pink", "pink"): 13,
    ("pink", "green", "green", "green"): 14,
    ("pink", "pink", "pink", "green"): 15
}

def detect_robot_id_and_orientation(robot_crop):
    """ 根据机器人顶上的色块排列识别 朝向 和 ID """
    hsv = cv2.cvtColor(robot_crop, cv2.COLOR_BGR2HSV)

    centers = []
    for color, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 50:  # 过滤小噪声
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            centers.append((cx, cy, color))

    if len(centers) != 4:
        return None, None  # 没找到4个色块，可能识别失败

    # 计算机器人中心（裁剪图中心）
    h, w = robot_crop.shape[:2]
    cx, cy = w//2, h//2

    # 给每个色块计算极角
    angles = []
    for (x, y, color) in centers:
        angle = math.degrees(math.atan2(-(y - cy), x - cx))  # 以正上为0°
        angle = (angle + 360) % 360
        angles.append((angle, color))

    #print(angles)

    # 按角度排序
    angles.sort(key=lambda t: t[0])
    print(angles)

    # 找出最大角度间隔，作为前方
    max_gap, max_idx = -1, -1
    n = len(angles)
    for i in range(n):
        a1 = angles[i][0]
        a2 = angles[(i+1) % n][0]
        gap = (a2 - a1 + 360) % 360
        if gap > max_gap:
            max_gap, max_idx = gap, i

    # 朝向角度 = 这两个点中间角度
    print(max_idx)

    front_angle1 = angles[max_idx][0]
    orientation = (front_angle1 + max_gap/2) % 360

    # 旋转归一化
    norm_seq = []
    for angle, c in angles:
        norm_angle = (angle - orientation + 360) % 360
        norm_seq.append((norm_angle, c))

    # 按顺时针排序，上(0°) → 右(90°) → 下(180°) → 左(270°)
    norm_seq.sort(key=lambda t: t[0])
    color_sequence = tuple([c for _, c in norm_seq])
    print(norm_seq)
    print(orientation)
    #print(color_sequence)

    # 查 ID
    robot_id = ID_MAP.get(color_sequence, None)

    return orientation, robot_id


# ========== 主程序 ==========
if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")

    img_path = "dataset/images/test/00013.png"  # 输入测试图
    results = model(img_path)

    img = cv2.imread(img_path)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0].item())
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            orientation, robot_id = None, None
            if label in ["blue_bot", "yellow_bot"]:
                crop = img[y1:y2, x1:x2]
                orientation, robot_id = detect_robot_id_and_orientation(crop)

            # 绘制检测框
            color = (0, 255, 0) if label == "blue_bot" else (0, 255, 255) if label == "yellow_bot" else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 显示文本
            if robot_id is not None:
                text = f"{label} ID={robot_id} angle={orientation:.1f}°"
            else:
                text = label

            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    os.makedirs("RoboIDTestResult", exist_ok=True)
    cv2.imwrite("RoboIDTestResult/detection_result.jpg", img)
    cv2.destroyAllWindows()
