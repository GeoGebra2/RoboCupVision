#这份文件用于训练yolov8的模型
from ultralytics import YOLO

# 加载预训练 YOLOv8 模型（轻量版推荐 yolo8n）
model = YOLO("yolov8n.pt")

# 训练
model.train(
    data="yolov8/data.yaml",  # 数据集配置文件
    imgsz=640,                 # 输入图片大小
    epochs=100,                # 训练轮数
    batch=16,                  # 批大小
    device="cpu"                   # GPU id (如果有GPU)
)
