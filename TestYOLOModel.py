#这份文件用于测试yolov8模型的检测结果
import cv2
from ultralytics import YOLO

# 1. 加载你训练好的模型
model = YOLO("runs/detect/train/weights/best.pt")  # 修改为你模型的路径

# 2. 要测试的图片路径列表（这里举例6张）
image_files = [
    "dataset/images/test/00013.png",
    "dataset/images/test/00502.png",
    "dataset/images/test/00038.png",
    "dataset/images/test/00189.png",
    "dataset/images/test/00078.png",
    "dataset/images/test/00116.png",
]

# 3. 遍历每张图片进行推理
cnt = 0
for img_path in image_files:
    cnt += 1
    results = model(img_path)  # 运行检测
    # results 是一个列表，这里取第一项
    result = results[0]

    # 4. 将检测结果画在图上
    annotated_frame = result.plot()

    # 5. 显示结果
    #cv2.imshow("YOLOv8 Detection", annotated_frame)
    print(f"检测结果 - {img_path}:")
    result.save(filename=f"ModelTestResult/output_{cnt}.jpg")
    #result.show()  # 控制台打印信息
    #cv2.waitKey(0)  # 按任意键显示下一张

cv2.destroyAllWindows()
