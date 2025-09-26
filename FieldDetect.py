# 这份代码是用于生成场地的标定文件homography
import cv2
import numpy as np

# ========== 参数 ==========
img_path = "dataset/images/test/00013.png"  # 你的场地图像
scale = 0.5  # 显示缩放比例（比如原图太大就改成0.3 或 0.5）

# ========== 加载图像并缩放显示 ==========
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"找不到图片 {img_path}")

h, w = img.shape[:2]
display_img = cv2.resize(img, (int(w*scale), int(h*scale)))

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # 把缩放后的点击点映射回原图坐标
        orig_x, orig_y = int(x/scale), int(y/scale)
        clicked_points.append([orig_x, orig_y])
        print(f"点击点 (缩放): ({x},{y}) -> (原图): ({orig_x},{orig_y})")
        cv2.circle(display_img, (x, y), 5, (0,0,255), -1)
        cv2.imshow("Click Field Corners", display_img)

cv2.imshow("Click Field Corners", display_img)
cv2.setMouseCallback("Click Field Corners", mouse_callback)

print("请依次点击场地的 4 个角点 (左上 -> 右上 -> 右下 -> 左下)")

cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked_points) != 4:
    raise ValueError("需要 4 个角点，实际只选了 {}".format(len(clicked_points)))

# ========== 世界坐标系定义 ==========
world_points = np.array([
    [-450, -300],  # 左上
    [450, -300],   # 右上
    [450, 300],    # 右下
    [-450, 300]    # 左下
], dtype=np.float32)

# ========== 计算 Homography ==========
image_points = np.array(clicked_points, dtype=np.float32)
H, _ = cv2.findHomography(image_points, world_points)

np.save("homography.npy", H)
print("✅ Homography 已保存到 homography.npy")
