#这份文件是用于把dataset分割为yolo的dataset格式
import os
import random
import shutil

# ================== 配置 ==================
src_dir = "PicturesForVision"   # 原始图片文件夹，里面都是 .jpg
output_dir = "dataset"   # 输出数据集目录
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1  # 数据划分比例
# ==========================================

def split_dataset(src_dir, output_dir, ratios=(0.7, 0.2, 0.1)):
    # 获取所有jpg文件
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(".png")]
    random.shuffle(images)

    total = len(images)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    n_test = total - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    # 创建YOLO目录结构
    for split, files in splits.items():
        img_dir = os.path.join(output_dir, "images", split)
        lbl_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for f in files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(img_dir, f)
            shutil.copy(src_path, dst_path)
            # 创建对应的空标签文件（标注时会填充）
            base = os.path.splitext(f)[0]
            open(os.path.join(lbl_dir, base + ".txt"), "w").close()

        print(f"📂 {split}: {len(files)} 张图片")

    # 写 data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""# YOLOv8 dataset config
path: {output_dir}
train: images/train
val: images/val
test: images/test

nc: 3
names: ['blue_bot', 'yellow_bot', 'ball']
""")
    print(f"✅ data.yaml 已生成: {yaml_path}")


if __name__ == "__main__":
    split_dataset(src_dir, output_dir, (train_ratio, val_ratio, test_ratio))
    print("🎉 数据集划分完成，可以开始标注啦！")
