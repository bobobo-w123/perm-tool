"""
数据集划分：80/20 训练验证集，分层采样保证类别分布一致
"""
import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(r"D:\kitti-object-detection\data\kitti_yolo")
IMAGE_DIR = Path(r"D:\kitti-object-detection\data\raw\training\image_2")
LABEL_DIR = DATA_DIR / "labels"

random.seed(42)

# 收集所有文件名
all_files = sorted([f.stem for f in LABEL_DIR.glob("*.txt")])
print(f"📂 总共 {len(all_files)} 个样本")

# 获取每张图的主导类别（用于分层采样）
def get_dominant_class(label_path):
    classes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.append(int(parts[0]))
    if not classes:
        return -1
    return max(set(classes), key=classes.count)

print("🔄 计算分层标签...")
file_classes = [get_dominant_class(LABEL_DIR / f"{f}.txt") for f in all_files]

# 分层划分 80/20
train_files, val_files = train_test_split(
    all_files, test_size=0.2, random_state=42, stratify=file_classes
)

# 创建目录并复制文件
for split, files in [("train", train_files), ("val", val_files)]:
    img_out = DATA_DIR / "images" / split
    lbl_out = DATA_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for fname in files:
        # 复制图像
        src_img = IMAGE_DIR / f"{fname}.png"
        dst_img = img_out / f"{fname}.png"
        if src_img.exists():
            shutil.copy2(src_img, dst_img)

        # 复制标签
        src_lbl = LABEL_DIR / f"{fname}.txt"
        dst_lbl = lbl_out / f"{fname}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)

# 清理根级 labels 目录（已分到 train/val）
for f in LABEL_DIR.glob("*.txt"):
    f.unlink()

print(f"\n✅ 数据集划分完成！")
print(f"   训练集: {len(train_files)} 张")
print(f"   验证集: {len(val_files)} 张")
print(f"   路径: {DATA_DIR}/images/{{train,val}} 和 {DATA_DIR}/labels/{{train,val}}")
