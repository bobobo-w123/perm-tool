"""
KITTI→YOLO 数据集探索性分析
输出：类别分布直方图、bbox 宽高散点图、不平衡比例饼图
"""
import os
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式，服务器也能跑
import matplotlib.pyplot as plt
import seaborn as sns

LABEL_DIR = Path(r"D:\kitti-object-detection\data\kitti_yolo\labels\train")
RESULTS_DIR = Path(r"D:\kitti-object-detection\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist", 3: "truck_van", 4: "tram"}

# 收集统计
class_counts = Counter()
bbox_widths, bbox_heights = [], []
img_bbox_counts = []

for label_file in LABEL_DIR.glob("*.txt"):
    bbox_in_img = 0
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            class_counts[cls_id] += 1
            bbox_widths.append(float(parts[3]))
            bbox_heights.append(float(parts[4]))
            bbox_in_img += 1
    img_bbox_counts.append(bbox_in_img)

total = sum(class_counts.values())

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 类别分布（条状图 + 标注）
names = [CLASS_NAMES.get(k, f"cls_{k}") for k in sorted(class_counts.keys())]
counts = [class_counts[k] for k in sorted(class_counts.keys())]
colors = sns.color_palette("husl", len(names))
bars = axes[0, 0].bar(names, counts, color=colors, edgecolor='white')
axes[0, 0].set_title("Class Distribution", fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel("# of BBoxes")
for bar, v in zip(bars, counts):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                    f"{v}\n({v/total*100:.1f}%)", ha='center', fontsize=9)

# 2. BBox 宽高散点图（还原为 640 像素）
ws = [w * 640 for w in bbox_widths]
hs = [h * 640 for h in bbox_heights]
sample_n = min(3000, len(ws))
idx = np.random.choice(len(ws), sample_n, replace=False)
axes[0, 1].scatter([ws[i] for i in idx], [hs[i] for i in idx], alpha=0.3, s=5, c='steelblue')
axes[0, 1].axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Min size (15px)')
axes[0, 1].axvline(x=15, color='red', linestyle='--', alpha=0.7)
axes[0, 1].set_xlabel("Width (px @640)", fontsize=12)
axes[0, 1].set_ylabel("Height (px @640)", fontsize=12)
axes[0, 1].set_title(f"BBox Size Distribution (sampled {sample_n})", fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 类别不平衡饼图
wedges, texts, autotexts = axes[1, 0].pie(
    counts, labels=names, autopct='%1.1f%%',
    colors=colors, explode=[0.02]*len(names),
    textprops={'fontsize': 11}
)
axes[1, 0].set_title("Class Imbalance Ratio", fontsize=14, fontweight='bold')

# 4. 每图 bbox 数直方图
axes[1, 1].hist(img_bbox_counts, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[1, 1].axvline(x=np.mean(img_bbox_counts), color='red', linestyle='--', label=f'Mean: {np.mean(img_bbox_counts):.1f}')
axes[1, 1].axvline(x=np.median(img_bbox_counts), color='orange', linestyle='--', label=f'Median: {np.median(img_bbox_counts):.0f}')
axes[1, 1].set_xlabel("BBoxes per Image", fontsize=12)
axes[1, 1].set_ylabel("Frequency", fontsize=12)
axes[1, 1].set_title("BBoxes per Image Distribution", fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("KITTI Dataset EDA — YOLO Format", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

save_path = RESULTS_DIR / "eda.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ EDA 图已保存到: {save_path}")
print(f"\n📊 统计摘要:")
print(f"   总 BBox 数: {total}")
print(f"   总图像数: {len(img_bbox_counts)}")
print(f"   平均每图 BBox: {np.mean(img_bbox_counts):.1f}")
print(f"   中位数 BBox: {np.median(img_bbox_counts):.0f}")
print(f"   最大类/最小类比例: {max(counts)/min(counts):.1f}:1")
for cls_id, count in sorted(class_counts.items()):
    name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
    print(f"   {name:12s}: {count:5d}  ({count/total*100:5.1f}%)")
