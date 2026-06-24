"""
KITTI Object Detection → YOLOv8 格式转换
处理边界情况：DontCare 区域、截断目标、极小目标过滤
"""
import os
import cv2
from pathlib import Path
from tqdm import tqdm

# ============ 配置 ============
KITTI_ROOT = Path(r"D:\kitti-object-detection\data\raw\training")
IMAGE_DIR = KITTI_ROOT / "image_2"
LABEL_DIR = KITTI_ROOT / "label_2"
OUTPUT_DIR = Path(r"D:\kitti-object-detection\data\kitti_yolo")

# 类别映射：KITTI class → YOLO class_id（-1 = 丢弃）
CLASS_MAP = {
    "Car": 0,
    "Van": 3,
    "Truck": 3,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
    "Tram": 4,
    "Misc": -1,
    "DontCare": -1,
}

CLASS_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist", 3: "truck_van", 4: "tram"}

# 极小目标过滤阈值（像素）
MIN_BBOX_WIDTH = 15
MIN_BBOX_HEIGHT = 15
# 截断目标过滤：truncated > 此阈值丢弃
MAX_TRUNCATION = 0.9


def convert_single(label_path, img_w, img_h):
    """转换单个 KITTI 标签文件为 YOLO 格式行列表"""
    yolo_lines = []
    stats = {"dontcare": 0, "truncated": 0, "small": 0, "kept": 0}

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            cls_name = parts[0]
            truncated = float(parts[1])

            # 丢弃 Misc 和 DontCare
            yolo_cls = CLASS_MAP.get(cls_name, -1)
            if yolo_cls == -1:
                stats["dontcare"] += 1
                continue

            # 过滤过度截断的目标
            if truncated > MAX_TRUNCATION:
                stats["truncated"] += 1
                continue

            # 2D bbox（pixel 坐标）
            left = float(parts[4])
            top = float(parts[5])
            right = float(parts[6])
            bottom = float(parts[7])

            w = right - left
            h = bottom - top

            # 过滤极小目标
            if w < MIN_BBOX_WIDTH or h < MIN_BBOX_HEIGHT:
                stats["small"] += 1
                continue

            # 转为 YOLO 归一化格式
            x_center = ((left + right) / 2) / img_w
            y_center = ((top + bottom) / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # 防御性裁剪
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            yolo_lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            stats["kept"] += 1

    return yolo_lines, stats


def main():
    # 创建输出目录
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    label_files = sorted(LABEL_DIR.glob("*.txt"))
    total_stats = {"dontcare": 0, "truncated": 0, "small": 0, "kept": 0}
    per_class = {}

    print(f"📂 找到 {len(label_files)} 个 KITTI 标签文件")
    print(f"🔄 开始转换 KITTI → YOLO ...")

    for label_path in tqdm(label_files, desc="Converting"):
        img_name = label_path.stem + ".png"
        img_path = IMAGE_DIR / img_name

        if not img_path.exists():
            print(f"  ⚠ Image not found: {img_name}, skipping")
            continue

        # 读取图像尺寸
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # 转换
        yolo_lines, stats = convert_single(label_path, w, h)

        for k in total_stats:
            total_stats[k] += stats[k]

        # 统计每类 bbox 数
        for line in yolo_lines:
            cls_id = int(line.split()[0])
            per_class[cls_id] = per_class.get(cls_id, 0) + 1

        # 保存 YOLO 标签
        out_label_path = OUTPUT_DIR / "labels" / (label_path.stem + ".txt")
        with open(out_label_path, 'w') as f:
            f.write("\n".join(yolo_lines))

    # 保存类别映射
    with open(OUTPUT_DIR / "classes.txt", 'w') as f:
        for k in sorted(CLASS_NAMES.keys()):
            f.write(f"{CLASS_NAMES[k]}\n")

    # 输出报告
    print(f"\n{'='*60}")
    print(f"  ✅ 转换完成！")
    print(f"  总 bbox 保留: {total_stats['kept']}")
    print(f"  过滤 DontCare/Misc: {total_stats['dontcare']}")
    print(f"  过滤过度截断: {total_stats['truncated']}")
    print(f"  过滤极小目标: {total_stats['small']}")
    print(f"\n  类别分布:")
    for cls_id, count in sorted(per_class.items()):
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        pct = count / total_stats['kept'] * 100
        print(f"    {name:12s}: {count:5d}  ({pct:5.1f}%)")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
