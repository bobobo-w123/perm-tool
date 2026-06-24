"""
YOLOv8s 训练脚本 — KITTI 车载目标检测
特性：Mosaic/Mixup 数据增强、AdamW 优化器、余弦退火、早停
"""
from ultralytics import YOLO
import datetime
import os

# ============ 配置 ============
PROJECT_ROOT = r"D:\kitti-object-detection"
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "data.yaml")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
RUN_NAME = f"kitti_yolov8s_{TIMESTAMP}"

CONFIG = {
    # === 数据 ===
    "data": DATA_YAML,
    "imgsz": 640,
    "batch": 16,
    "workers": 8,

    # === 训练 ===
    "epochs": 150,
    "patience": 20,
    "save_period": 10,
    "device": 0,
    "verbose": True,

    # === 数据增强 ===
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "flipud": 0.0,

    # === 优化器 ===
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "cos_lr": True,

    # === Loss 权重 ===
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    # === 项目路径 ===
    "project": os.path.join(PROJECT_ROOT, "runs"),
    "name": RUN_NAME,
    "exist_ok": True,
}

# ============ 训练 ============
print(f"{'='*60}")
print(f"  KITTI YOLOv8s 训练")
print(f"  模型: yolov8s.pt (预训练权重)")
print(f"  数据: {DATA_YAML}")
print(f"  轮数: {CONFIG['epochs']}, Batch: {CONFIG['batch']}, ImgSz: {CONFIG['imgsz']}")
print(f"  优化器: {CONFIG['optimizer']}, lr={CONFIG['lr0']}")
print(f"  增强: Mosaic={CONFIG['mosaic']}, Mixup={CONFIG['mixup']}")
print(f"  输出: {CONFIG['project']}/{CONFIG['name']}")
print(f"{'='*60}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    model = YOLO("yolov8s.pt")
    train_kwargs = {k: v for k, v in CONFIG.items()}
    results = model.train(**train_kwargs)

    # ============ 输出结果 ============
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"  最佳模型: {results.save_dir}/weights/best.pt")
    try:
        mAP50 = results.results_dict.get('metrics/mAP50(B)', 'N/A')
        mAP50_95 = results.results_dict.get('metrics/mAP50-95(B)', 'N/A')
        print(f"  mAP@0.5: {mAP50}")
        print(f"  mAP@0.5:0.95: {mAP50_95}")
    except Exception:
        print(f"  训练指标请查看: runs/{RUN_NAME}/results.csv")
    print(f"{'='*60}")
