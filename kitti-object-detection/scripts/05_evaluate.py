"""
模型评估 — 验证集 mAP + 各类别详细指标
"""
from ultralytics import YOLO
import json
import os

PROJECT_ROOT = r"D:\kitti-object-detection"
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "data.yaml")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 自动找 best.pt
BEST_PT = None
runs_dir = os.path.join(PROJECT_ROOT, "runs")
for root, _, files in os.walk(runs_dir):
    if "best.pt" in files:
        candidate = os.path.join(root, "best.pt")
        if BEST_PT is None or os.path.getmtime(candidate) > os.path.getmtime(BEST_PT):
            BEST_PT = candidate

if not BEST_PT:
    print("❌ 未找到 best.pt")
    exit(1)

print(f"评估模型: {BEST_PT}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    model = YOLO(BEST_PT)
    metrics = model.val(data=DATA_YAML, split="val", verbose=True)

    results = {
        "model": BEST_PT,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "per_class": {}
    }
    for name, ap50, ap in zip(metrics.names.values(), metrics.box.ap50, metrics.box.ap):
        results["per_class"][name] = {
            "AP50": round(float(ap50), 4),
            "AP50_95": round(float(ap), 4)
        }

    print(f"\n{'='*50}")
    print(f"  mAP@0.5:      {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {results['mAP50_95']:.4f}")
    print(f"{'='*50}")

    json_path = os.path.join(RESULTS_DIR, "eval_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"已保存: {json_path}")
