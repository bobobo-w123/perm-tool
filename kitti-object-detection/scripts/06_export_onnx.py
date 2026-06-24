"""
模型导出 — ONNX 格式（为 TensorRT / OpenVINO 部署做准备）
"""
from ultralytics import YOLO
import os
import shutil

PROJECT_ROOT = r"D:\kitti-object-detection"
EXPORT_DIR = os.path.join(PROJECT_ROOT, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

# 自动找 best.pt
BEST_PT = None
runs_dir = os.path.join(PROJECT_ROOT, "runs")
for root, _, files in os.walk(runs_dir):
    if "best.pt" in files:
        candidate = os.path.join(root, "best.pt")
        if BEST_PT is None or os.path.getmtime(candidate) > os.path.getmtime(BEST_PT):
            BEST_PT = candidate

if not BEST_PT:
    print("未找到 best.pt")
    exit(1)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    print(f"导出模型: {BEST_PT}")
    model = YOLO(BEST_PT)

    # FP32 ONNX
    print("\n导出 ONNX (FP32)...")
    model.export(format="onnx", simplify=True, opset=12, half=False)
    default_onnx = os.path.join(os.path.dirname(BEST_PT), "best.onnx")
    onnx_fp32 = os.path.join(EXPORT_DIR, "yolov8s_fp32.onnx")
    if os.path.exists(default_onnx):
        shutil.move(default_onnx, onnx_fp32)
        print(f"{onnx_fp32} ({os.path.getsize(onnx_fp32)/(1024*1024):.1f} MB)")

    # FP16 ONNX
    print("\n导出 ONNX (FP16)...")
    model.export(format="onnx", simplify=True, opset=12, half=True)
    onnx_fp16 = os.path.join(EXPORT_DIR, "yolov8s_fp16.onnx")
    if os.path.exists(default_onnx):
        shutil.move(default_onnx, onnx_fp16)
        print(f"{onnx_fp16} ({os.path.getsize(onnx_fp16)/(1024*1024):.1f} MB)")

    print(f"\n{'='*50}")
    print(f" 导出完成！可用于:")
    print(f" TensorRT 引擎转换 (scripts/07_export_trt.py)")
    print(f" penVINO 模型转换 (scripts/08_export_openvino.py)")
    print(f" {EXPORT_DIR}")
