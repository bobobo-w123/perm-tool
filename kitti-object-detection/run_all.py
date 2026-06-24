"""
🚀 KITTI 车载目标检测 — 一键全流程
用法:
    python run_all.py              # 全流程
    python run_all.py --skip-train # 跳过训练
    python run_all.py --step 1     # 只跑步骤 1
"""
import subprocess
import sys
import os

SCRIPTS_DIR = r"D:\kitti-object-detection\scripts"

STEPS = {
    1: ("01_convert_kitti2yolo.py", "KITTI → YOLO 格式转换"),
    2: ("02_split_dataset.py", "训练/验证集划分"),
    3: ("03_eda.py", "数据集探索性分析"),
    4: ("04_train.py", "YOLOv8s 训练"),
    5: ("05_evaluate.py", "模型评估"),
    6: ("06_prune.py", "模型剪枝 + ONNX 导出"),
    7: ("07_export_trt.py", "TensorRT 引擎导出"),
    8: ("08_export_openvino.py", "OpenVINO 模型导出"),
    9: ("09_benchmark.py", "全模型性能对比"),
}

def run_step(step_num):
    script, desc = STEPS[step_num]
    script_path = os.path.join(SCRIPTS_DIR, script)
    if not os.path.exists(script_path):
        print(f"❌ 脚本不存在: {script_path}")
        return False

    print(f"\n{'#'*60}")
    print(f"  📌 Step {step_num}: {desc}")
    print(f"  脚本: {script}")
    print(f"{'#'*60}")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=r"D:\kitti-object-detection"
    )
    return result.returncode == 0


def main():
    args = sys.argv[1:]

    skip_train = "--skip-train" in args
    step_only = None
    for a in args:
        if a.startswith("--step"):
            step_only = int(a.split()[-1]) if " " in a else int(args[args.index(a)+1]) if args.index(a)+1 < len(args) else None

    if step_only:
        run_step(step_only)
        return

    print("=" * 60)
    print("   🚗 KITTI 车载目标检测 — 全流程启动")
    print("=" * 60)
    print("\n  ⚠ 请确保：")
    print("    1. KITTI 数据已放到 data/raw/training/image_2/ 和 label_2/")
    print("    2. 已安装依赖: pip install -r requirements.txt")
    print("    3. GPU 可用且 CUDA 已配置")

    for step_num in sorted(STEPS.keys()):
        if skip_train and step_num >= 4:
            print(f"\n  ⏭ 跳过 Step {step_num} ({STEPS[step_num][1]})")
            continue
        if not run_step(step_num):
            print(f"\n❌ Step {step_num} 失败，后续步骤终止")
            print(f"   请修复问题后重试: python run_all.py --step {step_num}")
            break

    print(f"\n{'='*60}")
    print(f"   🎉 全流程完成！")
    print(f"   结果查看:")
    print(f"     训练权重: runs/kitti_yolov8s_xxx/weights/best.pt")
    print(f"     加速模型: exports/")
    print(f"     对比报告: results/benchmark_results.csv")
    print(f"     分析图表: results/eda.png, results/benchmark_comparison.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
