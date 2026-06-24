"""
推理脚本 — 单张 / 批量 / 摄像头
用法:
    python inference.py                          # 批量推理 val 集前 10 张
    python inference.py --image path/to/img.jpg  # 单张图片
    python inference.py --dir E:\test_imgs       # 整个文件夹
    python inference.py --video test.mp4         # 视频
    python inference.py --webcam                 # 摄像头实时
"""
import argparse
import os
from ultralytics import YOLO

PROJECT_ROOT = r"D:\kitti-object-detection"
BEST_PT = r"D:\kitti-object-detection\runs\kitti_yolov8s_20260624_0822\weights\best.pt"
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "inference")


def run_image(image_path, model, conf=0.25):
    """单张图片推理"""
    results = model(image_path, save=True, conf=conf, project=OUT_DIR, name="images", exist_ok=True)
    print(f"结果保存在: {OUT_DIR}/images/")


def run_dir(dir_path, model, conf=0.25):
    """文件夹批量推理"""
    results = model(dir_path, save=True, conf=conf, project=OUT_DIR, name="batch", exist_ok=True)
    print(f"结果保存在: {OUT_DIR}/batch/")


def run_video(video_path, model, conf=0.25):
    """视频推理"""
    results = model(video_path, save=True, conf=conf, project=OUT_DIR, name="video", exist_ok=True)
    print(f"视频保存在: {OUT_DIR}/video/")


def run_webcam(model, conf=0.25):
    """摄像头实时推理（按 q 退出）"""
    print("启动摄像头... 按 q 退出")
    results = model(source=0, show=True, conf=conf, stream=True)
    for r in results:
        pass  # 实时显示


def main():
    parser = argparse.ArgumentParser(description="YOLOv8s 推理")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--dir", type=str, help="图片文件夹路径")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--webcam", action="store_true", help="摄像头实时推理")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认 0.25)")
    args = parser.parse_args()

    model = YOLO(BEST_PT)
    print(f"模型: {BEST_PT}")

    if args.image:
        run_image(args.image, model, args.conf)
    elif args.dir:
        run_dir(args.dir, model, args.conf)
    elif args.video:
        run_video(args.video, model, args.conf)
    elif args.webcam:
        run_webcam(model, args.conf)
    else:
        # 默认：val 集前 10 张
        val_dir = os.path.join(PROJECT_ROOT, "data", "kitti_yolo", "images", "val")
        if os.path.exists(val_dir):
            imgs = sorted(os.listdir(val_dir))[:10]
            img_paths = [os.path.join(val_dir, f) for f in imgs]
            results = model(img_paths, save=True, conf=args.conf, project=OUT_DIR, name="demo", exist_ok=True)
            print(f"默认推理 10 张 val 图片 → {OUT_DIR}/demo/")
        else:
            print("未找到验证集图片，请指定 --image 或 --dir")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
