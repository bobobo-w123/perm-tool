# KITTI 车载目标检测 — YOLOv8s

基于 YOLOv8s 的 KITTI 数据集车载目标检测，5 类目标（Car / Pedestrian / Cyclist / Truck_Van / Tram），mAP@0.5 达 94.9%。

## 项目结构

```
kitti-object-detection/
├── scripts/                 # 核心脚本
│   ├── 01_convert_kitti2yolo.py   # KITTI → YOLO 格式转换
│   ├── 02_split_dataset.py        # 训练/验证集划分
│   ├── 03_eda.py                  # 数据集探索性分析
│   ├── 04_train.py                # 训练 (YOLOv8s, 150 epochs)
│   ├── 05_evaluate.py             # 模型评估 (mAP + 各类别指标)
│   ├── 06_export_onnx.py          # ONNX 导出 (为 TensorRT 部署留口)
│   └── inference.py               # 推理 (图片/视频/摄像头)
├── data/                    # 数据集
│   ├── raw/                      # 原始 KITTI 数据
│   ├── kitti_yolo/               # YOLO 格式标注
│   └── data.yaml                 # 数据集配置
├── runs/                    # 训练产物 (weights / 曲线 / 日志)
├── results/                 # 评估结果 & 推理输出
└── requirements.txt
```

## 快速开始

### 环境

```bash
conda create -n kitti-object-detection python=3.12
conda activate kitti-object-detection
pip install -r requirements.txt
```

### 数据准备

1. 下载 [KITTI 2D Object Detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) 数据集
2. 解压 `data_object_image_2.zip` 和 `data_object_label_2.zip`
3. 放入 `data/raw/training/image_2/` 和 `data/raw/training/label_2/`

### 一键运行

```bash
# 格式转换 + 划分 + 训练 (首次运行)
python scripts/01_convert_kitti2yolo.py
python scripts/02_split_dataset.py
python scripts/04_train.py

# 推理
python scripts/inference.py --image path/to/img.jpg
python scripts/inference.py --dir path/to/folder
python scripts/inference.py --video road.mp4
```

## 实验结果

| 指标 | 数值 |
|------|------|
| mAP@0.5 | **94.9%** |
| mAP@0.5:0.95 | **74.0%** |
| 模型大小 | 22.5 MB |
| 推理速度 | ~1.0 ms/image (RTX 4060) |
| 最佳 epoch | 123 / 150 |

### 各类别精度

| 类别 | AP@0.5 | AP@0.5:0.95 |
|------|--------|-------------|
| 🚗 Car | 97.3% | 82.7% |
| 🚶 Pedestrian | 88.5% | 53.6% |
| 🚴 Cyclist | 92.4% | 69.1% |
| 🚛 Truck / Van | 97.5% | 82.5% |
| 🚋 Tram | 98.6% | 82.0% |

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | YOLOv8s (预训练) |
| 输入尺寸 | 640×640 |
| Batch Size | 16 |
| 优化器 | AdamW (lr=0.001, weight_decay=5e-4) |
| 学习率策略 | 余弦退火 (warmup 3 epochs) |
| 数据增强 | Mosaic 1.0, MixUp 0.15, Copy-Paste 0.1 |
| 早停 | patience=20 |
| GPU | NVIDIA RTX 4060 Laptop 8GB |

## 后续扩展

- [ ] TensorRT 引擎导出 & INT8 量化
- [ ] NMS 优化 (TensorRT EfficientNMS)
- [ ] 多尺度训练 (imgsz 320-960)
- [ ] YOLOv8m/l 更大模型对比
