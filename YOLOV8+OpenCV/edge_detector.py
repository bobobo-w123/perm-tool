# -*- coding: utf-8 -*-
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple
import torch

# ==================== 数据类：定义检测结果的数据结构 ====================
@dataclass
class DetectionResult:
    """
    dataclass 是 Python 内置的，自动生成 __init__、__repr__ 等方法
    这里用来封装一帧图像的所有结果，方便后面传给 VLM
    """
    frame_idx: int  # 第几帧
    timestamp: float  # 时间戳（秒）
    image_path: str  # 原始图路径
    annotated_path: str  # 标注图路径
    detections: List[dict]  # 检测到的目标列表
    should_trigger_vlm: bool  # 是否触发云端 VLM
    trigger_reason: str  # 触发原因描述


class EdgeDetector:
    """
    边缘端检测器。
    职责：加载 YOLO、抽帧、检测、保存图片、判断是否需要云端推理。
    为什么叫 EdgeDetector？因为所有操作都在本地/边缘端完成，不上云。
    """

    def __init__(self, config: "Config"):
        self.config = config
        print(f"[Edge] 加载 YOLO 模型: {config.YOLO_MODEL}")

        # YOLOv8 由 Ultralytics 提供，一行代码就能加载
        self.model = YOLO(config.YOLO_MODEL)

        # 模型放到 GPU 上，加速推理
        # 如果 device="cpu" 也能跑，但慢很多
        device = "cpu" if config.DEVICE == "cpu" or not torch.cuda.is_available() else f"cuda:{config.DEVICE}"
        self.model.to(device)
        # 获取类别名称，比如 {0: 'person', 1: 'bicycle',
        # ...接上
        self.class_names = self.model.names
        print("[Edge] 模型加载完成")

    # ==================== 主流程：视频抽帧 + 逐帧处理 ====================
    def extract_frames(self, video_path: str, output_dir: str) -> List[DetectionResult]:
        """
        输入：视频路径
        输出：每一帧的 DetectionResult 列表
        流程：打开视频 → 按间隔抽帧 → 对每帧调用 YOLO → 保存结果
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # mkdir(parents=True) 表示如果父目录不存在也一起创建
        # exist_ok=True 表示目录已存在时不报错

        # OpenCV 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频元信息
        fps = cap.get(cv2.CAP_PROP_FPS)  # 原始帧率，比如 30fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
        interval = max(1, int(round(fps / self.config.FPS)))
        # 抽帧间隔计算：如果原视频 30fps，想抽 1fps，间隔就是 30
        # max(1, ...) 防止 interval 为 0

        print(f"[Edge] 视频 FPS: {fps}, 总帧数: {total_frames}, 抽帧间隔: {interval}")

        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            # cap.read() 返回两个值：是否成功、当前帧图像（numpy 数组）
            if not ret:
                break  # 视频读完了

            # 按 interval 抽帧
            if frame_idx % interval == 0:
                timestamp = frame_idx / fps  # 当前帧对应的真实时间

                result = self._process_frame(
                    frame, frame_idx, timestamp,
                    output_dir, video_path.stem
                )
                results.append(result)

                if len(results) % 10 == 0:
                    print(f"[Edge] 已处理 {len(results)} 帧...")

            frame_idx += 1

        cap.release()  # 释放视频资源
        print(f"[Edge] 抽帧完成，共 {len(results)} 帧")
        return results

    # ==================== 单帧处理：检测、保存、触发判断 ====================
    def _process_frame(self, frame, frame_idx, timestamp,
                       output_dir: Path, video_name: str) -> DetectionResult:
        """
        对单帧图像执行：
        1. 保存原图
        2. YOLO 目标检测
        3. 在图上画框
        4. 保存标注图
        5. 判断是否需要触发 VLM
        """
        h, w = frame.shape[:2]  # 图像高和宽，后面画框和计算相对位置要用

        # 保存原始帧（受 SAVE_RAW_FRAMES 控制，默认关闭以节省磁盘）
        raw_path = output_dir / f"{video_name}_frame_{frame_idx:06d}_raw.jpg"
        if self.config.SAVE_RAW_FRAMES:
            cv2.imwrite(str(raw_path), frame)

        # ==================== YOLO 推理 ====================
        # YOLOv8 的接口非常简洁，传入 numpy 数组即可
        # verbose=False 关闭 YOLO 自己的打印，避免输出太乱
        yolo_results = self.model(frame, conf=0.35, iou=0.5, verbose=False)
        # conf 从 0.15 提到 0.35：过滤低置信度的误检，如地面标线、非目标纹理

        # 解析检测结果
        detections = []
        annotated = frame.copy()  # 复制一份原图用于画框

        for box in yolo_results[0].boxes:
            cls_id = int(box.cls[0])  # 类别编号
            cls_name = self.class_names[cls_id]  # 类别名称
            conf = float(box.conf[0])  # 置信度
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框左上角和右下角坐标

            # 计算归一化中心点和相对大小
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h

            # 宽高比修正：YOLO 没有"电动车"类，经常把骑行者框成 person
            # 车辆扁长（>1.5），行人瘦高（<0.8），电动车/摩托接近正方形（0.8~1.5）
            if cls_name == 'person' and conf > 0.3:
                aspect_ratio = (x2 - x1) / max(y2 - y1, 1)
                if aspect_ratio > 1.5:
                    cls_name = 'car'
                elif 0.8 <= aspect_ratio <= 1.5:
                    cls_name = 'motorcycle'  # 电动车/摩托通常接近正方形框

            # 地面标线过滤：交通标志只会出现在画面上半部分（立柱上的牌子），
            # 画面下半部分的 "stop sign" / "traffic light" 大概率是地面标线
            if cls_name in ['traffic light', 'stop sign'] and cy > 0.6:
                continue  # 跳过地面误检，不画框也不加入检测列表

            # 面积与位置联合过滤：画面边缘 + 小面积 + 低置信度的车辆，多是路标/杂物
            # 真正需要关注的车都在画面中央偏下（本车前方），边缘的忽略
            area = bw * bh
            # 调试：打印所有 car 框的详细数据，方便调参过滤
            if cls_name == 'car':
                print(f"[CAR-DEBUG] frame={frame_idx} conf={conf:.2f} bw={bw:.3f} bh={bh:.3f} cx={cx:.2f} cy={cy:.2f} area={area:.3f}")
            in_left = cx < 0.2
            in_right = cx > 0.8
            at_bottom = cy > 0.85
            if cls_name in ['car', 'truck', 'bus']:
                # 扁宽 + 贴底 = 地面标线/路标，不是真车
                is_squat = bh < 0.05  # 高度不到画面5%，极度扁
                at_very_bottom = cy > 0.80  # 中心在画面下20%
                if is_squat and at_very_bottom:
                    continue
                # 画面边缘低置信度车辆
                if (cx < 0.25 or cx > 0.75) and conf < 0.55:
                    continue
                if at_bottom and conf < 0.5:
                    continue
                if area < 0.008 and conf < 0.6:
                    continue

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "center": (round(cx, 3), round(cy, 3)),
                "size": (round(bw, 3), round(bh, 3))
            })

            # 根据类别选颜色画框
            color = self._get_color(cls_name)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # rectangle: 图像、左上角、右下角、颜色、线宽

            # 画标签
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                annotated, label,
                (max(x1, 5), max(y1 - 10, 20)),  # 防止文字超出边界
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # 保存标注后的图，传给 VLM 时用它，因为框能提示模型关注什么
        ann_path = output_dir / f"{video_name}_frame_{frame_idx:06d}_ann.jpg"
        cv2.imwrite(str(ann_path), annotated)

        # 判断是否需要云端推理
        should_trigger, reason = self._check_trigger(detections)

        return DetectionResult(
            frame_idx=frame_idx,
            timestamp=round(timestamp, 2),
            image_path=str(raw_path),
            annotated_path=str(ann_path),
            detections=detections,
            should_trigger_vlm=should_trigger,
            trigger_reason=reason
        )

    # ==================== 触发判断：核心成本优化逻辑 ====================
    def _check_trigger(self, detections: List[dict]) -> Tuple[bool, str]:
        """
        这是整个项目最重要的工程决策之一：
        不是每帧都调 VLM，而是只选"值得分析"的关键帧。

        触发规则设计思路：
        1. 近距离大目标 → 马上可能撞，必须分析
        2. 正前方近处行人/非机动车 → 高危场景
        3. 中心区域交通灯/停车标志 → 需要理解语义
        4. 多目标复杂场景 → 简单 YOLO 判断不了关系

        返回：(是否触发, 触发原因)
        """
        reasons = []

        for det in detections:
            cls_name = det["class"]
            conf = det["confidence"]
            cx, cy = det["center"]
            bw, bh = det["size"]

            # 先按类别过滤置信度
            threshold = self.config.TRIGGER_CLASSES.get(cls_name, 0.95)
            if conf < threshold:
                continue  # 置信度不够，跳过

            area = bw * bh  # 相对面积，近似代表"距离近+尺寸大"

            # 规则1：占画面大 = 距离很近，必须上云分析
            if area > 0.08:
                reasons.append(f"近距离{cls_name}(占比{area:.1%})")
                continue

            # 规则2：正前方近处的行人/非机动车
            # 中心区域 + 画面下方，对应真实世界中的"正前方近距离"
            in_center = self.config.CENTER_REGION[0] < cx < self.config.CENTER_REGION[1]
            in_near = self.config.NEAR_REGION[0] < cy < self.config.NEAR_REGION[1]

            if cls_name in ['person', 'bicycle', 'motorcycle'] and in_center and in_near:
                reasons.append(f"正前方近处有{cls_name}")
                continue

            # 规则3：中心区域有交通灯/停车标志
            # 这类目标需要 VLM 理解颜色、状态，YOLO 只能检测到"有个灯"
            if cls_name in ['traffic light', 'stop sign'] and in_center:
                reasons.append(f"检测到{cls_name}")
                continue

        # 规则4：多目标复杂场景
        # 即使单个目标都不危险，多个目标同时出现也需要理解相互关系
        high_conf = [d for d in detections if d["confidence"] > 0.4]
        if len(high_conf) >= 5:
            reasons.append("多目标复杂场景")

        if reasons:
            return True, "；".join(reasons[:3])  # 最多保留3个原因
        return False, ""

    # ==================== 工具函数：类别颜色映射 ====================
    def _get_color(self, cls_name: str) -> Tuple[int, int, int]:
        """
        不同类别用不同颜色，这样 VLM 和人类看标注图都更直观。
        比如行人用红色（最危险），车辆用绿色。
        """
        color_map = {
            'person': (0, 0, 255),  # 红色
            'car': (0, 255, 0),  # 绿色
            'truck': (255, 0, 0),  # 蓝色
            'bus': (255, 255, 0),  # 青色
            'motorcycle': (255, 0, 255),  # 紫色
            'bicycle': (0, 255, 255),  # 黄色
            'traffic light': (255, 165, 0),  # 橙色
            'stop sign': (128, 0, 128),  # 深紫
        }
        return color_map.get(cls_name, (200, 200, 200))  # 默认灰色
