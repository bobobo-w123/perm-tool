# -*- coding: utf-8 -*-
import asyncio
import json
from pathlib import Path
from typing import Optional, Callable

from config import Config
from edge_detector import EdgeDetector
from cloud_reasoner import CloudReasoner


class AsyncPipeline:
    """
    异步处理管线。
    职责：把"本地检测"和"云端推理"串联起来，并支持进度回调。

    为什么叫 Pipeline？
    因为视频分析是一个多阶段流水线：抽帧 → 检测 → 触发判断 → VLM 推理 → 合并输出。
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.detector = EdgeDetector(self.config)
        self.reasoner = None  # 延迟初始化，避免 Semaphore 绑定到旧事件循环

    # ==================== 主流程：处理一个视频 ====================
    async def process_video(self, video_path: str,
                            output_dir: str = "outputs",
                            progress_callback: Optional[Callable] = None) -> dict:
        """
        完整流程：
        1. 本地 YOLO 检测所有帧
        2. 筛选出需要 VLM 的帧
        3. 异步并发调用 VLM
        4. 合并本地结果和 VLM 结果
        5. 保存 JSON 并返回摘要

        progress_callback: 用于 Gradio 前端展示进度
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ==================== 阶段1：本地检测 ====================
        if progress_callback:
            progress_callback("开始本地 YOLO 检测...", 0.05)

        detection_results = self.detector.extract_frames(video_path, output_dir)

        # 筛选出触发 VLM 的帧
        trigger_results = [d for d in detection_results if d.should_trigger_vlm]

        if progress_callback:
            progress_callback(
                f"YOLO检测完成：{len(detection_results)}帧，"
                f"触发VLM {len(trigger_results)}帧", 0.3
            )

        # ==================== 阶段2：异步 VLM 推理 ====================
        vlm_results = []

        if trigger_results:
            # 每次调用都新建 CloudReasoner：Semaphore 必须绑定到当前事件循环
            reasoner = CloudReasoner(self.config)

            total = len(trigger_results)
            completed = 0

            async def analyze_with_progress(det):
                """
                包装一下 analyze，每完成一个就更新进度。
                这里用了闭包，捕获 completed 变量。
                """
                nonlocal completed
                result = await reasoner.analyze(det)
                completed += 1

                if progress_callback:
                    progress_callback(
                        f"VLM推理中... {completed}/{total}",
                        0.3 + 0.65 * completed / total  # 进度从30%到95%
                    )
                return result

            # 创建所有任务
            tasks = [analyze_with_progress(d) for d in trigger_results]
            # asyncio.gather 并发执行所有任务
            # 因为 Semaphore 的存在，实际同时只有 3 个在跑
            vlm_results = await asyncio.gather(*tasks)

            # 过滤掉 None（理论上没有，因为触发的一定会调用）
            vlm_results = [r for r in vlm_results if r is not None]

        # ==================== 阶段3：合并与保存 ====================
        final_results = self._merge_results(detection_results, vlm_results)
        summary = self._save_results(final_results, output_dir)

        if progress_callback:
            progress_callback("处理完成！", 1.0)

        return summary

    # ==================== 合并结果：让每帧都有完整信息 ====================
    def _merge_results(self, detections, vlm_results):
        """
        用 frame_idx 作为 key，把 VLM 结果合并到对应的检测帧里。
        这样 JSON 里每一帧都有：检测框 + 触发原因 + 大模型分析。
        """
        vlm_map = {r.frame_idx: r for r in vlm_results}
        merged = []

        for det in detections:
            item = {
                "frame_idx": det.frame_idx,
                "timestamp": det.timestamp,
                "image_path": det.image_path,
                "annotated_path": det.annotated_path,
                "detections": det.detections,
                "trigger_reason": det.trigger_reason,
            }

            if det.frame_idx in vlm_map:
                vlm = vlm_map[det.frame_idx]
                item["vlm_analysis"] = vlm.analysis
                item["vlm_latency_ms"] = vlm.latency_ms
            else:
                # 未触发 VLM 的帧，这两个字段为 None
                item["vlm_analysis"] = None
                item["vlm_latency_ms"] = None

            merged.append(item)

        return merged

    # ==================== 保存结果 ====================
    def _save_results(self, merged, output_dir: Path) -> dict:
        """
        把完整结果保存为 JSON，方便前端读取和后续分析。
        """
        result_path = output_dir / "analysis_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        trigger_count = sum(1 for m in merged if m["vlm_analysis"])

        return {
            "total_frames": len(merged),
            "trigger_frames": trigger_count,
            "saved_path": str(result_path),
            "frames": merged
        }


# ==================== 命令行入口 ====================
async def main():
    pipeline = AsyncPipeline()
    result = await pipeline.process_video("test_video.mp4")
    print(f"\n总帧数: {result['total_frames']}")
    print(f"触发帧数: {result['trigger_frames']}")
    print(f"结果保存: {result['saved_path']}")


if __name__ == "__main__":
    asyncio.run(main())
