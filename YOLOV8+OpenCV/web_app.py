# -*- coding: utf-8 -*-
"""
端云协同驾驶场景感知系统 — Web 交互界面
技术栈：Gradio + 自定义 CSS
设计目标：Demo 级视觉效果，面试时打开就能镇场子
"""
import asyncio
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from config import Config
from async_pipeline import AsyncPipeline


class GradioApp:
    def __init__(self):
        self.config = Config()
        self.pipeline = AsyncPipeline(self.config)
        self.base_output_dir = "web_outputs"

    CUSTOM_CSS = """
    /* ===== 全局 ===== */
    body, .gradio-container {
        background: #0b0f19 !important;
        color: #e0e6f0 !important;
        font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif !important;
    }
    .gradio-container .prose { color: #c0cddc; }

    /* ===== Hero 标题区 ===== */
    .hero-section {
        background: linear-gradient(135deg, #0d1a3a 0%, #1a2d5c 50%, #0b1225 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 16px;
        padding: 30px 36px 22px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 20%, rgba(99,179,237,0.08) 0%, transparent 60%),
                    radial-gradient(circle at 70% 80%, rgba(56,178,172,0.06) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 28px; font-weight: 700;
        background: linear-gradient(135deg, #63b3ed, #48bb78);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 8px 0;
        position: relative;
    }
    .hero-subtitle {
        font-size: 14px; color: #8899b4;
        margin: 0; position: relative;
        letter-spacing: 0.5px;
    }
    .hero-badges {
        display: flex; gap: 10px;
        margin-top: 14px; flex-wrap: wrap;
        position: relative;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(99,179,237,0.1);
        border: 1px solid rgba(99,179,237,0.25);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px; color: #a0c4f0;
    }
    .hero-badge .dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-green  { background: #48bb78; box-shadow: 0 0 6px #48bb78; }
    .dot-blue   { background: #63b3ed; box-shadow: 0 0 6px #63b3ed; }
    .dot-purple { background: #b794f4; box-shadow: 0 0 6px #b794f4; }

    /* ===== 统计卡片（4列网格） ===== */
    .stat-cards {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 10px 0 18px;
    }
    .stat-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99,179,237,0.4);
    }
    .stat-card .stat-value {
        font-size: 28px; font-weight: 700;
        background: linear-gradient(135deg, #63b3ed, #a0c4f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    .stat-card .stat-label {
        font-size: 11px; color: #6b7d95;
        margin-top: 6px;
    }
    .stat-card.accent-green .stat-value {
        background: linear-gradient(135deg, #48bb78, #81e6a7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-card.accent-orange .stat-value {
        background: linear-gradient(135deg, #ed8936, #f6ad55);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ===== VLM 分析卡片（纯文字面板） ===== */
    .vlm-panel-wrapper {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(99,179,237,0.12);
        border-radius: 14px;
        padding: 20px 22px;
        margin: 6px 0;
    }
    .vlm-panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .vlm-panel-header .frame-info {
        font-size: 13px; color: #8899b4;
        font-weight: 600;
    }
    .vlm-panel-header .frame-tag {
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 10px;
        font-weight: 600;
    }
    .frame-tag.risk-high {
        background: rgba(248,113,113,0.15);
        color: #f87171;
        border: 1px solid rgba(248,113,113,0.3);
    }
    .frame-tag.risk-medium {
        background: rgba(251,191,36,0.12);
        color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.3);
    }
    .frame-tag.risk-low {
        background: rgba(72,187,120,0.12);
        color: #48bb78;
        border: 1px solid rgba(72,187,120,0.3);
    }

    .vlm-section {
        background: rgba(99,179,237,0.04);
        border-left: 3px solid #63b3ed;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .vlm-section:last-child { margin-bottom: 0; }
    .vlm-section h4 {
        margin: 0 0 6px 0;
        font-size: 13px; color: #63b3ed;
        font-weight: 600;
    }
    .vlm-section p {
        margin: 0;
        font-size: 12.5px;
        line-height: 1.65;
        color: #c0cddc;
    }

    /* ===== 目标芯片 ===== */
    .objects-row {
        display: flex; gap: 6px;
        flex-wrap: wrap;
        margin: 10px 0;
    }
    .obj-chip {
        display: inline-block;
        font-size: 11px;
        padding: 2px 9px;
        border-radius: 10px;
        font-weight: 500;
    }
    .obj-chip.person    { background: rgba(248,113,113,0.15); color: #f87171; }
    .obj-chip.car       { background: rgba(72,187,120,0.15); color: #48bb78; }
    .obj-chip.truck     { background: rgba(99,179,237,0.15); color: #63b3ed; }
    .obj-chip.bus       { background: rgba(56,178,172,0.15); color: #38b2ac; }
    .obj-chip.motorcycle { background: rgba(183,148,244,0.15); color: #b794f4; }
    .obj-chip.bicycle   { background: rgba(246,173,85,0.15); color: #f6ad55; }
    .obj-chip.traffic_light { background: rgba(251,191,36,0.15); color: #fbbf24; }
    .obj-chip.stop_sign  { background: rgba(248,113,113,0.15); color: #f87171; }

    /* ===== 分析区：左图右文 ===== */
    .analysis-layout {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        align-items: start;
    }
    @media (max-width: 1000px) {
        .analysis-layout { grid-template-columns: 1fr; }
        .stat-cards { grid-template-columns: repeat(2, 1fr); }
    }

    /* ===== Tabs 美化 ===== */
    .tab-nav button {
        background: transparent !important;
        color: #5a6d85 !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.2s !important;
    }
    .tab-nav button.selected {
        color: #63b3ed !important;
        border-bottom-color: #63b3ed !important;
    }
    """

    # ==================================================================
    # 静态 HTML 片段生成
    # ==================================================================

    @staticmethod
    def hero_html():
        return """
        <div class="hero-section">
            <div class="hero-title">🚗 端云协同驾驶场景感知与智能推理系统</div>
            <div class="hero-subtitle">Edge-Cloud Collaborative Driving Perception & Reasoning</div>
            <div class="hero-badges">
                <span class="hero-badge"><span class="dot dot-green"></span> YOLOv8m 边缘检测</span>
                <span class="hero-badge"><span class="dot dot-blue"></span> Qwen-VL 云端推理</span>
                <span class="hero-badge"><span class="dot dot-purple"></span> 异步并发管线</span>
            </div>
        </div>
        """

    @staticmethod
    def stat_cards_html(total: int, trigger: int, save_rate: float, avg_latency: float) -> str:
        tpl = """
        <div class="stat-cards">
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">📷 总处理帧数</div>
            </div>
            <div class="stat-card accent-orange">
                <div class="stat-value">{trigger}</div>
                <div class="stat-label">🧠 触发 VLM 推理</div>
            </div>
            <div class="stat-card accent-green">
                <div class="stat-value">{rate:.1%}</div>
                <div class="stat-label">💰 API 调用节约率</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{lat:.0f}<span style="font-size:14px"> ms</span></div>
                <div class="stat-label">⏱️ VLM 平均延迟</div>
            </div>
        </div>
        """
        return tpl.format(total=total, trigger=trigger, rate=save_rate, lat=avg_latency)

    @staticmethod
    def vlm_analysis_html(frame: dict) -> str:
        """把单帧的 VLM 分析渲染成结构化 HTML 面板"""
        trigger_reason = frame.get("trigger_reason", "")
        latency = frame.get("vlm_latency_ms") or 0
        detections = frame.get("detections", [])

        # 风险等级
        risk_class = "risk-medium"
        risk_text = "⚠️ 中等风险"
        if any(kw in trigger_reason for kw in ("行人", "碰撞", "紧急")):
            risk_class = "risk-high"
            risk_text = "🔴 高风险"
        elif not trigger_reason:
            risk_class = "risk-low"
            risk_text = "✅ 安全"

        # 解析 VLM 三维度
        vlm_text = frame.get("vlm_analysis") or ""
        sections = []
        current_dim = ""
        current_body = ""
        for line in vlm_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("【") and "】" in line:
                if current_dim:
                    sections.append((current_dim, current_body.strip()))
                current_dim = line
                current_body = ""
            else:
                current_body += line + "\n"
        if current_dim:
            sections.append((current_dim, current_body.strip()))

        vlm_block = ""
        for dim_title, dim_body in sections:
            if dim_body:
                vlm_block += f"""
                <div class="vlm-section">
                    <h4>{dim_title}</h4>
                    <p>{dim_body}</p>
                </div>"""

        # 检测目标 chips
        obj_counts = {}
        for det in detections:
            name = det["class"]
            obj_counts[name] = obj_counts.get(name, 0) + 1
        chips = "".join(
            f'<span class="obj-chip {name.replace(" ", "_")}">{name} ×{cnt}</span>'
            for name, cnt in obj_counts.items()
        )

        ts = frame["timestamp"]
        return f"""
        <div class="vlm-panel-wrapper">
            <div class="vlm-panel-header">
                <span class="frame-info">⏱ {ts:.1f}s &nbsp;|&nbsp; 📍 帧 #{frame['frame_idx']} &nbsp;|&nbsp; ⚡ {latency:.0f}ms</span>
                <span class="frame-tag {risk_class}">{risk_text} · {trigger_reason}</span>
            </div>
            <div class="objects-row">{chips}</div>
            {vlm_block}
        </div>
        """

    # ==================================================================
    # 主处理函数
    # ==================================================================
    def _process(self, video_path: str):
        if not video_path:
            empty_stats = self.stat_cards_html(0, 0, 0, 0)
            empty_msg = "<div style='color:#5a6d85;text-align:center;padding:60px;'>请先上传视频</div>"
            return empty_stats, [], empty_msg, [], empty_msg, "⏳ 等待上传..."

        progress_msgs = []

        def progress_callback(msg: str, prog: float):
            progress_msgs.append(msg)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(video_path).stem
        session_dir = Path(self.base_output_dir) / f"{ts}_{stem}"

        result = asyncio.run(
            self.pipeline.process_video(video_path, str(session_dir), progress_callback)
        )

        frames: List[Dict] = result["frames"]
        vlm_frames = [f for f in frames if f.get("vlm_analysis")]

        # ===== 统计卡片 =====
        latencies = [f["vlm_latency_ms"] for f in vlm_frames if f.get("vlm_latency_ms")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        save_rate = 1 - result["trigger_frames"] / max(result["total_frames"], 1)
        stats_html = self.stat_cards_html(result["total_frames"], result["trigger_frames"], save_rate, avg_latency)

        # ===== 关键帧 Gallery（图片）+ HTML（分析） =====
        key_gallery = [(f["annotated_path"], f"⏱ {f['timestamp']:.1f}s | {f.get('trigger_reason','')}")
                       for f in vlm_frames]

        # 所有关键帧分析合并成一段 HTML（滚动区域）
        key_analysis_html = ""
        for i, f in enumerate(vlm_frames):
            key_analysis_html += self.vlm_analysis_html(f)
        if not key_analysis_html:
            key_analysis_html = (
                "<div style='color:#5a6d85;text-align:center;padding:60px;'>"
                "📭 未触发 VLM 推理 —— 当前场景安全"
                "</div>"
            )

        # ===== 全量帧 Gallery（所有标注帧） =====
        all_gallery = [(f["annotated_path"],
                        f"⏱ {f['timestamp']:.1f}s | {'🧠 VLM' if f.get('vlm_analysis') else '✅ 安全'} | {len(f.get('detections',[]))} 目标")
                       for f in frames[:120]]  # 限制 120 帧

        all_analysis_html = ""
        for f in frames[:120]:
            if f.get("vlm_analysis"):
                all_analysis_html += self.vlm_analysis_html(f)

        # ===== 进度日志 =====
        progress_text = "\n".join(f"▸ {m}" for m in progress_msgs[-8:])

        return stats_html, key_gallery, key_analysis_html, all_gallery, all_analysis_html, progress_text

    # ==================================================================
    # 构建 UI
    # ==================================================================
    def build_ui(self):
        with gr.Blocks(
            title="端云协同驾驶场景感知系统",
            css=self.CUSTOM_CSS,
            theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
        ) as app:

            gr.HTML(self.hero_html())

            # 上传 + 触发规则
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    video_input = gr.Video(label="📤 上传行车记录仪视频", height=260)
                    analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 🔍 触发规则（云端 VLM）
                    | 条件 | 阈值 |
                    |------|------|
                    | 🚶 行人/非机动车 | 正前方近距离 |
                    | 🚗 近距离车辆 | 画面占比 > 8% |
                    | 🚦 交通信号灯 | 需理解颜色状态 |
                    | 📊 复杂场景 | ≥ 5 高置信目标 |

                    > 💡 只有命中以上规则的关键帧才会调用云端大模型
                    """)

            # 统计卡片
            stats_html = gr.HTML(self.stat_cards_html(0, 0, 0, 0))

            # 进度
            progress_text = gr.Textbox(label="📋 处理日志", value="⏳ 等待上传视频...", interactive=False, lines=2, max_lines=6)

            # 结果区
            with gr.Tabs():
                with gr.TabItem("🧠 关键帧分析", id="tab-key"):
                    gr.Markdown("*仅展示触发云端 Qwen-VL 推理的关键帧*")
                    with gr.Row():
                        with gr.Column(scale=1):
                            key_gallery = gr.Gallery(
                                label="📸 关键帧标注图",
                                columns=2, rows=3, height=600,
                                object_fit="contain",
                                show_label=True,
                            )
                        with gr.Column(scale=1):
                            key_analysis = gr.HTML(
                                "<div style='color:#5a6d85;text-align:center;padding:40px;'>等待处理...</div>"
                            )

                with gr.TabItem("📷 全量检测帧", id="tab-all"):
                    gr.Markdown("*展示所有抽帧的 YOLO 检测结果。🧠 标记表示该帧触发了 VLM 推理*")
                    all_gallery = gr.Gallery(
                        label="全量检测帧",
                        columns=4, rows=4, height=800,
                        object_fit="contain",
                    )
                    all_analysis = gr.HTML(
                        "<div style='color:#5a6d85;text-align:center;padding:20px;'>等待处理...</div>"
                    )

            # 绑定
            analyze_btn.click(
                fn=self._process,
                inputs=[video_input],
                outputs=[stats_html, key_gallery, key_analysis, all_gallery, all_analysis, progress_text],
                show_progress="full",
            )

        return app


def main():
    app = GradioApp()
    demo = app.build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)


if __name__ == "__main__":
    main()
