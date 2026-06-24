# 🚗 端云协同驾驶场景感知与智能推理系统

> **Edge-Cloud Collaborative Driving Scene Perception & Reasoning System**

基于 **YOLOv8 + Qwen-VL** 的双层智能架构：本地边缘端做实时目标检测与关键帧筛选，云端大模型对高风险场景进行深度语义推理，兼顾实时性与智能化。

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange) ![Qwen-VL](https://img.shields.io/badge/Qwen--VL-阿里云百炼-red) ![Gradio](https://img.shields.io/badge/UI-Gradio-green)

---

## 🎯 项目亮点

| 亮点 | 说明 |
|------|------|
| **端云协同架构** | 本地 YOLO 做快速检测，云端 VLM 做深度推理，各司其职 |
| **智能触发机制** | 不是每帧都调大模型——只有近距离/高危/复杂场景才触发，API 调用节约率 **>50%** |
| **异步并发管线** | `asyncio + Semaphore` 控制并发，多帧同时推理不互相阻塞 |
| **工程化代码** | 模块分离、dataclass 解耦、配置集中管理、带注释说明设计决策 |
| **可演示 Web UI** | Gradio 一键启动，上传视频即可看到检测结果 + VLM 点评 |

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         用户上传视频                              │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     EdgeDetector (边缘端)                         │
│  • OpenCV 抽帧 (可调 FPS)                                        │
│  • YOLOv8 目标检测 + 画框标注                                     │
│  • 误检过滤：宽高比修正、地面标线过滤、位置面积联合过滤             │
│  • 智能触发判断 ── 筛选"值得分析"的关键帧                          │
└──────────────────────────┬───────────────────────────────────────┘
                           │ 仅关键帧
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CloudReasoner (云端)                           │
│  • 异步并发调用 Qwen-VL (阿里云百炼)                               │
│  • 图像 base64 编码直传                                           │
│  • Prompt Engineering：驾校教练角色 + 三维度结构化输出              │
│  • 维度：风险预警 / 行为评价 / 操作建议                            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                         合并输出                                  │
│  • JSON 完整结果 (每帧：检测框 + VLM分析 + 延迟)                    │
│  • 标注图 + 关键帧 Gallery                                         │
│  • Gradio Web UI 交互展示                                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🧠 触发机制（核心工程决策）

不是每帧都调用昂贵的 VLM，而是通过多层规则筛选：

| 触发规则 | 阈值 | 设计理由 |
|----------|------|----------|
| 近距离大目标 | 画面占比 > 8% | 马上可能碰撞，必须分析 |
| 正前方近处行人/非机动车 | 中心区域 + 画面下方 | 高危场景 |
| 中心区域交通灯/停车标志 | 置信度 > 0.3 | YOLO 只能检测"有个灯"，VLM 才能理解颜色/状态 |
| 多目标复杂场景 | 高置信目标 ≥ 5 个 | 物体间关系需要语言理解 |

同时做了大量误检过滤：宽高比修正（电动车从 person 中分离）、地面标线过滤、边缘低质目标过滤。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install ultralytics opencv-python gradio aiohttp torch
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件（已加入 `.gitignore`，不会被提交）：

```bash
# .env
QWEN_API_KEY=你的阿里云百炼 API Key
```

程序启动时会自动加载 `.env` 文件。如果没找到，再读系统环境变量。

> 获取 Key：[阿里云百炼控制台](https://bailian.console.aliyun.com/)

### 3. 下载模型

项目默认使用 `yolov8m.pt`，首次运行会自动下载。也可手动放置 `yolov8n.pt` / `yolov8s.pt` 来切换轻量模型。

### 4. 命令行运行

```bash
python async_pipeline.py
```

处理 `test_video.mp4`，结果保存至 `outputs/`。

### 5. 启动 Web UI

```bash
python web_app.py
```

浏览器打开 `http://127.0.0.1:7861`，上传视频即可交互分析。

---

## 📊 Demo 效果

![Web UI 界面预览](#)

系统内置 **Gradio Web 交互界面**，暗色专业主题，支持：

- 🔝 顶部 Hero 区展示项目信息与技术栈
- 📊 **实时统计卡片**：总帧数、触发数、API 节约率、平均延迟
- 🧠 **关键帧深度分析**：每帧卡片化展示，VLM 三维度分析（风险预警 / 行为评价 / 操作建议）分色高亮
- 📷 **全量检测帧**：所有帧的画廊视图，标注 YOLO 检测框
- 🏷️ **检测目标标签**：类别 chip 化展示（行人/车辆/交通灯）
- 🟢/🔴 **风险等级标记**：高风险帧红色标签，安全帧绿色标签
- 📋 **处理日志**：实时进度反馈

### 真实视频分析指标

```
总处理帧数: 288
触发 VLM 帧数: 141
API 调用节约率: 51%
VLM 平均延迟: ~2100ms
VLM 模型: qwen-vl-max
```

### VLM 分析样例

```
【风险预警】行人（红色框）在路口横穿，存在潜在碰撞风险。
绿色框内车辆距离过近，制动空间不足。

【行为评价】车速适中，但与前方车辆距离较近，需保持安全车距。

【操作建议】立即减速，保持至少3秒以上跟车间距；
密切观察行人动态，准备随时停车避让。
```

---

## 📁 项目结构

```
YOLOV8+OpenCV/
├── config.py              # 配置集中管理（模型/设备/触发阈值/并发数）
├── edge_detector.py        # 边缘端检测器：抽帧 + YOLO 检测 + 触发判断
├── cloud_reasoner.py       # 云端推理器：异步调用 Qwen-VL
├── async_pipeline.py       # 异步处理管线：串联完整流程
├── web_app.py              # Gradio Web 交互界面
├── yolov8n.pt              # YOLOv8 Nano (可选，最小最快)
├── yolov8s.pt              # YOLOv8 Small (可选)
├── yolov8m.pt              # YOLOv8 Medium (默认)
├── test_video.mp4          # 示例测试视频
├── .env                    # API Key (不提交，需自行创建)
├── .gitignore              # Git 忽略规则
├── requirements.txt        # Python 依赖
├── outputs/                # 输出：标注帧 + analysis_result.json
└── web_outputs/            # Web UI 处理结果
```

---

## 🔧 配置项

所有参数集中在 `config.py`，支持环境变量覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_API_KEY` | `os.getenv("QWEN_API_KEY")` | 阿里云百炼 API Key |
| `QWEN_MODEL` | `qwen-vl-max` | 视觉语言模型选择 |
| `YOLO_MODEL` | `yolov8m.pt` | YOLO 模型路径 (n/s/m 可选) |
| `DEVICE` | `0` | GPU 编号，`cpu` 用 CPU |
| `FPS` | `0.5` | 抽帧频率（帧/秒） |
| `MAX_CONCURRENT_API` | `8` | 最大并发 VLM 请求数 |

---

## 📝 技术栈

- **目标检测**：Ultralytics YOLOv8
- **视频处理**：OpenCV (`cv2`)
- **大模型推理**：阿里云百炼 Qwen-VL (兼容 OpenAI 接口)
- **异步框架**：Python `asyncio` + `aiohttp`
- **Web UI**：Gradio
- **数据结构**：Python `dataclass`

---

## 🤔 为什么选 YOLOv8 + Qwen-VL 而不是端到端方案？

- **成本**：VLM 按 token 计费，每帧都调不现实。YOLO 做前置过滤，节约 50%+ API 开销
- **延迟**：YOLO 推理 < 50ms/帧 (GPU)，VLM 推理 1-3s/次。异步并发可缓解但本质差距大
- **可控性**：YOLO 输出结构化（bbox + 置信度），可做精确过滤规则；纯 VLM 输出靠 prompt 约束，不够稳定
- **可演进**：端侧检测可以换更强模型，云端推理可以换 GPT-4V，两部分独立升级

---

## 📌 TODO / 改进方向

- [ ] 添加 `requirements.txt`
- [ ] 视频流实时输入（非离线文件）
- [ ] 多模型对比 Benchmark（YOLOv8n vs m，Qwen-VL vs GPT-4V）
- [ ] TensorRT / ONNX 推理加速
- [ ] 单元测试
- [ ] Docker 部署

---

## 📄 License

MIT
