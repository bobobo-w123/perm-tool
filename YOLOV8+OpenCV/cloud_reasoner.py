# -*- coding: utf-8 -*-
import base64
import aiohttp
import asyncio
from dataclasses import dataclass
from typing import Optional


# ==================== 数据类：VLM 推理结果 ====================
@dataclass
class VLMResult:
    """
    封装云端 VLM 的返回结果。
    为什么要单独定义？因为 VLM 返回需要额外记录推理延迟，方便后面做性能分析。
    """
    frame_idx: int
    timestamp: float
    annotated_path: str  # 被分析的标注图路径
    trigger_reason: str  # 触发原因
    analysis: str  # VLM 生成的文本分析
    latency_ms: float  # API 调用耗时，毫秒


class CloudReasoner:
    """
    云端推理器。
    职责：调用阿里云百炼的 Qwen-VL 视觉语言模型，对关键帧做场景理解。
    为什么叫 Cloud？因为这些调用是走网络的、按量计费的、能力更强的远端服务。
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.api_key = config.QWEN_API_KEY
        self.base_url = config.QWEN_BASE_URL
        self.model = config.QWEN_MODEL

        # Semaphore 信号量：控制同时最多 3 个 API 请求
        # 这是异步并发的安全阀，防止并发过高导致内存或限流问题
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_API)

        # ==================== Prompt 工程 ====================
        # 这是多模态大模型应用的核心之一。
        # 好的 prompt 能让模型输出稳定、结构化、符合业务需求的答案。
        self.system_prompt = """你是一位拥有20年驾龄的驾校教练和高级驾驶安全分析师。
请仔细观察这张行车记录仪画面（图中已用彩色框标注检测到的车辆、行人、交通标志等）。

请从以下三个维度给出专业、简短、结构化的点评：

【风险预警】当前画面中最需要警惕的1-2个风险点；
【行为评价】当前驾驶行为是否规范（车速、车距、车道等）；
【操作建议】具体、可执行的下一步驾驶建议。

要求：
1. 总字数控制在120字以内；
2. 优先关注红色框（行人）和绿色框（近距离车辆）；
3. 如果没有明显风险，直接写"当前场景安全，保持正常驾驶"；
4. 语气严肃专业，不要道歉。"""
        # 为什么给模型看标注图而不是原图？
        # 因为框能引导模型关注重要目标，相当于"软注意力机制"
        # 同时告诉它不同颜色框的含义，让它利用这些先验信息

    # ==================== 图片编码：本地图片转 base64 ====================
    def _encode_image(self, image_path: str) -> str:
        """
        Qwen-VL 接收图片的方式之一：直接传 base64 编码的图片数据。
        优点：不需要把图片上传到图床，本地文件直接传。
        缺点：图片大时请求体也大，一张 1080p 图大约几百 KB 到 1MB。
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode("utf-8")

    # ==================== 构造消息体 ====================
    def _build_messages(self, image_path: str) -> list:
        """
        百炼兼容 OpenAI 的 chat.completions 格式。
        content 是一个列表，包含 image_url 和 text 两种类型的消息。
        """
        base64_img = self._encode_image(image_path)

        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # data URI 格式：告诉模型这是 jpeg 图片的 base64
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请分析这张行车记录仪画面，给出风险预警、行为评价和操作建议。"
                    }
                ]
            }
        ]

    # ==================== 异步调用 VLM ====================
    async def analyze(self, detection_result: "DetectionResult") -> Optional[VLMResult]:
        """
        对单个 DetectionResult 调用 Qwen-VL。
        使用 async/await 实现非阻塞调用，这是整个 pipeline 能并发的关键。
        """
        if not detection_result.should_trigger_vlm:
            return None  # 不该触发的帧直接跳过

        # async with 确保同时最多 MAX_CONCURRENT_API 个请求在执行
        async with self.semaphore:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": self._build_messages(detection_result.annotated_path)
            }

            start_time = asyncio.get_event_loop().time()

            try:
                # aiohttp 是异步 HTTP 客户端，不会阻塞事件循环
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                            # 60秒超时，VLM 推理通常几秒到十几秒
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise Exception(f"API错误 {resp.status}: {text}")

                        data = await resp.json()
                        # OpenAI 格式：choices[0].message.content
                        analysis = data["choices"][0]["message"]["content"]

            except Exception as e:
                # 异常处理：即使 VLM 失败也不让整段视频分析崩溃
                analysis = f"Qwen-VL调用失败: {str(e)}"

            # 计算耗时
            latency = (asyncio.get_event_loop().time() - start_time) * 1000

            return VLMResult(
                frame_idx=detection_result.frame_idx,
                timestamp=detection_result.timestamp,
                annotated_path=detection_result.annotated_path,
                trigger_reason=detection_result.trigger_reason,
                analysis=analysis,
                latency_ms=round(latency, 1)
            )
