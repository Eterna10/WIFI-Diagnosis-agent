# agent.py
"""
Agent 层：基于 OpenAI 兼容接口的 function calling 实现单 Agent 工具调用循环。

    用户问题 → LLM 决定调用哪个工具 → 执行工具 → 工具结果回传 LLM → ... → LLM 输出结论

同时提供 DirectLLM（对照组）：同一个模型、同样的提示词要求，但不能调用工具，
只能读取直接塞进提示词里的原始监控数据。
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from prompts import AGENT_SYSTEM_PROMPT, BASELINE_SYSTEM_PROMPT
from tools import DATA_FILE, TOOL_SCHEMAS, execute_tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_llm_config() -> Dict[str, str]:
    """优先读取项目目录下的 .env，没有则读取上一级（Day4）目录的 .env"""
    for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"):
        if env_path.exists():
            load_dotenv(env_path)
            break
    config = {
        "model": os.getenv("LLM_MODEL_ID"),
        "api_key": os.getenv("LLM_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL"),
    }
    missing = [k for k, v in config.items() if not v]
    if missing:
        raise RuntimeError(f"缺少 LLM 配置：{missing}，请在 .env 中设置 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL")
    return config


def parse_label(text: str) -> Optional[str]:
    """从回答最后的 '诊断标签: XXX' 中解析标签"""
    m = re.findall(r"诊断标签\s*[:：]\s*\**\s*([A-Z_]+)", text or "")
    return m[-1] if m else None


@dataclass
class RunResult:
    answer: str
    label: Optional[str]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    elapsed_s: float = 0.0
    messages: List[Dict[str, Any]] = field(default_factory=list)


class WifiDiagnosisAgent:
    def __init__(self, max_steps: int = 8, temperature: float = 0.2):
        cfg = load_llm_config()
        self.model = cfg["model"]
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=120)
        self.max_steps = max_steps
        self.temperature = temperature

    def run(self, question: str, on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> RunResult:
        """
        on_event(event_type, payload) 用于界面实时展示过程：
        'thought'（LLM 在调用工具前的说明）、'tool_call'、'tool_result'、'final'
        """
        emit = on_event or (lambda *_: None)
        start = time.time()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_log: List[Dict[str, Any]] = []

        for step in range(1, self.max_steps + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=self.temperature,
            )
            msg = resp.choices[0].message

            # 没有工具调用 → LLM 已给出最终回答
            if not msg.tool_calls:
                answer = msg.content or ""
                messages.append({"role": "assistant", "content": answer})
                emit("final", {"answer": answer})
                return RunResult(answer, parse_label(answer), tool_log, step, time.time() - start, messages)

            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            }
            # 部分推理模型要求在多轮工具调用中回传 reasoning_content
            reasoning = getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning_content")
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)
            if msg.content:
                emit("thought", {"step": step, "content": msg.content})

            for tc in msg.tool_calls:
                emit("tool_call", {"step": step, "name": tc.function.name, "arguments": tc.function.arguments})
                result = execute_tool(tc.function.name, tc.function.arguments)
                emit("tool_result", {"step": step, "name": tc.function.name, "result": result})
                tool_log.append({"step": step, "name": tc.function.name,
                                 "arguments": tc.function.arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False)})

        answer = (f"已达到最大推理步数（{self.max_steps}），未能完成分析。目前数据不足，无法确定原因。\n"
                  "诊断标签: INSUFFICIENT_DATA")
        emit("final", {"answer": answer})
        return RunResult(answer, parse_label(answer), tool_log, self.max_steps, time.time() - start, messages)


class DirectLLM:
    """对照组：不调用工具，把全部原始监控数据直接放进提示词，由 LLM 自行查找和计算"""

    def __init__(self, temperature: float = 0.2):
        cfg = load_llm_config()
        self.model = cfg["model"]
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=120)
        self.temperature = temperature
        self.raw_data = DATA_FILE.read_text(encoding="utf-8")

    def run(self, question: str) -> RunResult:
        start = time.time()
        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": f"【网管监控原始数据（JSON）】\n{self.raw_data}\n\n【用户问题】\n{question}"},
        ]
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature)
        answer = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": answer})
        return RunResult(answer, parse_label(answer), [], 1, time.time() - start, messages)
