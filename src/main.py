# main.py
"""
命令行交互入口。

    python src/main.py                     # 交互模式，连续提问
    python src/main.py -q "图书馆三楼网速很慢"   # 单次提问
    python src/main.py -q "..." --baseline  # 对照：直接问 LLM（不调用工具）
"""
import argparse
import json
import sys
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent import PROJECT_ROOT, DirectLLM, WifiDiagnosisAgent

console = Console()
LOG_DIR = PROJECT_ROOT / "results" / "logs"

EXAMPLES = [
    "图书馆三楼自习区手机 Wi-Fi 信号满格，但下载速度只有 1~2 Mbps，为什么？",
    "宿舍5号楼512房间晚上 Wi-Fi 经常卡，网页打开很慢，帮我看看是什么原因",
    "信道带宽 40 MHz、SNR 20 dB、单空间流，理论容量是多少？实际大概能跑多快？",
]


def show_event(event: str, payload: dict) -> None:
    if event == "thought":
        console.print(f"[dim]🤔 第 {payload['step']} 步思考：{payload['content']}[/dim]")
    elif event == "tool_call":
        console.print(f"[cyan]🔧 第 {payload['step']} 步调用工具[/cyan] [bold]{payload['name']}[/bold]  参数: {payload['arguments']}")
    elif event == "tool_result":
        text = json.dumps(payload["result"], ensure_ascii=False, indent=2)
        style = "red" if "error" in payload["result"] else "green"
        console.print(Panel(text, title=f"返回结果 · {payload['name']}", border_style=style, expand=False))


def save_log(question: str, mode: str, result) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{mode}.json"
    path.write_text(json.dumps({
        "time": datetime.now().isoformat(timespec="seconds"), "mode": mode, "question": question,
        "label": result.label, "steps": result.steps, "elapsed_s": round(result.elapsed_s, 2),
        "tool_calls": result.tool_calls, "answer": result.answer,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.relative_to(PROJECT_ROOT))


def ask(runner, question: str, baseline: bool) -> None:
    console.rule(f"[bold]{'直接 LLM（无工具）' if baseline else 'Wi-Fi 诊断 Agent'}[/bold]")
    console.print(f"[bold yellow]👤 用户：[/bold yellow]{question}\n")
    try:
        result = runner.run(question) if baseline else runner.run(question, on_event=show_event)
    except Exception as e:  # 网络错误、API Key 错误等
        console.print(f"[red]调用 LLM 失败：{type(e).__name__}: {e}[/red]")
        return
    console.print(Panel(Markdown(result.answer), title="📋 诊断结论", border_style="bold blue"))
    console.print(f"[dim]用时 {result.elapsed_s:.1f}s · LLM 轮次 {result.steps} · "
                  f"工具调用 {len(result.tool_calls)} 次 · 日志 {save_log(question, 'baseline' if baseline else 'agent', result)}[/dim]")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="校园 Wi-Fi 诊断 Agent")
    parser.add_argument("-q", "--question", help="单次提问；不填则进入交互模式")
    parser.add_argument("--baseline", action="store_true", help="对照模式：直接使用 LLM，不调用工具")
    args = parser.parse_args()

    runner = DirectLLM() if args.baseline else WifiDiagnosisAgent()
    console.print(f"[dim]模型：{runner.model}[/dim]")

    if args.question:
        ask(runner, args.question, args.baseline)
        return

    console.print(Panel("输入网络问题开始诊断，输入 1/2/3 使用示例问题，输入 q 退出。\n" +
                        "\n".join(f"  {i}. {q}" for i, q in enumerate(EXAMPLES, 1)),
                        title="📶 校园 Wi-Fi 诊断助手", border_style="cyan"))
    while True:
        try:
            question = console.input("[bold yellow]请输入问题 > [/bold yellow]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question in {"1", "2", "3"}:
            question = EXAMPLES[int(question) - 1]
        if question:
            ask(runner, question, args.baseline)


if __name__ == "__main__":
    main()
