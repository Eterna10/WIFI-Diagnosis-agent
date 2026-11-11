# draw_architecture.py
"""生成系统架构图 docs/architecture.png：python docs/draw_architecture.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#cde2fb", "#fbd9c9", "#c9efe0", "#f0efec"

fig, ax = plt.subplots(figsize=(12, 7.2))
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 120)
ax.set_ylim(0, 72)
ax.axis("off")


def box(x, y, w, h, title, sub="", color=GRAY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color, ec=INK2, lw=1))
    ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), title, ha="center", va="center",
            fontsize=11, color=INK, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=8.5, color=INK2)


def arrow(p, q, text="", both=False, offset=(1, 0)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=12,
                                 color=INK2, lw=1.2))
    if text:
        ax.text((p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1], text,
                fontsize=8.5, color=INK2, ha="left", va="center")


box(44, 62, 32, 7.5, "用户（学生 / 网管）", "命令行 main.py 输入网络问题")
box(30, 41, 60, 13, "WifiDiagnosisAgent（agent.py）",
    "系统提示词：工作原则 + 参考判据 + 输出格式\n循环：LLM 决策 → 执行工具 → 结果回传 → … → 结论", BLUE)
box(96, 43, 21, 9, "LLM", "DeepSeek（.env 配置）\nOpenAI 兼容接口", ORANGE)
box(2, 19, 26, 10, "list_access_points", "按地点查 AP 编号", AQUA)
box(31, 19, 26, 10, "get_ap_metrics", "RSSI / 底噪 / SNR\n终端数 / 利用率 / 重传", AQUA)
box(60, 19, 26, 10, "get_uplink_status", "上联速率 / 利用率\nRTT / 丢包 / CRC", AQUA)
box(89, 19, 29, 10, "analyze_wireless_capacity", "香农容量 / MCS 估计\n实际吞吐 / 人均带宽", AQUA)
box(18, 3, 54, 8, "data/sample_data.json", "模拟校园网管监控数据（6 个 AP）")
box(80, 3, 34, 8, "802.11ax MCS 表 + 香农公式", "tools.py 内置计算")
box(2, 62, 34, 7.5, "诊断报告输出", "Incident … Recommendation + 诊断标签", GRAY)

arrow((60, 62), (60, 54.6), "问题")
arrow((30, 47.5), (19, 62), "结论 + 依据", offset=(-12, 1))
arrow((90.4, 47.5), (95.6, 47.5), both=True)
ax.text(93, 45, "请求/响应", fontsize=8, color=INK2, ha="center")
for cx in (15, 44, 73, 103.5):
    arrow((60, 40.6), (cx, 29.4), both=True)
ax.text(60, 34.5, "function calling：调用工具 / 返回 JSON 结果", fontsize=8.5, color=INK2, ha="center",
        bbox=dict(fc=SURFACE, ec="none", pad=1.5))
for cx in (15, 44, 73):
    arrow((cx, 18.6), (min(max(cx, 24), 66), 11.4))
arrow((103.5, 18.6), (97, 11.4))

ax.text(60, 0.2, "工具只返回数据和计算结果，不返回“根本原因”；原因由 Agent 根据证据推理", fontsize=8.5,
        color=INK2, ha="center")
out = Path(__file__).resolve().parent / "architecture.png"
fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
print(f"已保存 {out}")
