# run_evaluation.py
"""
对照实验：
    实验 A  直接 LLM：同一模型，全部原始监控数据放进提示词，不能调用工具
    实验 B  Agent + Tools：同一模型，通过工具查询数据和计算

评判规则（自动、两组相同）：
    结论正确：回答末尾的"诊断标签"属于该 Case 的可接受标签
    关键数值正确：numeric_checks 中的数值出现在回答中（误差在容差内），
                  且 forbidden_numbers 中的数值（编造/擅自修正得到的值）没有出现
    任务成功：结论正确 且 关键数值正确
    Tool 调用正确（仅 B 组）：required_tools 都被调用，且没有调用 forbidden_tools

用法：
    python tests/run_evaluation.py              # 每个 Case 每组运行 3 次
    python tests/run_evaluation.py --repeat 1 --cases C1 C4
"""
import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import DirectLLM, WifiDiagnosisAgent  # noqa: E402

RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "eval_logs"
MODES = {"A_LLM": "直接 LLM", "B_Agent": "Agent + Tools"}


def numbers_in(text: str):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", (text or "").replace(",", ""))]


def check_numeric(answer: str, check: dict) -> bool:
    target, tol = check["value"], check["tolerance_pct"]
    allowed = max(abs(target) * tol / 100, 0.5 if tol == 0 else 0)
    return any(abs(n - target) <= allowed for n in numbers_in(answer))


def judge(case: dict, mode: str, result) -> dict:
    called = [t["name"] for t in result.tool_calls]
    label_ok = result.label in case["expected_label"]
    numeric_ok = (all(check_numeric(result.answer, c) for c in case["numeric_checks"])
                  and not any(check_numeric(result.answer, c) for c in case.get("forbidden_numbers", [])))
    has_numeric = case["numeric_checks"] or case.get("forbidden_numbers")
    row = {
        "case": case["id"], "category": case["category"], "mode": mode,
        "expected_label": "/".join(case["expected_label"]), "predicted_label": result.label or "(未输出)",
        "label_correct": int(label_ok),
        "numeric_correct": int(numeric_ok) if has_numeric else "",
        "task_success": int(label_ok and numeric_ok),
        "tool_correct": "", "tools_called": " → ".join(called),
        "n_tool_calls": len(called), "llm_rounds": result.steps, "elapsed_s": round(result.elapsed_s, 2),
    }
    if mode == "B_Agent":
        row["tool_correct"] = int(set(case["required_tools"]) <= set(called)
                                  and not set(case["forbidden_tools"]) & set(called))
    return row


def run_one(case: dict, mode: str, rep: int) -> dict:
    runner = DirectLLM() if mode == "A_LLM" else WifiDiagnosisAgent()
    try:
        result = runner.run(case["input"])
    except Exception as e:
        from agent import RunResult
        result = RunResult(f"运行失败：{type(e).__name__}: {e}", None)
    row = judge(case, mode, result)
    row["rep"] = rep
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{case['id']}_{mode}_{rep}.json").write_text(json.dumps({
        "case": case, "mode": mode, "rep": rep, "judge": row,
        "tool_calls": result.tool_calls, "answer": result.answer,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def mean(values):
    values = [v for v in values if v != ""]
    return sum(values) / len(values) if values else None


def summarize(rows, cases):
    overall = []
    for mode, name in MODES.items():
        r = [x for x in rows if x["mode"] == mode]
        overall.append({
            "mode": name, "runs": len(r),
            "结论正确率": mean([x["label_correct"] for x in r]),
            "关键数值正确率": mean([x["numeric_correct"] for x in r]),
            "任务成功率": mean([x["task_success"] for x in r]),
            "Tool调用正确率": mean([x["tool_correct"] for x in r]),
            "平均耗时s": mean([x["elapsed_s"] for x in r]),
            "平均工具调用次数": mean([x["n_tool_calls"] for x in r]),
        })
    per_case = []
    for c in cases:
        item = {"case": c["id"], "category": c["category"], "expected": "/".join(c["expected_label"])}
        for mode in MODES:
            r = [x for x in rows if x["mode"] == mode and x["case"] == c["id"]]
            item[f"{mode}_success"] = mean([x["task_success"] for x in r])
            item[f"{mode}_labels"] = ", ".join(x["predicted_label"] for x in sorted(r, key=lambda x: x["rep"]))
        per_case.append(item)
    return overall, per_case


def plot(overall, per_case, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"A_LLM": "#2a78d6", "B_Agent": "#eb6834"}  # 分类色板第 1、2 位
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 2.2]})
    fig.patch.set_facecolor(surface)
    width = 0.36

    metrics = ["结论正确率", "关键数值正确率", "任务成功率"]
    for i, (mode, ov) in enumerate(zip(MODES, overall)):
        xs = [m + (i - 0.5) * (width + 0.02) for m in range(len(metrics))]
        vals = [(ov[m] or 0) * 100 for m in metrics]
        ax1.bar(xs, vals, width, color=colors[mode], label=MODES[mode], zorder=2)
        for x, v in zip(xs, vals):
            ax1.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=ink)
    ax1.set_xticks(range(len(metrics)), metrics)
    ax1.set_title("总体指标", loc="left", color=ink, fontsize=12)

    ids = [c["case"] for c in per_case]
    for i, mode in enumerate(MODES):
        xs = [k + (i - 0.5) * (width + 0.02) for k in range(len(ids))]
        vals = [(c[f"{mode}_success"] or 0) * 100 for c in per_case]
        ax2.bar(xs, vals, width, color=colors[mode], label=MODES[mode], zorder=2)
        for x, v in zip(xs, vals):
            if v < 100:  # 只标注未全部成功的柱子，0% 的柱子否则看不见
                ax2.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=ink)
    ax2.set_xticks(range(len(ids)), ids)
    ax2.set_title("各 Case 任务成功率", loc="left", color=ink, fontsize=12)
    ax2.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.13), ncol=2)

    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.set_ylim(0, 110)
        ax.set_ylabel("%", color=ink2)
        ax.grid(axis="y", color=grid, zorder=0)
        ax.tick_params(colors=ink2, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(ink2)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)


def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3, help="每个 Case 每组的重复次数")
    parser.add_argument("--cases", nargs="*", help="只运行指定 Case，如 C1 C4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replot", action="store_true", help="不调用 LLM，只根据已有 evaluation.csv 重新汇总和出图")
    args = parser.parse_args()

    cases = json.loads((ROOT / "tests" / "test_cases.json").read_text(encoding="utf-8"))
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]

    if args.replot:
        with open(RESULTS_DIR / "evaluation.csv", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in ("label_correct", "numeric_correct", "task_success", "tool_correct", "n_tool_calls", "rep"):
                r[k] = int(r[k]) if r[k] != "" else ""
            r["elapsed_s"] = float(r["elapsed_s"])
        overall, per_case = summarize(rows, cases)
        plot(overall, per_case, RESULTS_DIR / "evaluation_chart.png")
        print(f"已重新生成 {RESULTS_DIR / 'evaluation_chart.png'}")
        return
    jobs = [(c, m, r) for c in cases for m in MODES for r in range(1, args.repeat + 1)]
    print(f"共 {len(jobs)} 次运行（{len(cases)} 个 Case × 2 组 × {args.repeat} 次）")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, *j) for j in jobs]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"[{len(rows):>3}/{len(jobs)}] {row['case']} {row['mode']:<8} #{row['rep']} "
                  f"标签={row['predicted_label']:<18} 成功={row['task_success']} 工具={row['tools_called'] or '-'}")

    rows.sort(key=lambda x: (x["case"], x["mode"], x["rep"]))
    overall, per_case = summarize(rows, cases)
    RESULTS_DIR.mkdir(exist_ok=True)
    write_csv(RESULTS_DIR / "evaluation.csv", rows)
    write_csv(RESULTS_DIR / "summary_overall.csv", overall)
    write_csv(RESULTS_DIR / "summary_per_case.csv", per_case)
    plot(overall, per_case, RESULTS_DIR / "evaluation_chart.png")

    print("\n===== 总体结果 =====")
    for ov in overall:
        print(" | ".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in ov.items()))
    print("\n===== 各 Case =====")
    for c in per_case:
        print(f"{c['case']} {c['category']:<14} 期望={c['expected']:<26} "
              f"A=[{c['A_LLM_labels']}]  B=[{c['B_Agent_labels']}]")
    print(f"\n结果已保存到 {RESULTS_DIR}")


if __name__ == "__main__":
    main()
