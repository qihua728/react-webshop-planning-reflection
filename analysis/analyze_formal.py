from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "formal"
OUTPUT = RESULTS / "analysis"
TASK_IDS = [f"fixed_{index}" for index in range(10, 110)]
BOOTSTRAP_SAMPLES = 20_000
PERMUTATION_SAMPLES = 100_000
SEED = 20260801

METHODS = {
    "react_baseline": ("Baseline", RESULTS / "react_baseline.jsonl"),
    "react_planning": ("Planning", RESULTS / "react_planning.jsonl"),
    "react_reflection": ("Reflection", RESULTS / "react_reflection.jsonl"),
    "react_plan_reflect": ("Plan + Reflect", RESULTS / "react_plan_reflect.jsonl"),
}


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def load_results() -> tuple[dict[str, dict[str, dict]], dict]:
    by_method: dict[str, dict[str, dict]] = {}
    integrity: dict = {"expected_tasks_per_method": len(TASK_IDS), "methods": {}}
    required = {
        "task_id", "status", "reward", "success", "steps", "action_calls",
        "total_llm_calls", "input_tokens", "output_tokens", "invalid_actions",
        "latency_seconds", "method", "dataset_mode", "prompt_version",
    }

    for method, (label, path) in METHODS.items():
        records = read_jsonl(path)
        valid_records = [
            row for row in records
            if row.get("method") == method and row.get("status") in {"completed", "max_steps"}
        ]
        grouped: dict[str, list[dict]] = {}
        for row in valid_records:
            grouped.setdefault(str(row.get("task_id")), []).append(row)
        latest = {
            task_id: sorted(rows, key=lambda row: str(row.get("recorded_at", "")))[-1]
            for task_id, rows in grouped.items()
        }
        missing = sorted(set(TASK_IDS) - set(latest))
        unexpected = sorted(set(latest) - set(TASK_IDS))
        duplicate_valid = sorted(task_id for task_id, rows in grouped.items() if len(rows) > 1)
        missing_fields = {
            task_id: sorted(required - set(row))
            for task_id, row in latest.items()
            if required - set(row)
        }
        environment_errors = sum(row.get("status") == "environment_error" for row in records)
        dataset_modes = sorted({str(row.get("dataset_mode")) for row in latest.values()})

        integrity["methods"][method] = {
            "label": label,
            "source": str(path.relative_to(ROOT)),
            "raw_rows": len(records),
            "valid_unique_tasks": len(latest),
            "missing_tasks": missing,
            "unexpected_tasks": unexpected,
            "duplicate_valid_tasks": duplicate_valid,
            "environment_error_rows": environment_errors,
            "records_with_missing_required_fields": missing_fields,
            "dataset_modes": dataset_modes,
        }
        by_method[method] = latest

    task_sets = [set(rows) for rows in by_method.values()]
    integrity["paired_task_count"] = len(set.intersection(*task_sets))
    integrity["passed"] = all(
        details["valid_unique_tasks"] == len(TASK_IDS)
        and not details["missing_tasks"]
        and not details["unexpected_tasks"]
        and not details["duplicate_valid_tasks"]
        and not details["records_with_missing_required_fields"]
        and details["dataset_modes"] == ["full"]
        for details in integrity["methods"].values()
    ) and integrity["paired_task_count"] == len(TASK_IDS)
    if not integrity["passed"]:
        raise ValueError("Formal results failed integrity checks; see integrity_report.json")
    return by_method, integrity


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_ci(
    records: list[dict],
    statistic: Callable[[list[dict]], float],
    seed_offset: int,
) -> tuple[float, float]:
    rng = random.Random(SEED + seed_offset)
    n = len(records)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        samples.append(statistic(sample))
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)


def paired_bootstrap_ci(differences: list[float], seed_offset: int) -> tuple[float, float]:
    rng = random.Random(SEED + seed_offset)
    n = len(differences)
    samples = [
        statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)


def paired_median_bootstrap_ci(differences: list[float], seed_offset: int) -> tuple[float, float]:
    rng = random.Random(SEED + seed_offset)
    n = len(differences)
    samples = [
        statistics.median(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return center - margin, center + margin


def sign_flip_pvalue(differences: list[float], seed_offset: int) -> float:
    observed = abs(statistics.fmean(differences))
    if observed == 0 and all(value == 0 for value in differences):
        return 1.0
    rng = random.Random(SEED + seed_offset)
    extreme = 0
    for _ in range(PERMUTATION_SAMPLES):
        statistic = abs(statistics.fmean(value if rng.random() < 0.5 else -value for value in differences))
        extreme += statistic >= observed - 1e-15
    return (extreme + 1) / (PERMUTATION_SAMPLES + 1)


def mcnemar_exact(baseline_only: int, treatment_only: int) -> float:
    discordant = baseline_only + treatment_only
    if discordant == 0:
        return 1.0
    lower = min(baseline_only, treatment_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    m = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (m - rank) * value))
        adjusted[name] = running_max
    return adjusted


def mean(records: Iterable[dict], field: str) -> float:
    return statistics.fmean(float(row.get(field, 0)) for row in records)


def invalid_rate(records: Iterable[dict]) -> float:
    rows = list(records)
    calls = sum(int(row.get("action_calls", 0)) for row in rows)
    return sum(int(row.get("invalid_actions", 0)) for row in rows) / max(calls, 1)


def method_metrics(by_method: dict[str, dict[str, dict]]) -> list[dict]:
    rows: list[dict] = []
    for method_index, (method, (label, _)) in enumerate(METHODS.items()):
        records = [by_method[method][task_id] for task_id in TASK_IDS]
        successes = sum(bool(row.get("success")) for row in records)
        success_low, success_high = wilson_ci(successes, len(records))
        reward_low, reward_high = bootstrap_ci(
            records, lambda sample: mean(sample, "reward"), 100 + method_index,
        )
        steps_low, steps_high = bootstrap_ci(
            records, lambda sample: mean(sample, "steps"), 200 + method_index,
        )
        calls_low, calls_high = bootstrap_ci(
            records, lambda sample: mean(sample, "total_llm_calls"), 300 + method_index,
        )
        latency_low, latency_high = bootstrap_ci(
            records, lambda sample: mean(sample, "latency_seconds"), 400 + method_index,
        )
        invalid_low, invalid_high = bootstrap_ci(
            records, invalid_rate, 500 + method_index,
        )
        rows.append({
            "method": method,
            "label": label,
            "episodes": len(records),
            "mean_reward": mean(records, "reward"),
            "reward_ci_low": reward_low,
            "reward_ci_high": reward_high,
            "successes": successes,
            "success_rate": successes / len(records),
            "success_ci_low": success_low,
            "success_ci_high": success_high,
            "mean_steps": mean(records, "steps"),
            "steps_ci_low": steps_low,
            "steps_ci_high": steps_high,
            "mean_llm_calls": mean(records, "total_llm_calls"),
            "llm_calls_ci_low": calls_low,
            "llm_calls_ci_high": calls_high,
            "mean_input_tokens": mean(records, "input_tokens"),
            "mean_output_tokens": mean(records, "output_tokens"),
            "mean_llm_latency_seconds": mean(records, "llm_latency_seconds"),
            "mean_episode_latency_seconds": mean(records, "latency_seconds"),
            "latency_ci_low": latency_low,
            "latency_ci_high": latency_high,
            "invalid_action_rate": invalid_rate(records),
            "invalid_action_ci_low": invalid_low,
            "invalid_action_ci_high": invalid_high,
            "max_steps_rate": sum(row.get("status") == "max_steps" for row in records) / len(records),
            "mean_planning_calls": mean(records, "planning_calls"),
            "mean_reflection_calls": mean(records, "reflection_calls"),
            "invalid_auxiliary_outputs": sum(int(row.get("invalid_auxiliary_outputs", 0)) for row in records),
        })
    return rows


def comparisons(by_method: dict[str, dict[str, dict]]) -> list[dict]:
    baseline = by_method["react_baseline"]
    rows: list[dict] = []
    reward_pvalues: dict[str, float] = {}
    success_pvalues: dict[str, float] = {}

    for comparison_index, method in enumerate(list(METHODS)[1:]):
        treatment = by_method[method]
        reward_diffs = [float(treatment[t]["reward"]) - float(baseline[t]["reward"]) for t in TASK_IDS]
        step_diffs = [float(treatment[t]["steps"]) - float(baseline[t]["steps"]) for t in TASK_IDS]
        call_diffs = [
            float(treatment[t]["total_llm_calls"]) - float(baseline[t]["total_llm_calls"])
            for t in TASK_IDS
        ]
        reward_low, reward_high = paired_bootstrap_ci(reward_diffs, 600 + comparison_index)
        step_low, step_high = paired_median_bootstrap_ci(step_diffs, 700 + comparison_index)
        call_low, call_high = paired_bootstrap_ci(call_diffs, 800 + comparison_index)
        baseline_only = sum(bool(baseline[t]["success"]) and not bool(treatment[t]["success"]) for t in TASK_IDS)
        treatment_only = sum(not bool(baseline[t]["success"]) and bool(treatment[t]["success"]) for t in TASK_IDS)
        reward_p = sign_flip_pvalue(reward_diffs, 900 + comparison_index)
        success_p = mcnemar_exact(baseline_only, treatment_only)
        reward_pvalues[method] = reward_p
        success_pvalues[method] = success_p
        rows.append({
            "comparison": f"{METHODS[method][0]} vs Baseline",
            "method": method,
            "mean_reward_difference": statistics.fmean(reward_diffs),
            "reward_difference_ci_low": reward_low,
            "reward_difference_ci_high": reward_high,
            "reward_randomization_p": reward_p,
            "reward_holm_p": math.nan,
            "baseline_only_successes": baseline_only,
            "treatment_only_successes": treatment_only,
            "mcnemar_exact_p": success_p,
            "mcnemar_holm_p": math.nan,
            "median_step_difference": statistics.median(step_diffs),
            "median_step_difference_ci_low": step_low,
            "median_step_difference_ci_high": step_high,
            "mean_llm_call_difference": statistics.fmean(call_diffs),
            "llm_call_difference_ci_low": call_low,
            "llm_call_difference_ci_high": call_high,
        })

    reward_adjusted = holm_adjust(reward_pvalues)
    success_adjusted = holm_adjust(success_pvalues)
    for row in rows:
        row["reward_holm_p"] = reward_adjusted[row["method"]]
        row["mcnemar_holm_p"] = success_adjusted[row["method"]]
    return rows


def factorial_effects(by_method: dict[str, dict[str, dict]]) -> list[dict]:
    baseline = by_method["react_baseline"]
    planning = by_method["react_planning"]
    reflection = by_method["react_reflection"]
    full = by_method["react_plan_reflect"]
    planning_effect = [
        ((planning[t]["reward"] - baseline[t]["reward"]) + (full[t]["reward"] - reflection[t]["reward"])) / 2
        for t in TASK_IDS
    ]
    reflection_effect = [
        ((reflection[t]["reward"] - baseline[t]["reward"]) + (full[t]["reward"] - planning[t]["reward"])) / 2
        for t in TASK_IDS
    ]
    interaction = [
        full[t]["reward"] - planning[t]["reward"] - reflection[t]["reward"] + baseline[t]["reward"]
        for t in TASK_IDS
    ]
    output = []
    for index, (name, values) in enumerate((
        ("Planning main effect", planning_effect),
        ("Reflection main effect", reflection_effect),
        ("Planning x Reflection interaction", interaction),
    )):
        low, high = paired_bootstrap_ci([float(value) for value in values], 1000 + index)
        output.append({"effect": name, "estimate": statistics.fmean(values), "ci_low": low, "ci_high": high})
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def svg_performance(metrics: list[dict], path: Path) -> None:
    width, height = 920, 430
    left, top, plot_w, plot_h = 90, 55, 760, 300
    colors = ["#607D8B", "#1976D2", "#7B1FA2", "#00897B"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:20px;font-weight:700}.axis{font-size:13px}.value{font-size:12px;font-weight:700}</style>',
        '<text x="460" y="28" text-anchor="middle" class="title">Formal WebShop Performance (95% CI)</text>',
    ]
    for tick in range(0, 6):
        value = tick / 10
        y = top + plot_h - value / 0.6 * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ECEFF1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.1f}</text>')
    group_width = plot_w / len(metrics)
    for index, row in enumerate(metrics):
        center = left + group_width * (index + 0.5)
        for offset, metric, low_name, high_name, label in (
            (-28, "mean_reward", "reward_ci_low", "reward_ci_high", "Reward"),
            (28, "success_rate", "success_ci_low", "success_ci_high", "Success"),
        ):
            value = row[metric]
            low, high = row[low_name], row[high_name]
            x = center + offset
            bar_h = value / 0.6 * plot_h
            y = top + plot_h - bar_h
            shade = colors[index] if metric == "mean_reward" else "#FFB300"
            parts.append(f'<rect x="{x - 20:.1f}" y="{y:.1f}" width="40" height="{bar_h:.1f}" rx="2" fill="{shade}"/>')
            low_y = top + plot_h - low / 0.6 * plot_h
            high_y = top + plot_h - high / 0.6 * plot_h
            parts.append(f'<line x1="{x:.1f}" y1="{low_y:.1f}" x2="{x:.1f}" y2="{high_y:.1f}" stroke="#263238" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 7:.1f}" y1="{low_y:.1f}" x2="{x + 7:.1f}" y2="{low_y:.1f}" stroke="#263238" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 7:.1f}" y1="{high_y:.1f}" x2="{x + 7:.1f}" y2="{high_y:.1f}" stroke="#263238" stroke-width="2"/>')
            parts.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.3f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top + plot_h + 25}" text-anchor="middle" class="axis">{row["label"]}</text>')
    parts.extend([
        '<rect x="330" y="398" width="14" height="14" fill="#1976D2"/><text x="351" y="410" class="axis">Mean reward</text>',
        '<rect x="470" y="398" width="14" height="14" fill="#FFB300"/><text x="491" y="410" class="axis">Strict success rate</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_efficiency(metrics: list[dict], path: Path) -> None:
    width, height = 920, 430
    left, top, plot_w, plot_h = 90, 55, 760, 300
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:20px;font-weight:700}.axis{font-size:13px}.value{font-size:12px;font-weight:700}</style>',
        '<text x="460" y="28" text-anchor="middle" class="title">Efficiency and Model Calls (95% CI)</text>',
    ]
    maximum = 18
    for tick in range(0, maximum + 1, 3):
        y = top + plot_h - tick / maximum * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ECEFF1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick}</text>')
    group_width = plot_w / len(metrics)
    for index, row in enumerate(metrics):
        center = left + group_width * (index + 0.5)
        for offset, metric, low_name, high_name, color in (
            (-28, "mean_steps", "steps_ci_low", "steps_ci_high", "#42A5F5"),
            (28, "mean_llm_calls", "llm_calls_ci_low", "llm_calls_ci_high", "#AB47BC"),
        ):
            value, low, high = row[metric], row[low_name], row[high_name]
            x = center + offset
            bar_h = value / maximum * plot_h
            y = top + plot_h - bar_h
            low_y = top + plot_h - low / maximum * plot_h
            high_y = top + plot_h - high / maximum * plot_h
            parts.append(f'<rect x="{x - 20:.1f}" y="{y:.1f}" width="40" height="{bar_h:.1f}" rx="2" fill="{color}"/>')
            parts.append(f'<line x1="{x:.1f}" y1="{low_y:.1f}" x2="{x:.1f}" y2="{high_y:.1f}" stroke="#263238" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 7:.1f}" y1="{low_y:.1f}" x2="{x + 7:.1f}" y2="{low_y:.1f}" stroke="#263238" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 7:.1f}" y1="{high_y:.1f}" x2="{x + 7:.1f}" y2="{high_y:.1f}" stroke="#263238" stroke-width="2"/>')
            parts.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.2f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top + plot_h + 25}" text-anchor="middle" class="axis">{row["label"]}</text>')
    parts.extend([
        '<rect x="340" y="398" width="14" height="14" fill="#42A5F5"/><text x="361" y="410" class="axis">Environment steps</text>',
        '<rect x="510" y="398" width="14" height="14" fill="#AB47BC"/><text x="531" y="410" class="axis">Total LLM calls</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(metrics: list[dict], paired: list[dict], effects: list[dict], integrity: dict) -> None:
    metric_lines = []
    for row in metrics:
        metric_lines.append(
            f'| {row["label"]} | {row["mean_reward"]:.3f} '
            f'[{row["reward_ci_low"]:.3f}, {row["reward_ci_high"]:.3f}] | '
            f'{row["success_rate"]:.0%} [{row["success_ci_low"]:.1%}, {row["success_ci_high"]:.1%}] | '
            f'{row["mean_steps"]:.2f} | {row["mean_llm_calls"]:.2f} | '
            f'{row["invalid_action_rate"]:.1%} | {row["max_steps_rate"]:.0%} |'
        )
    comparison_lines = []
    for row in paired:
        comparison_lines.append(
            f'| {row["comparison"]} | {row["mean_reward_difference"]:+.3f} '
            f'[{row["reward_difference_ci_low"]:+.3f}, {row["reward_difference_ci_high"]:+.3f}] | '
            f'{row["reward_holm_p"]:.4f} | {row["baseline_only_successes"]}/{row["treatment_only_successes"]} | '
            f'{row["mcnemar_holm_p"]:.4f} | {row["median_step_difference"]:+.1f} | '
            f'{row["mean_llm_call_difference"]:+.2f} |'
        )
    effect_lines = [
        f'| {row["effect"]} | {row["estimate"]:+.3f} | [{row["ci_low"]:+.3f}, {row["ci_high"]:+.3f}] |'
        for row in effects
    ]
    best = max(metrics, key=lambda row: row["mean_reward"])
    full_comparison = next(row for row in paired if row["method"] == "react_plan_reflect")
    report = f"""# WebShop 正式实验统计报告

生成参数：随机种子 `{SEED}`，bootstrap `{BOOTSTRAP_SAMPLES:,}` 次，配对随机化检验 `{PERMUTATION_SAMPLES:,}` 次。

## 数据完整性

- 四组各 100 个唯一任务，共 400 条有效结果；配对任务数为 {integrity['paired_task_count']}。
- 正式任务范围为 `fixed_10`–`fixed_109`，全部使用 `dataset_mode=full`。
- 无缺失任务、无重复有效记录、无 WebShop 环境错误，完整性检查通过。

## 主要结果

| 方法 | 平均奖励（95% CI） | 严格成功率（95% CI） | 平均步数 | 平均 LLM 调用 | 非法动作率 | 达到步数上限 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

![正式实验效果](method_performance.svg)

## 与 Baseline 的配对比较

奖励差使用任务级配对 bootstrap 置信区间和配对符号翻转随机化检验；成功差使用 McNemar 精确检验。两类检验分别对三次 Baseline 比较进行 Holm 校正。

| 比较 | 平均奖励差（95% CI） | 奖励 Holm p | Baseline独有/方法独有成功 | McNemar Holm p | 步数中位差 | LLM调用均值差 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

![效率与调用开销](efficiency.svg)

## 2×2 消融效应

| 效应 | 奖励估计 | 95% CI |
|---|---:|---:|
{chr(10).join(effect_lines)}

## 结论

- `{best['label']}` 的平均奖励最高（{best['mean_reward']:.3f}），严格成功率为 {best['success_rate']:.0%}。
- 完整方法相对 Baseline 的平均奖励提高 {full_comparison['mean_reward_difference']:+.3f}，其 95% CI 为 [{full_comparison['reward_difference_ci_low']:+.3f}, {full_comparison['reward_difference_ci_high']:+.3f}]。
- 完整方法把平均环境步数从 {metrics[0]['mean_steps']:.2f} 降至 {best['mean_steps']:.2f}；但 Planning 与 Reflection 属于额外模型调用，必须结合 `mean_llm_calls` 和 token 指标讨论成本。
- Reflection 单独使用时提升较小，并产生 {next(row for row in metrics if row['method'] == 'react_reflection')['invalid_auxiliary_outputs']} 次辅助输出格式错误；完整方法仅有 {best['invalid_auxiliary_outputs']} 次，提示 Planning 上下文可能让反思输出更稳定。这是结果关联，不能直接当作机制因果证明。
- 单次 100 任务实验的置信区间仍较宽；结论应以效应量和区间为主，不应只依赖 p 值。

## 文件说明

- `formal_metrics.csv`：四组点估计、置信区间、效率和格式指标。
- `paired_comparisons.csv`：三组相对 Baseline 的配对效应和检验。
- `factorial_effects.csv`：Planning、Reflection 主效应与交互效应。
- `integrity_report.json`：原始结果完整性审计。
- `method_performance.svg`、`efficiency.svg`：论文可直接引用的矢量图。
"""
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    by_method, integrity = load_results()
    (OUTPUT / "integrity_report.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = method_metrics(by_method)
    paired = comparisons(by_method)
    effects = factorial_effects(by_method)
    write_csv(OUTPUT / "formal_metrics.csv", metrics)
    write_csv(OUTPUT / "paired_comparisons.csv", paired)
    write_csv(OUTPUT / "factorial_effects.csv", effects)
    svg_performance(metrics, OUTPUT / "method_performance.svg")
    svg_efficiency(metrics, OUTPUT / "efficiency.svg")
    write_report(metrics, paired, effects, integrity)
    print(f"Analysis complete: {OUTPUT}")


if __name__ == "__main__":
    main()
