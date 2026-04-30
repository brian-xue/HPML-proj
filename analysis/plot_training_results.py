from __future__ import annotations

import ast
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for user envs
    raise SystemExit(
        "matplotlib is required to generate plots. Install it with `pip install matplotlib` "
        "in the environment you use to run this script."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
PLOTS_DIR = REPO_ROOT / "analysis" / "plots"
LEGACY_FILES = [
    REPO_ROOT / "final_ddp_1.json",
    REPO_ROOT / "final_ddp_2.json",
    REPO_ROOT / "final_ddp_4.json",
    REPO_ROOT / "final_fsdp_2.json",
    REPO_ROOT / "final_fsdp_4.json",
]
NSYS_SQLITES = {
    "ddp": OUTPUT_DIR / "nsys_ddp_4gpu.sqlite",
    "fsdp": OUTPUT_DIR / "nsys_fsdp_4gpu.sqlite",
}
NSYS_SQLITES_PEFT = {
    "ddp": OUTPUT_DIR / "nsys_ddp_4gpu_peft.sqlite",
    "fsdp": OUTPUT_DIR / "nsys_fsdp_4gpu_peft.sqlite",
}

METHOD_COLORS = {"ddp": "#1f77b4", "fsdp": "#ff7f0e"}
PHASE_COLORS = {
    "backward": "#4c78a8",
    "forward_loss": "#72b7b2",
    "optimizer_step": "#f58518",
}
COMM_COLORS = {
    "allreduce": "#e45756",
    "broadcast": "#f2cf5b",
    "reduce_scatter": "#54a24b",
    "all_gather": "#b279a2",
}
NEUTRAL_GRID = "#d9d9d9"


@dataclass
class ResultRecord:
    source: str
    experiment: str
    method: str
    gpus: int
    label: str
    version: str
    train_steps: int | None
    train_loss: float | None
    eval_accuracy: float | None
    step_time_s: float | None
    total_runtime_s: float | None
    tokens_per_s: float | None
    peak_mb: float | None
    alloc_mb: float | None
    reserved_mb: float | None
    peft_enabled: bool | None


@dataclass
class TtaPoint:
    method: str
    gpus: int
    version: str
    step: int
    accuracy: float
    eval_runtime_s: float
    eval_tokens_per_s: float


def ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    ensure_plots_dir()
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_experiment_name(experiment: str) -> tuple[str, int, str] | None:
    match = re.match(r"^(ddp|fsdp)_scaling_(\d)gpu(?:_(.+))?$", experiment)
    if not match:
        return None
    method = match.group(1)
    gpus = int(match.group(2))
    label = match.group(3) or "base"
    return method, gpus, label


def load_output_results() -> list[ResultRecord]:
    records: list[ResultRecord] = []
    for path in sorted(OUTPUT_DIR.glob("*/v*/final_results.json")):
        parsed = parse_experiment_name(path.parent.parent.name)
        if not parsed:
            continue
        method, gpus, label = parsed
        data = json.loads(path.read_text())
        runtime = data.get("runtime", {}) or {}
        train = data.get("train", {}) or {}
        eval_metrics = (data.get("eval", {}) or {}).get("metrics", {}) or {}
        peft = data.get("peft", {}) or {}
        records.append(
            ResultRecord(
                source="output",
                experiment=path.parent.parent.name,
                method=method,
                gpus=gpus,
                label=label,
                version=path.parent.name,
                train_steps=train.get("train_steps"),
                train_loss=train.get("train_loss"),
                eval_accuracy=eval_metrics.get("accuracy"),
                step_time_s=runtime.get("average_step_time_seconds"),
                total_runtime_s=runtime.get("global_total_runtime_seconds", runtime.get("total_runtime_seconds")),
                tokens_per_s=runtime.get("global_tokens_per_second", runtime.get("tokens_per_second")),
                peak_mb=runtime.get(
                    "global_peak_gpu_memory_allocated_mb_max_rank",
                    runtime.get("gpu_peak_memory_allocated_mb"),
                ),
                alloc_mb=runtime.get("gpu_memory_allocated_mb"),
                reserved_mb=runtime.get("gpu_memory_reserved_mb"),
                peft_enabled=peft.get("enabled"),
            )
        )
    return records


def load_legacy_peft_results() -> list[ResultRecord]:
    records: list[ResultRecord] = []
    for path in LEGACY_FILES:
        if not path.exists():
            continue
        match = re.match(r"^final_(ddp|fsdp)_(\d)\.json$", path.name)
        if not match:
            continue
        method = match.group(1)
        gpus = int(match.group(2))
        data = json.loads(path.read_text())
        runtime = data.get("runtime", {}) or {}
        train = data.get("train", {}) or {}
        peft = data.get("peft", {}) or {}
        records.append(
            ResultRecord(
                source="legacy_peft",
                experiment=path.stem,
                method=method,
                gpus=gpus,
                label="legacy_peft",
                version="legacy",
                train_steps=train.get("train_steps"),
                train_loss=train.get("train_loss"),
                eval_accuracy=None,
                step_time_s=runtime.get("average_step_time_seconds"),
                total_runtime_s=runtime.get("global_total_runtime_seconds", runtime.get("total_runtime_seconds")),
                tokens_per_s=runtime.get("global_tokens_per_second", runtime.get("tokens_per_second")),
                peak_mb=runtime.get(
                    "global_peak_gpu_memory_allocated_mb_max_rank",
                    runtime.get("gpu_peak_memory_allocated_mb"),
                ),
                alloc_mb=runtime.get("gpu_memory_allocated_mb"),
                reserved_mb=runtime.get("gpu_memory_reserved_mb"),
                peft_enabled=peft.get("enabled"),
            )
        )
    return records


def load_tta_points() -> list[TtaPoint]:
    points: list[TtaPoint] = []
    pattern = re.compile(r"step=(\d+) eval=(\{.*\})")
    for run_log in OUTPUT_DIR.glob("*/v*/run.log"):
        parsed = parse_experiment_name(run_log.parent.parent.name)
        if not parsed:
            continue
        method, gpus, _label = parsed
        version = run_log.parent.name
        for line in run_log.read_text().splitlines():
            match = pattern.search(line)
            if not match:
                continue
            step = int(match.group(1))
            try:
                payload = ast.literal_eval(match.group(2))
            except (ValueError, SyntaxError):
                continue
            if "accuracy" not in payload:
                continue
            points.append(
                TtaPoint(
                    method=method,
                    gpus=gpus,
                    version=version,
                    step=step,
                    accuracy=float(payload["accuracy"]),
                    eval_runtime_s=float(payload.get("total_runtime_seconds", 0.0)),
                    eval_tokens_per_s=float(payload.get("tokens_per_second", 0.0)),
                )
            )
    return sorted(points, key=lambda point: (point.method, point.gpus, point.version, point.step))


def query_one_value(conn: sqlite3.Connection, query: str, args: tuple[Any, ...] = ()) -> float:
    row = conn.execute(query, args).fetchone()
    if row is None or row[0] is None:
        return 0.0
    return float(row[0])


def query_rows(conn: sqlite3.Connection, query: str, args: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    return list(conn.execute(query, args).fetchall())


def load_profiler_metrics() -> dict[str, dict[str, Any]]:
    return load_profiler_metrics_from_paths(NSYS_SQLITES)


def load_profiler_metrics_from_paths(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for method, path in paths.items():
        if not path.exists():
            continue
        conn = sqlite3.connect(path)
        total_kernel_s = query_one_value(conn, "SELECT SUM(end - start) / 1e9 FROM CUPTI_ACTIVITY_KIND_KERNEL")
        nccl_rows = query_rows(
            conn,
            """
            SELECT s.value, SUM(k.end - k.start) / 1e9 AS total_s
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            WHERE s.value LIKE 'ncclDevKernel%'
            GROUP BY s.value
            ORDER BY total_s DESC
            """,
        )
        d2d_row = query_rows(
            conn,
            """
            SELECT
                SUM(m.bytes) / 1024.0 / 1024.0 AS total_mb,
                COUNT(*) AS copies,
                AVG(m.bytes) / 1024.0 / 1024.0 AS avg_mb
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            JOIN ENUM_CUDA_MEM_KIND sk ON m.srcKind = sk.id
            JOIN ENUM_CUDA_MEM_KIND dk ON m.dstKind = dk.id
            WHERE sk.label = 'Device' AND dk.label = 'Device'
            """,
        )
        phase_rows = query_rows(
            conn,
            """
            SELECT COALESCE(n.text, s.value) AS phase, SUM(n.end - n.start) / 1e9 AS total_s
            FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON n.textId = s.id
            WHERE n.end IS NOT NULL
              AND COALESCE(n.text, s.value) IN ('forward_loss', 'backward', 'optimizer_step')
            GROUP BY COALESCE(n.text, s.value)
            """,
        )
        phase_totals = {str(name): float(total_s) for name, total_s in phase_rows}
        phase_sum = sum(phase_totals.values()) or 1.0
        metrics[method] = {
            "total_kernel_s": total_kernel_s,
            "nccl_breakdown_s": {str(name): float(total_s) for name, total_s in nccl_rows},
            "nccl_total_s": sum(float(total_s) for _name, total_s in nccl_rows),
            "d2d_total_mb": float(d2d_row[0][0]) if d2d_row and d2d_row[0][0] is not None else 0.0,
            "d2d_copies": int(d2d_row[0][1]) if d2d_row else 0,
            "d2d_avg_mb": float(d2d_row[0][2]) if d2d_row and d2d_row[0][2] is not None else 0.0,
            "phase_totals_s": phase_totals,
            "phase_pcts": {name: 100.0 * value / phase_sum for name, value in phase_totals.items()},
        }
        conn.close()
    return metrics


def summarize_by_gpu(records: list[ResultRecord], label_prefix: str) -> dict[str, dict[int, list[ResultRecord]]]:
    grouped: dict[str, dict[int, list[ResultRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.label.startswith(label_prefix):
            grouped[record.method][record.gpus].append(record)
    return grouped


def apply_report_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "font.size": 10,
            "legend.frameon": False,
            "grid.color": NEUTRAL_GRID,
            "grid.alpha": 0.5,
        }
    )


def metric_mean(records: list[ResultRecord], field: str) -> float:
    values = [getattr(record, field) for record in records if getattr(record, field) is not None]
    return mean(values) if values else 0.0


def annotate_bars(ax: plt.Axes, bars: Any, fmt: str, percent: bool = False) -> None:
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        label = fmt.format(height)
        if percent:
            label += "%"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2.0, height),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )


def styled_axis(ax: plt.Axes, y_label: str, title: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y")


def plot_scaling_step_time(records: list[ResultRecord]) -> Path | None:
    grouped = summarize_by_gpu(records, "scale_r")
    if not grouped:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ("ddp", "fsdp"):
        xs = sorted(grouped.get(method, {}).keys())
        ys = [mean(record.step_time_s for record in grouped[method][gpu] if record.step_time_s is not None) for gpu in xs]
        yerr = [
            pstdev([record.step_time_s for record in grouped[method][gpu] if record.step_time_s is not None])
            if len(grouped[method][gpu]) > 1
            else 0.0
            for gpu in xs
        ]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=4, label=method.upper(), color=METHOD_COLORS[method])
    ax.set_title("Full Fine-Tuning Scaling: Average Step Time")
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Average Step Time (s)")
    ax.set_xticks([1, 2, 4])
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, "scaling_step_time.png")


def plot_scaling_tokens_per_second(records: list[ResultRecord]) -> Path | None:
    grouped = summarize_by_gpu(records, "scale_r")
    if not grouped:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ("ddp", "fsdp"):
        xs = sorted(grouped.get(method, {}).keys())
        ys = [mean(record.tokens_per_s for record in grouped[method][gpu] if record.tokens_per_s is not None) for gpu in xs]
        yerr = [
            pstdev([record.tokens_per_s for record in grouped[method][gpu] if record.tokens_per_s is not None])
            if len(grouped[method][gpu]) > 1
            else 0.0
            for gpu in xs
        ]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=4, label=method.upper(), color=METHOD_COLORS[method])
    ax.set_title("Full Fine-Tuning Scaling: Global Tokens per Second")
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Global Tokens / s")
    ax.set_xticks([1, 2, 4])
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, "scaling_tokens_per_second.png")


def plot_scaling_memory(records: list[ResultRecord]) -> Path | None:
    grouped = summarize_by_gpu(records, "scale_r")
    if not grouped:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for method in ("ddp", "fsdp"):
        xs = sorted(grouped.get(method, {}).keys())
        peak_vals = [mean(record.peak_mb for record in grouped[method][gpu] if record.peak_mb is not None) for gpu in xs]
        alloc_vals = [mean(record.alloc_mb for record in grouped[method][gpu] if record.alloc_mb is not None) for gpu in xs]
        axes[0].plot(xs, peak_vals, marker="o", linewidth=2, label=method.upper(), color=METHOD_COLORS[method])
        axes[1].plot(xs, alloc_vals, marker="o", linewidth=2, label=method.upper(), color=METHOD_COLORS[method])
    axes[0].set_title("Peak Allocated GPU Memory")
    axes[0].set_ylabel("Memory (MB)")
    axes[1].set_title("Current Allocated GPU Memory")
    for ax in axes:
        ax.set_xlabel("Number of GPUs")
        ax.set_xticks([1, 2, 4])
        ax.grid(alpha=0.25)
        ax.legend()
    return save_figure(fig, "scaling_memory.png")


def plot_scaling_train_loss(records: list[ResultRecord]) -> Path | None:
    grouped = summarize_by_gpu(records, "scale_r")
    if not grouped:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ("ddp", "fsdp"):
        xs = sorted(grouped.get(method, {}).keys())
        ys = [mean(record.train_loss for record in grouped[method][gpu] if record.train_loss is not None) for gpu in xs]
        ax.plot(xs, ys, marker="o", linewidth=2, label=method.upper(), color=METHOD_COLORS[method])
    ax.set_title("Final Train Loss Across GPU Counts")
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Train Loss")
    ax.set_xticks([1, 2, 4])
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, "scaling_train_loss.png")


def plot_repeat_stability(records: list[ResultRecord]) -> Path | None:
    grouped = summarize_by_gpu(records, "scale_r")
    rows: list[tuple[str, int, float, float]] = []
    for method, method_groups in grouped.items():
        for gpu, gpu_records in sorted(method_groups.items()):
            tok_vals = [record.tokens_per_s for record in gpu_records if record.tokens_per_s is not None]
            step_vals = [record.step_time_s for record in gpu_records if record.step_time_s is not None]
            if not tok_vals or not step_vals:
                continue
            tok_cv = 100.0 * pstdev(tok_vals) / mean(tok_vals) if len(tok_vals) > 1 else 0.0
            step_cv = 100.0 * pstdev(step_vals) / mean(step_vals) if len(step_vals) > 1 else 0.0
            rows.append((method, gpu, tok_cv, step_cv))
    if not rows:
        return None
    labels = [f"{method.upper()}-{gpu}G" for method, gpu, _tok_cv, _step_cv in rows]
    tok_cvs = [tok_cv for _method, _gpu, tok_cv, _step_cv in rows]
    step_cvs = [step_cv for _method, _gpu, _tok_cv, step_cv in rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([value - 0.2 for value in x], tok_cvs, width=0.4, label="Tokens/s CV %", color="#59a14f")
    ax.bar([value + 0.2 for value in x], step_cvs, width=0.4, label="Step-time CV %", color="#e15759")
    ax.set_title("Repeat Stability Across Scaling Runs")
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return save_figure(fig, "repeat_stability.png")


def plot_memory_stress_peak_memory(records: list[ResultRecord]) -> Path | None:
    mem_records = [record for record in records if record.label.startswith("memlen")]
    if not mem_records:
        return None
    grouped: dict[str, dict[int, ResultRecord]] = defaultdict(dict)
    for record in mem_records:
        length = int(record.label.replace("memlen", ""))
        grouped[record.method][length] = record
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ("ddp", "fsdp"):
        lengths = sorted(grouped.get(method, {}).keys())
        peaks = [grouped[method][length].peak_mb for length in lengths]
        ax.plot(lengths, peaks, marker="o", linewidth=2, label=method.upper(), color=METHOD_COLORS[method])
    ax.set_title("Memory Stress at 4 GPUs: Peak Memory vs Sequence Length")
    ax.set_xlabel("Max Sequence Length")
    ax.set_ylabel("Peak Allocated GPU Memory (MB)")
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, "memory_stress_peak_memory.png")


def plot_memory_stress_step_time(records: list[ResultRecord]) -> Path | None:
    mem_records = [record for record in records if record.label.startswith("memlen")]
    if not mem_records:
        return None
    grouped: dict[str, dict[int, ResultRecord]] = defaultdict(dict)
    for record in mem_records:
        length = int(record.label.replace("memlen", ""))
        grouped[record.method][length] = record
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ("ddp", "fsdp"):
        lengths = sorted(grouped.get(method, {}).keys())
        times = [grouped[method][length].step_time_s for length in lengths]
        ax.plot(lengths, times, marker="o", linewidth=2, label=method.upper(), color=METHOD_COLORS[method])
    ax.set_title("Memory Stress at 4 GPUs: Step Time vs Sequence Length")
    ax.set_xlabel("Max Sequence Length")
    ax.set_ylabel("Average Step Time (s)")
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, "memory_stress_step_time.png")


def plot_speed_memory_tradeoff(records: list[ResultRecord]) -> Path | None:
    candidates = [
        record for record in records if record.label.startswith("scale_r") or record.label.startswith("memlen")
    ]
    if not candidates:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for record in candidates:
        marker = "o" if record.label.startswith("scale_r") else "s"
        label = f"{record.method.upper()} {record.gpus}G {record.label}"
        ax.scatter(record.peak_mb, record.step_time_s, s=90, marker=marker, color=METHOD_COLORS[record.method], alpha=0.8)
        ax.annotate(label, (record.peak_mb, record.step_time_s), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Speed vs Memory Tradeoff Across Completed Runs")
    ax.set_xlabel("Peak Allocated GPU Memory (MB)")
    ax.set_ylabel("Average Step Time (s)")
    ax.grid(alpha=0.25)
    return save_figure(fig, "speed_memory_tradeoff.png")


def plot_peft_vs_full_finetune(records: list[ResultRecord], legacy_records: list[ResultRecord]) -> Path | None:
    """
    Slide title:
    Slide 2: FT vs PEFT Peak Memory

    Caption:
    This is the reference chart for absolute peak-memory values. Within each GPU
    count, Full FT and PEFT are shown as separate DDP/FSDP pairs.

    Speaker notes:
    Use this as a grounding chart, not the headline chart. It shows the absolute
    memory picture behind the regime-dependence result.
    """
    full_records = [record for record in records if record.label.startswith("scale_r") and record.gpus in (1, 2, 4)]
    peft_records = [record for record in records if record.label == "peft" and record.gpus in (1, 2, 4)]
    if not full_records:
        return None
    full_by_method_gpu: dict[tuple[str, int], list[ResultRecord]] = defaultdict(list)
    for record in full_records:
        full_by_method_gpu[(record.method, record.gpus)].append(record)

    peft_by_method_gpu: dict[tuple[str, int], ResultRecord] = {}
    for record in peft_records:
        peft_by_method_gpu[(record.method, record.gpus)] = record
    for record in legacy_records:
        if (record.method, record.gpus) not in peft_by_method_gpu:
            peft_by_method_gpu[(record.method, record.gpus)] = record

    if not peft_by_method_gpu:
        return None

    gpu_groups = [
        gpu
        for gpu in (1, 2, 4)
        if any((method, gpu) in peft_by_method_gpu or (method, gpu) in full_by_method_gpu for method in ("ddp", "fsdp"))
    ]
    if not gpu_groups:
        return None

    labels = ["DDP FT", "FSDP FT", "DDP PEFT", "FSDP PEFT"]
    offsets = [-0.34, -0.12, 0.12, 0.34]
    bar_width = 0.20
    style_colors = {
        "DDP FT": METHOD_COLORS["ddp"],
        "FSDP FT": METHOD_COLORS["fsdp"],
        "DDP PEFT": METHOD_COLORS["ddp"],
        "FSDP PEFT": METHOD_COLORS["fsdp"],
    }
    peak_series: dict[str, list[float]] = {label: [] for label in labels}

    for gpu in gpu_groups:
        ddp_ft = full_by_method_gpu.get(("ddp", gpu), [])
        fsdp_ft = full_by_method_gpu.get(("fsdp", gpu), [])
        ddp_peft = peft_by_method_gpu.get(("ddp", gpu))
        fsdp_peft = peft_by_method_gpu.get(("fsdp", gpu))

        peak_series["DDP FT"].append(mean(record.peak_mb for record in ddp_ft if record.peak_mb is not None) if ddp_ft else 0.0)
        peak_series["FSDP FT"].append(mean(record.peak_mb for record in fsdp_ft if record.peak_mb is not None) if fsdp_ft else 0.0)
        peak_series["DDP PEFT"].append(ddp_peft.peak_mb if ddp_peft and ddp_peft.peak_mb is not None else 0.0)
        peak_series["FSDP PEFT"].append(fsdp_peft.peak_mb if fsdp_peft and fsdp_peft.peak_mb is not None else 0.0)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    x = list(range(len(gpu_groups)))
    for idx, label in enumerate(labels):
        shifted = [value + offsets[idx] for value in x]
        hatch = "" if "FT" in label and "PEFT" not in label else "//"
        bars = ax.bar(shifted, peak_series[label], width=bar_width, label=label, color=style_colors[label], alpha=0.9, hatch=hatch)
        annotate_bars(ax, bars, "{:.0f}")
    ax.set_title("Peak Memory: Full FT Pair and PEFT Pair at Each GPU Count")
    ax.set_ylabel("Peak Allocated GPU Memory (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{gpu} GPU" if gpu == 1 else f"{gpu} GPUs" for gpu in gpu_groups])
    ax.grid(axis="y")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4)
    fig.text(
        0.5,
        -0.03,
        "Solid bars are Full FT; hatched bars are PEFT. Read within each GPU count as FT pair, then PEFT pair.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    return save_figure(fig, "slide2_peft_vs_full_finetune.png")


def plot_scaling_summary(records: list[ResultRecord]) -> Path | None:
    """
    Slide title:
    Slide 1: FT DDP vs FSDP Scaling Summary

    Caption:
    This chart captures the core full-FT story: loss stays aligned across GPU counts,
    peak memory scales very differently, and the 4-GPU live-memory view makes FSDP's
    steady-state memory advantage explicit.

    Speaker notes:
    Use this chart to separate "which method?" from "how does it scale?".
    The important read is that optimization outcome stays similar while memory
        and memory scaling differ, especially in full fine-tuning.
    """
    grouped = summarize_by_gpu(records, "scale_r")
    if not grouped:
        return None
    gpus = [gpu for gpu in (1, 2, 4) if any(gpu in grouped.get(method, {}) for method in ("ddp", "fsdp"))]
    if not gpus:
        return None

    x = list(gpus)
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    ddp_peak = [metric_mean(grouped.get("ddp", {}).get(gpu, []), "peak_mb") for gpu in gpus]
    fsdp_peak = [metric_mean(grouped.get("fsdp", {}).get(gpu, []), "peak_mb") for gpu in gpus]
    ddp_loss = [metric_mean(grouped.get("ddp", {}).get(gpu, []), "train_loss") for gpu in gpus]
    fsdp_loss = [metric_mean(grouped.get("fsdp", {}).get(gpu, []), "train_loss") for gpu in gpus]
    ddp_live_4g = metric_mean(grouped.get("ddp", {}).get(4, []), "alloc_mb")
    fsdp_live_4g = metric_mean(grouped.get("fsdp", {}).get(4, []), "alloc_mb")

    idx = list(range(len(gpus)))
    ddp_loss_bars = axes[0].bar([value - width / 2 for value in idx], ddp_loss, width=width, color=METHOD_COLORS["ddp"], label="DDP")
    fsdp_loss_bars = axes[0].bar([value + width / 2 for value in idx], fsdp_loss, width=width, color=METHOD_COLORS["fsdp"], label="FSDP")
    styled_axis(axes[0], "Final Train Loss", "Optimization Outcome")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels([str(gpu) for gpu in gpus])
    axes[0].set_xlabel("GPUs")
    annotate_bars(axes[0], ddp_loss_bars, "{:.3f}")
    annotate_bars(axes[0], fsdp_loss_bars, "{:.3f}")

    ddp_bars = axes[1].bar([value - width / 2 for value in idx], ddp_peak, width=width, color=METHOD_COLORS["ddp"], label="DDP")
    fsdp_bars = axes[1].bar([value + width / 2 for value in idx], fsdp_peak, width=width, color=METHOD_COLORS["fsdp"], label="FSDP")
    styled_axis(axes[1], "Peak GPU Memory (MB)", "Per-Device Peak Memory")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels([str(gpu) for gpu in gpus])
    axes[1].set_xlabel("GPUs")
    annotate_bars(axes[1], ddp_bars, "{:.0f}")
    annotate_bars(axes[1], fsdp_bars, "{:.0f}")

    live_vals = [ddp_live_4g, fsdp_live_4g]
    live_bars = axes[2].bar(["DDP", "FSDP"], live_vals, color=[METHOD_COLORS["ddp"], METHOD_COLORS["fsdp"]], width=0.6)
    styled_axis(axes[2], "Allocated GPU Memory (MB)", "4-GPU Live Memory")
    annotate_bars(axes[2], live_bars, "{:.0f}")

    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
    fig.suptitle("Full Fine-Tuning: DDP vs FSDP Across GPU Counts", y=1.03, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.03,
        "Loss stays aligned, while peak and live memory expose the main systems advantage of FSDP in full fine-tuning.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    return save_figure(fig, "slide1_scaling_summary.png")


def plot_slide1_final(records: list[ResultRecord], profiler_metrics: dict[str, dict[str, Any]]) -> Path | None:
    """
    Slide title:
    Slide 1 Final: FT DDP vs FSDP

    Caption:
    Full fine-tuning shows the main DDP-vs-FSDP result: similar training outcome,
    but sharply different memory behavior and communication burden.

    Speaker notes:
    Read left-to-right. First, peak memory across GPU counts shows the scaling
    difference. Second, 4-GPU live memory shows the steady-state footprint gap.
    Third, communication burden explains why FSDP behaves differently.
    """
    grouped = summarize_by_gpu(records, "scale_r")
    if not grouped or not profiler_metrics:
        return None
    gpus = [gpu for gpu in (1, 2, 4) if any(gpu in grouped.get(method, {}) for method in ("ddp", "fsdp"))]
    if not gpus:
        return None

    ddp_peak = [metric_mean(grouped.get("ddp", {}).get(gpu, []), "peak_mb") for gpu in gpus]
    fsdp_peak = [metric_mean(grouped.get("fsdp", {}).get(gpu, []), "peak_mb") for gpu in gpus]
    ddp_live_4g = metric_mean(grouped.get("ddp", {}).get(4, []), "alloc_mb")
    fsdp_live_4g = metric_mean(grouped.get("fsdp", {}).get(4, []), "alloc_mb")

    methods = [method for method in ("ddp", "fsdp") if method in profiler_metrics]
    comm_x = list(range(len(methods)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    idx = list(range(len(gpus)))
    width = 0.34
    ddp_peak_bars = axes[0].bar([value - width / 2 for value in idx], ddp_peak, width=width, color=METHOD_COLORS["ddp"], label="DDP")
    fsdp_peak_bars = axes[0].bar([value + width / 2 for value in idx], fsdp_peak, width=width, color=METHOD_COLORS["fsdp"], label="FSDP")
    styled_axis(axes[0], "Peak GPU Memory (MB)", "Per-Device Peak Memory")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels([str(gpu) for gpu in gpus])
    axes[0].set_xlabel("GPUs")
    annotate_bars(axes[0], ddp_peak_bars, "{:.0f}")
    annotate_bars(axes[0], fsdp_peak_bars, "{:.0f}")

    live_vals = [ddp_live_4g, fsdp_live_4g]
    live_bars = axes[1].bar(["DDP", "FSDP"], live_vals, color=[METHOD_COLORS["ddp"], METHOD_COLORS["fsdp"]], width=0.6)
    styled_axis(axes[1], "Allocated GPU Memory (MB)", "4-GPU Live Memory")
    annotate_bars(axes[1], live_bars, "{:.0f}")

    bottoms = [0.0 for _ in methods]
    series = [
        ("allreduce", "AllReduce"),
        ("reduce_scatter", "ReduceScatter"),
        ("all_gather", "AllGather"),
        ("broadcast", "Broadcast"),
    ]
    for key, label in series:
        vals = []
        for method in methods:
            breakdown = profiler_metrics[method]["nccl_breakdown_s"]
            total_kernel_s = profiler_metrics[method]["total_kernel_s"] or 1.0
            if key == "allreduce":
                raw = sum(value for name, value in breakdown.items() if "AllReduce" in name)
            elif key == "reduce_scatter":
                raw = sum(value for name, value in breakdown.items() if "ReduceScatter" in name)
            elif key == "all_gather":
                raw = sum(value for name, value in breakdown.items() if "AllGather" in name)
            else:
                raw = sum(value for name, value in breakdown.items() if "Broadcast" in name)
            vals.append(100.0 * raw / total_kernel_s)
        axes[2].bar(comm_x, vals, bottom=bottoms, label=label, color=COMM_COLORS[key], width=0.6)
        for i, value in enumerate(vals):
            if value >= 1.0:
                axes[2].annotate(
                    f"{label} {value:.1f}%",
                    (comm_x[i], bottoms[i] + value / 2.0),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#222222",
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, vals)]
    other_vals = [100.0 - value for value in bottoms]
    axes[2].bar(comm_x, other_vals, bottom=bottoms, label="Other GPU Kernels", color="#d0d0d0", width=0.6)
    axes[2].set_xticks(comm_x)
    axes[2].set_xticklabels([method.upper() for method in methods])
    styled_axis(axes[2], "Share of Total GPU Kernel Time (%)", "Communication Burden")
    axes[2].set_ylim(0, 100)
    for idx, total_comm in enumerate(bottoms):
        axes[2].annotate(
            f"NCCL {total_comm:.1f}%",
            (comm_x[idx], total_comm),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3)
    fig.suptitle("Full Fine-Tuning: DDP vs FSDP", y=1.03, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.03,
        "FSDP reduces peak and live memory, and its lower NCCL burden explains why it is more efficient in full fine-tuning.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    return save_figure(fig, "slide_1_final.png")


def plot_four_gpu_headline(records: list[ResultRecord]) -> Path | None:
    """
    Slide title:
    4-GPU Headline Comparison

    Caption:
    This is a compact raw-metrics snapshot for the most important deployment-sized
    setup in the study: 4 GPUs.

    Speaker notes:
    Use only if you want one quick top-line summary before showing the scaling
    and regime charts. Otherwise this can stay as appendix material.
    """
    grouped = summarize_by_gpu(records, "scale_r")
    ddp_records = grouped.get("ddp", {}).get(4, [])
    fsdp_records = grouped.get("fsdp", {}).get(4, [])
    if not ddp_records or not fsdp_records:
        return None

    metrics = [
        ("step_time_s", "Step Time (s)", "{:.3f}"),
        ("peak_mb", "Peak Memory (MB)", "{:.0f}"),
        ("alloc_mb", "Live Memory (MB)", "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    labels = ["DDP", "FSDP"]
    colors = [METHOD_COLORS["ddp"], METHOD_COLORS["fsdp"]]
    for ax, (field, title, fmt) in zip(axes, metrics):
        values = [metric_mean(ddp_records, field), metric_mean(fsdp_records, field)]
        bars = ax.bar(labels, values, color=colors, width=0.6)
        styled_axis(ax, title, title)
        annotate_bars(ax, bars, fmt)
    fig.suptitle("4-GPU Headline Comparison", y=1.02, fontsize=14, fontweight="bold")
    return save_figure(fig, "slide1_four_gpu_headline.png")


def plot_memory_stress_summary(records: list[ResultRecord]) -> Path | None:
    mem_records = [record for record in records if record.label.startswith("memlen")]
    if not mem_records:
        return None
    grouped: dict[str, dict[int, ResultRecord]] = defaultdict(dict)
    for record in mem_records:
        length = int(record.label.replace("memlen", ""))
        grouped[record.method][length] = record

    lengths = sorted({int(record.label.replace("memlen", "")) for record in mem_records})
    x = list(range(len(lengths)))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    metric_specs = [
        ("peak_mb", "Peak GPU Memory (MB)", "Memory Stress: Peak Memory"),
        ("step_time_s", "Average Step Time (s)", "Memory Stress: Step Time"),
    ]
    for ax, (field, ylabel, title) in zip(axes, metric_specs):
        ddp_vals = [getattr(grouped.get("ddp", {}).get(length), field, 0.0) or 0.0 for length in lengths]
        fsdp_vals = [getattr(grouped.get("fsdp", {}).get(length), field, 0.0) or 0.0 for length in lengths]
        ddp_bars = ax.bar([value - width / 2 for value in x], ddp_vals, width=width, color=METHOD_COLORS["ddp"], label="DDP")
        fsdp_bars = ax.bar([value + width / 2 for value in x], fsdp_vals, width=width, color=METHOD_COLORS["fsdp"], label="FSDP")
        styled_axis(ax, ylabel, title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(length) for length in lengths])
        ax.set_xlabel("Max Sequence Length")
        annotate_bars(ax, ddp_bars, "{:.3f}" if field == "step_time_s" else "{:.0f}")
        annotate_bars(ax, fsdp_bars, "{:.3f}" if field == "step_time_s" else "{:.0f}")

    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)
    return save_figure(fig, "report_memory_stress_summary.png")


def plot_profiler_communication_summary(metrics: dict[str, dict[str, Any]]) -> Path | None:
    """
    Slide title:
    Slide 1: Why FSDP Is Different

    Caption:
    Each bar is 100% of total GPU kernel time. Colored NCCL segments show both
    total communication burden and which collectives dominate it.

    Speaker notes:
    This combines two ideas into one visual: how communication-heavy the run is,
    and which collective operations are responsible. The gray remainder is
    non-communication GPU compute.
    """
    if not metrics:
        return None
    methods = [method for method in ("ddp", "fsdp") if method in metrics]
    if not methods:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = list(range(len(methods)))
    bottoms = [0.0 for _ in methods]
    series = [
        ("allreduce", "AllReduce"),
        ("reduce_scatter", "ReduceScatter"),
        ("all_gather", "AllGather"),
        ("broadcast", "Broadcast"),
    ]
    for key, label in series:
        vals = []
        for method in methods:
            breakdown = metrics[method]["nccl_breakdown_s"]
            total_kernel_s = metrics[method]["total_kernel_s"] or 1.0
            if key == "allreduce":
                raw = sum(value for name, value in breakdown.items() if "AllReduce" in name)
            elif key == "reduce_scatter":
                raw = sum(value for name, value in breakdown.items() if "ReduceScatter" in name)
            elif key == "all_gather":
                raw = sum(value for name, value in breakdown.items() if "AllGather" in name)
            else:
                raw = sum(value for name, value in breakdown.items() if "Broadcast" in name)
            vals.append(100.0 * raw / total_kernel_s)
        bars = ax.bar(x, vals, bottom=bottoms, label=label, color=COMM_COLORS[key], width=0.6)
        for idx, value in enumerate(vals):
            if value >= 1.0:
                ax.annotate(
                    f"{label} {value:.1f}%",
                    (x[idx], bottoms[idx] + value / 2.0),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#222222",
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, vals)]

    other_vals = [100.0 - value for value in bottoms]
    ax.bar(x, other_vals, bottom=bottoms, label="Other GPU Kernels", color="#d0d0d0", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([method.upper() for method in methods])
    styled_axis(ax, "Share of Total GPU Kernel Time (%)", "Communication Burden and Collective Mix")
    ax.set_ylim(0, 100)
    for idx, total_comm in enumerate(bottoms):
        ax.annotate(
            f"NCCL {total_comm:.1f}%",
            (x[idx], total_comm),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3)
    fig.suptitle("Profiler Summary: Why FSDP Behaves Differently", y=1.05, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.03,
        "Colored NCCL segments sum to total communication burden; gray is non-communication GPU compute. Broadcast is tiny and appears only in DDP.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    return save_figure(fig, "slide1_why_fsdp_is_different.png")


def plot_tta_summary(points: list[TtaPoint]) -> Path | None:
    if not points:
        return None
    grouped: dict[tuple[str, int], list[TtaPoint]] = defaultdict(list)
    for point in points:
        grouped[(point.method, point.gpus)].append(point)
    series_keys = sorted(grouped.keys(), key=lambda item: (item[1], item[0]))
    if not series_keys:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for method, gpus in series_keys:
        run_points = sorted(grouped[(method, gpus)], key=lambda point: point.step)
        xs = [point.step for point in run_points]
        acc = [100.0 * point.accuracy for point in run_points]
        runtime = [point.eval_runtime_s for point in run_points]
        label = f"{method.upper()} {gpus}G"
        axes[0].plot(xs, acc, marker="o", linewidth=2.2, label=label, color=METHOD_COLORS.get(method, "#333333"))
        axes[1].plot(xs, runtime, marker="o", linewidth=2.2, label=label, color=METHOD_COLORS.get(method, "#333333"))

    styled_axis(axes[0], "Validation Accuracy (%)", "Time-to-Accuracy")
    styled_axis(axes[1], "Eval Runtime (s)", "Cost of Evaluation")
    for ax in axes:
        ax.set_xlabel("Training Step")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
    return save_figure(fig, "report_tta_summary.png")


def plot_relative_improvement(records: list[ResultRecord], legacy_records: list[ResultRecord]) -> Path | None:
    """
    Slide title:
    Regime Dependence: FSDP Helps in Full FT, Not in PEFT

    Caption:
    Positive values mean FSDP improves over DDP. Full fine-tuning benefits from
    sharding, while PEFT shrinks or reverses that advantage.

    Speaker notes:
    Use this as the headline GPU chart. The important visual is the sign flip:
    FSDP is beneficial in full FT but not in PEFT for step time and peak memory.
    Train loss remains essentially unchanged, so this is a systems-cost result.
    """
    full_grouped = summarize_by_gpu(records, "scale_r")
    peft_records = [record for record in records if record.label == "peft" and record.gpus in (2, 4)]
    peft_by_method_gpu: dict[tuple[str, int], ResultRecord] = {}
    for record in peft_records:
        peft_by_method_gpu[(record.method, record.gpus)] = record
    for record in legacy_records:
        if record.gpus in (2, 4) and (record.method, record.gpus) not in peft_by_method_gpu:
            peft_by_method_gpu[(record.method, record.gpus)] = record

    labels = []
    step_improvement = []
    mem_improvement = []

    for gpu in (2, 4):
        ddp_full = metric_mean(full_grouped.get("ddp", {}).get(gpu, []), "step_time_s")
        fsdp_full = metric_mean(full_grouped.get("fsdp", {}).get(gpu, []), "step_time_s")
        ddp_full_mem = metric_mean(full_grouped.get("ddp", {}).get(gpu, []), "peak_mb")
        fsdp_full_mem = metric_mean(full_grouped.get("fsdp", {}).get(gpu, []), "peak_mb")
        if ddp_full and fsdp_full and ddp_full_mem and fsdp_full_mem:
            labels.append(f"FT {gpu}G")
            step_improvement.append(100.0 * (ddp_full - fsdp_full) / ddp_full)
            mem_improvement.append(100.0 * (ddp_full_mem - fsdp_full_mem) / ddp_full_mem)

        ddp_peft = peft_by_method_gpu.get(("ddp", gpu))
        fsdp_peft = peft_by_method_gpu.get(("fsdp", gpu))
        if ddp_peft and fsdp_peft and ddp_peft.step_time_s and fsdp_peft.step_time_s and ddp_peft.peak_mb and fsdp_peft.peak_mb:
            labels.append(f"PEFT {gpu}G")
            step_improvement.append(100.0 * (ddp_peft.step_time_s - fsdp_peft.step_time_s) / ddp_peft.step_time_s)
            mem_improvement.append(100.0 * (ddp_peft.peak_mb - fsdp_peft.peak_mb) / ddp_peft.peak_mb)

    if not labels:
        return None

    x = list(range(len(labels)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    step_bars = ax.bar(
        [value - width / 2 for value in x],
        step_improvement,
        width=width,
        color="#59a14f",
        label="Step-Time Improvement",
    )
    mem_bars = ax.bar(
        [value + width / 2 for value in x],
        mem_improvement,
        width=width,
        color="#e15759",
        label="Peak-Memory Reduction",
    )
    styled_axis(ax, "FSDP Improvement vs DDP (%)", "Regime Dependence of FSDP Benefits")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    annotate_bars(ax, step_bars, "{:.1f}", percent=True)
    annotate_bars(ax, mem_bars, "{:.1f}", percent=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    fig.text(
        0.5,
        -0.03,
        "Positive means FSDP is better. Full FT shows clear gains; PEFT removes or reverses them.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    return save_figure(fig, "slide1_relative_improvement.png")


def plot_slide1_regime_dumbbell(records: list[ResultRecord], legacy_records: list[ResultRecord]) -> Path | None:
    """
    Slide title:
    Regime Dependence at 4 GPUs

    Caption:
    Each line connects DDP to FSDP for the same regime. Full FT moves toward
    lower memory and lower step time with FSDP, while PEFT moves in the opposite direction.

    Speaker notes:
    Use this when you want a less dashboard-like visual than grouped bars.
    The viewer can immediately see directionality: FSDP shifts left for full FT
    and right for PEFT on both step time and peak memory.
    """
    full_grouped = summarize_by_gpu(records, "scale_r")
    peft_records = [record for record in records if record.label == "peft" and record.gpus == 4]
    peft_by_method: dict[str, ResultRecord] = {record.method: record for record in peft_records}
    for record in legacy_records:
        if record.gpus == 4 and record.method not in peft_by_method:
            peft_by_method[record.method] = record

    ddp_ft = metric_mean(full_grouped.get("ddp", {}).get(4, []), "step_time_s")
    fsdp_ft = metric_mean(full_grouped.get("fsdp", {}).get(4, []), "step_time_s")
    ddp_ft_mem = metric_mean(full_grouped.get("ddp", {}).get(4, []), "peak_mb")
    fsdp_ft_mem = metric_mean(full_grouped.get("fsdp", {}).get(4, []), "peak_mb")
    ddp_peft = peft_by_method.get("ddp")
    fsdp_peft = peft_by_method.get("fsdp")

    if not all([ddp_ft, fsdp_ft, ddp_ft_mem, fsdp_ft_mem, ddp_peft, fsdp_peft]):
        return None

    regimes = ["Full FT", "PEFT"]
    y = [1, 0]
    step_pairs = [
        (ddp_ft, fsdp_ft),
        (ddp_peft.step_time_s or 0.0, fsdp_peft.step_time_s or 0.0),
    ]
    mem_pairs = [
        (ddp_ft_mem, fsdp_ft_mem),
        (ddp_peft.peak_mb or 0.0, fsdp_peft.peak_mb or 0.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, pairs, xlabel, title, fmt in [
        (axes[0], step_pairs, "Average Step Time (s)", "4-GPU Step Time", "{:.3f}"),
        (axes[1], mem_pairs, "Peak GPU Memory (MB)", "4-GPU Peak Memory", "{:.0f}"),
    ]:
        for idx, ((ddp_val, fsdp_val), regime) in enumerate(zip(pairs, regimes)):
            ax.plot([ddp_val, fsdp_val], [y[idx], y[idx]], color="#999999", linewidth=2.0, zorder=1)
            ax.scatter(ddp_val, y[idx], color=METHOD_COLORS["ddp"], s=90, zorder=2, label="DDP" if idx == 0 else None)
            ax.scatter(fsdp_val, y[idx], color=METHOD_COLORS["fsdp"], s=90, zorder=2, label="FSDP" if idx == 0 else None)
            ax.annotate(f"DDP {fmt.format(ddp_val)}", (ddp_val, y[idx]), xytext=(-4, 9), textcoords="offset points", ha="right", fontsize=8)
            ax.annotate(f"FSDP {fmt.format(fsdp_val)}", (fsdp_val, y[idx]), xytext=(4, 9), textcoords="offset points", ha="left", fontsize=8)
            mid = (ddp_val + fsdp_val) / 2.0
            delta_pct = 100.0 * (ddp_val - fsdp_val) / ddp_val if ddp_val else 0.0
            ax.annotate(f"{delta_pct:+.1f}%", (mid, y[idx]), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8, color="#444444")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")
        ax.legend(loc="lower right")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(regimes)
    axes[0].set_ylabel("Training Regime")
    fig.suptitle("Regime Dependence at 4 GPUs", y=1.03, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.03,
        "FSDP helps in full fine-tuning but hurts in PEFT on these 4-GPU runs, while loss remains essentially unchanged.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    return save_figure(fig, "slide1_regime_dumbbell.png")


def plot_profiler_regime_dependence(ft_metrics: dict[str, dict[str, Any]], peft_metrics: dict[str, dict[str, Any]]) -> Path | None:
    """
    Slide title:
    Nsight Systems Explains the Regime Shift

    Caption:
    Full fine-tuning creates a large DDP communication burden that FSDP reduces.
    In PEFT, the profiler gap narrows, so FSDP overhead no longer pays off.

    Speaker notes:
    The left panel is the main apples-to-apples communication proxy: NCCL share
    of total GPU kernel time. The right panel reinforces it with device-to-device
    traffic volume. Use this slide to explain the sign flip seen in the regime plot.
    """
    if not ft_metrics or not peft_metrics:
        return None
    methods = [method for method in ("ddp", "fsdp") if method in ft_metrics and method in peft_metrics]
    if not methods:
        return None

    x = list(range(len(methods)))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ft_nccl = [100.0 * ft_metrics[method]["nccl_total_s"] / ft_metrics[method]["total_kernel_s"] for method in methods]
    peft_nccl = [100.0 * peft_metrics[method]["nccl_total_s"] / peft_metrics[method]["total_kernel_s"] for method in methods]
    ft_bars = axes[0].bar([value - width / 2 for value in x], ft_nccl, width=width, color="#4c78a8", label="Full FT")
    peft_bars = axes[0].bar([value + width / 2 for value in x], peft_nccl, width=width, color="#9ecae9", label="PEFT")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([method.upper() for method in methods])
    styled_axis(axes[0], "NCCL Share of GPU Kernel Time (%)", "Communication Burden")
    annotate_bars(axes[0], ft_bars, "{:.1f}", percent=True)
    annotate_bars(axes[0], peft_bars, "{:.1f}", percent=True)

    ft_d2d = [ft_metrics[method]["d2d_total_mb"] / 1024.0 for method in methods]
    peft_d2d = [peft_metrics[method]["d2d_total_mb"] / 1024.0 for method in methods]
    ft_d2d_bars = axes[1].bar([value - width / 2 for value in x], ft_d2d, width=width, color="#f58518", label="Full FT")
    peft_d2d_bars = axes[1].bar([value + width / 2 for value in x], peft_d2d, width=width, color="#fdd0a2", label="PEFT")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([method.upper() for method in methods])
    styled_axis(axes[1], "Device-to-Device Copy Volume (GB)", "Inter-GPU Data Movement")
    annotate_bars(axes[1], ft_d2d_bars, "{:.0f}")
    annotate_bars(axes[1], peft_d2d_bars, "{:.0f}")

    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)
    fig.suptitle("Nsight Systems Explains the Regime Shift", y=1.04, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.03,
        "In full FT, DDP is much more communication-heavy. In PEFT, the profiler gap compresses, matching the weaker FSDP case.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    return save_figure(fig, "slide2_profiler_regime_dependence.png")


def plot_tta_accuracy(points: list[TtaPoint]) -> Path | None:
    if not points:
        return None
    grouped: dict[tuple[str, int, str], list[TtaPoint]] = defaultdict(list)
    for point in points:
        grouped[(point.method, point.gpus, point.version)].append(point)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (method, gpus, version), run_points in sorted(grouped.items()):
        xs = [point.step for point in run_points]
        ys = [100.0 * point.accuracy for point in run_points]
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"{method.upper()} {gpus}G {version}", color=METHOD_COLORS.get(method, "#333333"))
    ax.set_title("Time-to-Accuracy Checkpoints")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, "tta_accuracy.png")


def plot_tta_eval_runtime(points: list[TtaPoint]) -> Path | None:
    if not points:
        return None
    grouped: dict[tuple[str, int, str], list[TtaPoint]] = defaultdict(list)
    for point in points:
        grouped[(point.method, point.gpus, point.version)].append(point)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (method, gpus, version), run_points in sorted(grouped.items()):
        xs = [point.step for point in run_points]
        ys = [point.eval_runtime_s for point in run_points]
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"{method.upper()} {gpus}G {version}", color=METHOD_COLORS.get(method, "#333333"))
    ax.set_title("Eval Pass Cost at TTA Checkpoints")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Eval Runtime (s)")
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, "tta_eval_runtime.png")


def plot_profiler_nccl_share(metrics: dict[str, dict[str, Any]]) -> Path | None:
    if not metrics:
        return None
    methods = [method for method in ("ddp", "fsdp") if method in metrics]
    shares = [100.0 * metrics[method]["nccl_total_s"] / metrics[method]["total_kernel_s"] for method in methods]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([method.upper() for method in methods], shares, color=[METHOD_COLORS[method] for method in methods], alpha=0.85)
    ax.set_title("Profiler: NCCL Share of Total GPU Kernel Time")
    ax.set_ylabel("NCCL Kernel Time (%)")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, "profiler_nccl_share.png")


def plot_profiler_nccl_breakdown(metrics: dict[str, dict[str, Any]]) -> Path | None:
    if not metrics:
        return None
    methods = [method for method in ("ddp", "fsdp") if method in metrics]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bottoms = [0.0 for _ in methods]
    series = [
        ("allreduce", "AllReduce", lambda names: sum(value for name, value in names.items() if "AllReduce" in name)),
        ("reduce_scatter", "ReduceScatter", lambda names: sum(value for name, value in names.items() if "ReduceScatter" in name)),
        ("all_gather", "AllGather", lambda names: sum(value for name, value in names.items() if "AllGather" in name)),
        ("broadcast", "Broadcast", lambda names: sum(value for name, value in names.items() if "Broadcast" in name)),
    ]
    x = list(range(len(methods)))
    for key, label, reducer in series:
        vals = [reducer(metrics[method]["nccl_breakdown_s"]) for method in methods]
        ax.bar(x, vals, bottom=bottoms, label=label, color=COMM_COLORS[key], alpha=0.9)
        bottoms = [bottom + value for bottom, value in zip(bottoms, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels([method.upper() for method in methods])
    ax.set_title("Profiler: NCCL GPU Kernel Breakdown")
    ax.set_ylabel("Total NCCL GPU Kernel Time (s)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return save_figure(fig, "profiler_nccl_breakdown.png")


def plot_profiler_nvtx_phase_breakdown(metrics: dict[str, dict[str, Any]]) -> Path | None:
    if not metrics:
        return None
    methods = [method for method in ("ddp", "fsdp") if method in metrics]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = list(range(len(methods)))
    bottoms = [0.0 for _ in methods]
    for phase in ("backward", "forward_loss", "optimizer_step"):
        vals = [metrics[method]["phase_pcts"].get(phase, 0.0) for method in methods]
        ax.bar(x, vals, bottom=bottoms, label=phase, color=PHASE_COLORS[phase], alpha=0.9)
        bottoms = [bottom + value for bottom, value in zip(bottoms, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels([method.upper() for method in methods])
    ax.set_title("Profiler: NVTX Training Phase Breakdown")
    ax.set_ylabel("Share of Named Train-Phase Time (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return save_figure(fig, "profiler_nvtx_phase_breakdown.png")


def plot_profiler_memcpy_volume(metrics: dict[str, dict[str, Any]]) -> Path | None:
    if not metrics:
        return None
    methods = [method for method in ("ddp", "fsdp") if method in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    total_gb = [metrics[method]["d2d_total_mb"] / 1024.0 for method in methods]
    copies = [metrics[method]["d2d_copies"] for method in methods]
    axes[0].bar([method.upper() for method in methods], total_gb, color=[METHOD_COLORS[method] for method in methods], alpha=0.85)
    axes[0].set_title("Profiler: Device-to-Device Copy Volume")
    axes[0].set_ylabel("Total D2D Copy Volume (GB)")
    axes[1].bar([method.upper() for method in methods], copies, color=[METHOD_COLORS[method] for method in methods], alpha=0.85)
    axes[1].set_title("Profiler: Device-to-Device Copy Count")
    axes[1].set_ylabel("Number of D2D Copies")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, "profiler_d2d_memcpy.png")


def main() -> None:
    apply_report_style()
    ensure_plots_dir()
    output_records = load_output_results()
    legacy_records = load_legacy_peft_results()
    tta_points = load_tta_points()
    profiler_metrics = load_profiler_metrics()
    profiler_metrics_peft = load_profiler_metrics_from_paths(NSYS_SQLITES_PEFT)

    saved_paths = [
        plot_four_gpu_headline(output_records),
        plot_scaling_summary(output_records),
        plot_slide1_final(output_records, profiler_metrics),
        plot_slide1_regime_dumbbell(output_records, legacy_records),
        plot_peft_vs_full_finetune(output_records, legacy_records),
        plot_relative_improvement(output_records, legacy_records),
        plot_memory_stress_summary(output_records),
        plot_tta_summary(tta_points),
        plot_profiler_communication_summary(profiler_metrics),
        plot_profiler_regime_dependence(profiler_metrics, profiler_metrics_peft),
    ]

    print("Saved plots:")
    for path in saved_paths:
        if path is not None:
            print(f"- {path}")


if __name__ == "__main__":
    main()
