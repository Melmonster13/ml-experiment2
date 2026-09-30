"""Render the four result charts from results/runs.csv into results/plots/."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"

# Categorical slots in fixed order: fp32, 4-bit, 8-bit.
PRECISION_COLORS = {"32": "#2a78d6", "4": "#eb6834", "8": "#1baf7a"}
PRECISION_LABELS = {"32": "fp32 (LoRA)", "4": "4-bit (QLoRA)", "8": "8-bit (QLoRA)"}
BAR_COLOR = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEFAULT_METAL_CAP_GB = 11.84  # max_recommended_working_set_size before raising iogpu.wired_limit_mb

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def load_runs() -> list[dict]:
    with (RESULTS / "runs.csv").open() as f:
        return list(csv.DictReader(f))


def label(r: dict) -> str:
    return r["run_id"].upper()


def memory_vs_loss(runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ok = [r for r in runs if r["status"] == "ok"]
    for bits, color in PRECISION_COLORS.items():
        pts = [r for r in ok if r["bits"] == bits]
        if pts:
            ax.scatter([float(r["peak_mem_gb"]) for r in pts], [float(r["final_val_loss"]) for r in pts],
                       s=64, color=color, edgecolor="white", linewidth=2, label=PRECISION_LABELS[bits], zorder=3)
    for r in ok:
        ax.annotate(label(r), (float(r["peak_mem_gb"]), float(r["final_val_loss"])),
                    xytext=(7, 4), textcoords="offset points", color=INK, fontsize=9)
    ax.axvline(DEFAULT_METAL_CAP_GB, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.text(DEFAULT_METAL_CAP_GB - 0.15, ax.get_ylim()[1], "default Metal cap\n(11.84 GB)",
            ha="right", va="top", color=MUTED, fontsize=8)
    failed = [label(r) for r in runs if r["status"] != "ok"]
    if failed:
        ax.text(0.99, 0.02, f"Not plotted (OOM/error): {', '.join(failed)}", transform=ax.transAxes,
                ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.set_xlabel("Peak memory (GB)")
    ax.set_ylabel("Final val loss")
    ax.set_title("Peak memory vs validation loss", loc="left", color=INK)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(PLOTS / "memory_vs_val_loss.png")


def read_metrics(run_id: str) -> tuple[list, list]:
    train, val = [], []
    with (RESULTS / "logs" / f"{run_id}_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            if row["train_loss"]:
                train.append((int(row["iter"]), float(row["train_loss"])))
            if row["val_loss"]:
                val.append((int(row["iter"]), float(row["val_loss"])))
    return sorted(train), sorted(val)


def loss_curves(runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_id = {r["run_id"]: r for r in runs}
    for run_id in ["r1", "r2"]:
        if run_id not in by_id:
            continue
        color = PRECISION_COLORS[by_id[run_id]["bits"]]
        train, val = read_metrics(run_id)
        name = f"{run_id.upper()} {PRECISION_LABELS[by_id[run_id]['bits']]}"
        ax.plot(*zip(*train), color=color, linewidth=1, alpha=0.45, label=f"{name}: train")
        ax.plot(*zip(*val), color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=2, label=f"{name}: val")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over training: LoRA (R1) vs QLoRA 4-bit (R2)", loc="left", color=INK)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "loss_curves_r1_r2.png")


def bars(ax, names: list[str], values: list[float], fmt: str) -> None:
    rects = ax.bar(names, values, width=0.6, color=BAR_COLOR, edgecolor="white", linewidth=2)
    ax.bar_label(rects, labels=[fmt.format(v) for v in values], padding=3, color=INK, fontsize=9)
    ax.grid(axis="x", visible=False)


def tokens_per_sec(runs: list[dict]) -> None:
    ok = [r for r in runs if r["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars(ax, [label(r) for r in ok], [float(r["tokens_per_sec"]) for r in ok], "{:.0f}")
    ax.set_ylabel("Tokens/sec (mean over reports)")
    ax.set_title("Training throughput per run", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(PLOTS / "tokens_per_sec.png")


def group_size(runs: list[dict]) -> None:
    q4 = sorted((r for r in runs if r["bits"] == "4" and r["status"] == "ok" and r["run_id"] in {"r2", "r4", "r5"}),
                key=lambda r: int(r["group_size"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars(ax, [f"g{r['group_size']} ({label(r)})" for r in q4], [float(r["final_val_loss"]) for r in q4], "{:.3f}")
    ax.set_ylabel("Final val loss")
    ax.set_title("4-bit val loss by quantization group size", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(PLOTS / "val_loss_by_group_size.png")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    memory_vs_loss(runs)
    loss_curves(runs)
    tokens_per_sec(runs)
    group_size(runs)
    print(f"Wrote charts to {PLOTS}")


if __name__ == "__main__":
    main()
