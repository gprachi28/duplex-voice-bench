"""Generates PNGs from benchmark results -- see benchmarks/README.md.

Two plots today (one wired combination, single machine):
  - stage_breakdown.png -- p50/p95 per-stage latency, stacked bar per combination
  - ttfa_distribution.png -- TTFA histogram with p50/p95/p99 lines, per combination

The Pareto frontier (accuracy vs p95 TTFA) needs >= 2 combinations with a WER
axis and isn't here yet -- see design.md's "Benchmark Combinations" section.

Colors are the dataviz skill's validated reference categorical palette
(references/palette.md), used in its fixed slot order.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.eval_latency import group_by_combination, load_runs, percentile

STAGE_LABELS = {
    "end_of_turn_s": "End of turn",
    "transcription_s": "Transcription",
    "llm_ttft_s": "LLM TTFT",
    "sentence_buffer_s": "Sentence buffer",
    "tts_first_chunk_s": "TTS first chunk",
}
# ttfa_s is deliberately excluded from STAGE_LABELS -- it's the sum of the
# other five, not a stage of its own, and would double-count the bar.

STAGE_COLORS = {
    "end_of_turn_s": "#2a78d6",
    "transcription_s": "#1baf7a",
    "llm_ttft_s": "#eda100",
    "sentence_buffer_s": "#008300",
    "tts_first_chunk_s": "#4a3aa7",
}
COMBO_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

CHART_SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"


def stage_breakdown_series(summary: dict, quantile: str) -> list[tuple[str, float]]:
    """Ordered (stage label, value) pairs for one combination's stacked bar
    at the given quantile ("p50" or "p95"/"p99"). Stages with no recorded
    value are skipped."""
    return [
        (label, summary[stage][quantile])
        for stage, label in STAGE_LABELS.items()
        if summary[stage][quantile] is not None
    ]


def plot_stage_breakdown(latency_summary: dict[str, dict], out_path: str) -> None:
    combos = list(latency_summary)
    quantiles = ("p50", "p95")
    bar_height = 0.32
    gap = 0.06

    fig, ax = plt.subplots(figsize=(9, 1.1 * len(combos) + 1.5))
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    row_labels, row_ticks = [], []
    for i, combo in enumerate(combos):
        for j, quantile in enumerate(quantiles):
            y = i * (len(quantiles) * (bar_height + gap)) + j * (bar_height + gap)
            row_labels.append(f"{combo} ({quantile})")
            row_ticks.append(y)

            left = 0.0
            for stage_label, value in stage_breakdown_series(
                latency_summary[combo], quantile
            ):
                stage_key = next(k for k, v in STAGE_LABELS.items() if v == stage_label)
                ax.barh(
                    y,
                    value,
                    left=left,
                    height=bar_height,
                    color=STAGE_COLORS[stage_key],
                    edgecolor=CHART_SURFACE,
                    linewidth=2,
                )
                left += value
            ax.text(left + 0.01, y, f"{left:.2f}s", va="center", fontsize=8, color=PRIMARY_INK)

    ax.set_yticks(row_ticks)
    ax.set_yticklabels(row_labels, color=PRIMARY_INK, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (s)", color=MUTED_INK)
    ax.grid(axis="x", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in STAGE_COLORS.values()]
    ax.legend(
        handles,
        STAGE_LABELS.values(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        labelcolor=PRIMARY_INK,
    )

    fig.tight_layout()
    # bbox_inches="tight" here (not just fig.tight_layout()) because the
    # legend sits below the axes via bbox_to_anchor -- without it, the
    # legend's right-most entry gets clipped at the figure's saved bounds.
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_ttfa_distribution(grouped_records: dict[str, list[dict]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    for i, (combo, records) in enumerate(grouped_records.items()):
        values = sorted(r["ttfa_s"] for r in records if r.get("ttfa_s") is not None)
        if not values:
            continue
        color = COMBO_COLORS[i % len(COMBO_COLORS)]
        ax.hist(
            values,
            bins=min(20, max(5, len(values) // 2)),
            color=color,
            alpha=0.55,
            label=combo,
            edgecolor=CHART_SURFACE,
        )
        for q, style in ((0.50, "-"), (0.95, "--"), (0.99, ":")):
            v = percentile(values, q)
            if v is not None:
                ax.axvline(v, color=color, linestyle=style, linewidth=1.5)

    ax.set_xlabel("TTFA (s)", color=MUTED_INK)
    ax.set_ylabel("Turns", color=MUTED_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def main(
    runs_dir: str = "benchmarks/results/runs",
    summary_path: str = "benchmarks/results/latency_summary.json",
    plots_dir: str = "benchmarks/plots",
) -> None:
    with open(summary_path) as f:
        latency_summary = json.load(f)
    grouped_records = group_by_combination(load_runs(runs_dir))

    os.makedirs(plots_dir, exist_ok=True)
    plot_stage_breakdown(latency_summary, os.path.join(plots_dir, "stage_breakdown.png"))
    plot_ttfa_distribution(grouped_records, os.path.join(plots_dir, "ttfa_distribution.png"))


if __name__ == "__main__":
    main()
