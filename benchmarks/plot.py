"""Generates PNGs from benchmark results -- see benchmarks/README.md.

Four plots today (one wired combination, single machine):
  - stage_breakdown.png -- p50/p95 per-stage latency, stacked bar per combination
  - ttfa_distribution.png -- TTFA strip plot (one row per combination, each
    turn a dot, p50/p95 marked) -- small multiples instead of overlapping
    histograms, see plot_ttfa_distribution's docstring for why
  - ttfa_trend.png -- TTFA p50/p95 across a combination's change-tags, chronological
  - stage_trend.png -- per-stage p50 stacked area across the same change-tags

The Pareto frontier (accuracy vs p95 TTFA) needs >= 2 combinations with a WER
axis and isn't here yet -- see design.md's "Benchmark Combinations" section.

Colors are the dataviz skill's validated reference categorical palette
(references/palette.md), used in its fixed slot order.
"""

from __future__ import annotations

import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.eval_latency import (
    combo_tag_key,
    group_by_combination_and_tag,
    load_runs_with_tags,
    ordered_tags_by_combination,
    percentile,
)

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
    """One row per combination: every turn's TTFA as a dot, p50 as a filled
    diamond, p95 as a tick (skipped if it ties p50 -- common at n<20 with
    nearest-rank percentiles). Replaces an earlier overlapping-histogram
    version: combination_id strings are long, so a legend of them overflowed
    the figure, and histogram bins are noisy at the ~11-17 turns per
    combination this project runs. Row labels carry identity instead of a
    legend, the same pattern plot_stage_breakdown already uses."""
    combos = list(grouped_records)
    row_span = 0.34
    row_unit = 1.6  # spacing between row centers -- wide enough that one
    # row's p95 label and the next row's p50 label never occupy the same y

    fig, ax = plt.subplots(figsize=(9, 0.9 * row_unit * len(combos) + 1.2))
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    for i, combo in enumerate(combos):
        y0 = i * row_unit
        values = sorted(r["ttfa_s"] for r in grouped_records[combo] if r.get("ttfa_s") is not None)
        if not values:
            continue
        color = COMBO_COLORS[i % len(COMBO_COLORS)]
        jitter = random.Random(combo)
        y = [y0 + jitter.uniform(-row_span, row_span) for _ in values]
        ax.scatter(values, y, s=70, color=color, edgecolors=CHART_SURFACE, linewidths=1.5, alpha=0.85, zorder=3)

        p50 = percentile(values, 0.50)
        p95 = percentile(values, 0.95)
        if p50 is not None:
            ax.scatter([p50], [y0], marker="D", s=110, color=color, edgecolors=CHART_SURFACE, linewidths=1.5, zorder=4)
            ax.text(p50, y0 - row_span * 1.35, f"p50 {p50:.2f}s", color=PRIMARY_INK, fontsize=8, ha="center")
        if p95 is not None and p95 != p50:
            ax.plot([p95, p95], [y0 - row_span, y0 + row_span], color=color, linewidth=3, zorder=4)
            ax.text(p95, y0 + row_span * 1.55, f"p95 {p95:.2f}s", color=MUTED_INK, fontsize=8, ha="center")

    ax.set_yticks([i * row_unit for i in range(len(combos))])
    ax.set_yticklabels(combos, color=PRIMARY_INK, fontsize=8)
    # Explicit (inverted) limits, not invert_yaxis() on the autoscaled range --
    # autoscale doesn't reserve room for the top row's p50 label or the
    # bottom row's p95 label, so both were clipping against the figure edge.
    ax.set_ylim((len(combos) - 1) * row_unit + row_span * 2.0, -row_span * 2.0)
    ax.set_xlabel("TTFA (s)", color=MUTED_INK)
    ax.grid(axis="x", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def _single_trend_base(ordered_tags: dict[str, list[str]]) -> str | None:
    """Returns the one (combination_id, prompt_version) base with >=2
    change-tags, or None if none qualify. Cross-combination trend
    comparison is out of scope (only one combination is wired today) --
    if more than one base qualifies, warns to stderr and returns None
    rather than guessing which to plot."""
    candidates = [base for base, tags in ordered_tags.items() if len(tags) >= 2]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(
            "trend plots: skipped, multiple combinations have >=2 change-tags "
            f"({candidates}) -- cross-combination trend charts are out of scope",
            file=sys.stderr,
        )
    return None


def plot_ttfa_trend(
    latency_summary: dict[str, dict],
    ordered_tags: dict[str, list[str]],
    out_path: str,
) -> None:
    base = _single_trend_base(ordered_tags)
    if base is None:
        print(f"plot_ttfa_trend: skipped ({out_path}), no single combination has >=2 change-tags yet", file=sys.stderr)
        return
    tags = ordered_tags[base]

    p50s, p95s = [], []
    for tag in tags:
        stage = latency_summary.get(combo_tag_key(base, tag), {}).get("ttfa_s", {})
        p50s.append(stage.get("p50") or 0.0)
        p95s.append(stage.get("p95") or 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    ax.plot(tags, p50s, color=COMBO_COLORS[0], linewidth=2, marker="o", label="p50")
    ax.plot(tags, p95s, color=COMBO_COLORS[0], linewidth=2, linestyle="--", marker="o", label="p95")
    ax.axhline(1.0, color=MUTED_INK, linestyle="--", linewidth=1)
    ax.text(0, 1.02, "1.0s target", color=MUTED_INK, fontsize=8)

    ax.set_title(base, color=PRIMARY_INK, fontsize=10)
    ax.set_ylabel("TTFA (s)", color=MUTED_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_stage_trend(
    latency_summary: dict[str, dict],
    ordered_tags: dict[str, list[str]],
    out_path: str,
) -> None:
    base = _single_trend_base(ordered_tags)
    if base is None:
        print(f"plot_stage_trend: skipped ({out_path}), no single combination has >=2 change-tags yet", file=sys.stderr)
        return
    tags = ordered_tags[base]

    series: dict[str, list[float]] = {stage: [] for stage in STAGE_LABELS}
    for tag in tags:
        summary = latency_summary.get(combo_tag_key(base, tag), {})
        for stage in STAGE_LABELS:
            value = summary.get(stage, {}).get("p50")
            series[stage].append(value if value is not None else 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    x = range(len(tags))
    ax.stackplot(
        x,
        *[series[stage] for stage in STAGE_LABELS],
        labels=list(STAGE_LABELS.values()),
        colors=[STAGE_COLORS[stage] for stage in STAGE_LABELS],
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(tags, color=PRIMARY_INK)
    ax.set_title(base, color=PRIMARY_INK, fontsize=10)
    ax.set_ylabel("p50 latency (s)", color=MUTED_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper left")

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
    tagged_records = load_runs_with_tags(runs_dir)
    grouped_records = group_by_combination_and_tag(tagged_records)
    ordered_tags = ordered_tags_by_combination(tagged_records)

    os.makedirs(plots_dir, exist_ok=True)
    plot_stage_breakdown(latency_summary, os.path.join(plots_dir, "stage_breakdown.png"))
    plot_ttfa_distribution(grouped_records, os.path.join(plots_dir, "ttfa_distribution.png"))
    plot_ttfa_trend(latency_summary, ordered_tags, os.path.join(plots_dir, "ttfa_trend.png"))
    plot_stage_trend(latency_summary, ordered_tags, os.path.join(plots_dir, "stage_trend.png"))


if __name__ == "__main__":
    main()
