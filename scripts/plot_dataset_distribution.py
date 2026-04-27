"""Generate dataset distribution figures for thesis Chapter 4.

Outputs:
  - docs/Bao_cao_khoa_luan/figures/c4/dataset_difficulty_donut.png
  - docs/Bao_cao_khoa_luan/figures/c4/dataset_intent_barh.png
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "data" / "evaluation" / "traces" / "full_run_v12.jsonl"
OUTPUT_DIR = ROOT / "docs" / "Bao_cao_khoa_luan" / "figures" / "c4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False

DIFF_LABELS = {"easy": "Đơn giản", "medium": "Trung bình", "hard": "Đa bước"}
DIFF_ORDER = ["easy", "medium", "hard"]
DIFF_COLORS = ["#B7E4C7", "#52B788", "#1B4332"]

INTENT_LABELS = {
    "activity_weather": "Tư vấn hoạt động",
    "weather_alert": "Cảnh báo thời tiết",
    "rain_query": "Hỏi về mưa",
    "temperature_query": "Hỏi về nhiệt độ",
    "smalltalk_weather": "Trò chuyện thời tiết",
    "daily_forecast": "Dự báo theo ngày",
    "seasonal_context": "So sánh mùa vụ",
    "wind_query": "Hỏi về gió",
    "humidity_fog_query": "Hỏi về độ ẩm, sương mù",
    "hourly_forecast": "Dự báo theo giờ",
    "weather_overview": "Tổng quan thời tiết",
    "historical_weather": "Thời tiết quá khứ",
    "expert_weather_param": "Thông số chuyên sâu",
    "current_weather": "Thời tiết hiện tại",
    "location_comparison": "So sánh địa điểm",
}


def load_distribution() -> tuple[Counter, Counter]:
    intent_count: Counter = Counter()
    diff_count: Counter = Counter()
    with TRACE_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            intent_count[d["expected"]["intent"]] += 1
            diff_count[d["expected"]["difficulty"]] += 1
    return intent_count, diff_count


def plot_donut(diff_count: Counter) -> Path:
    values = [diff_count[d] for d in DIFF_ORDER]
    labels = [DIFF_LABELS[d] for d in DIFF_ORDER]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    wedges, _, autotexts = ax.pie(
        values,
        labels=labels,
        colors=DIFF_COLORS,
        autopct=lambda p: f"{int(round(p * total / 100))}\n({p:.1f}%)",
        pctdistance=0.78,
        startangle=90,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 13, "fontweight": "bold"},
    )

    text_colors = ["#1B4332", "white", "white"]
    for at, c in zip(autotexts, text_colors):
        at.set_color(c)
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax.set_title("Phân bố theo độ khó", fontsize=15, fontweight="bold", pad=18)
    plt.tight_layout()

    out = OUTPUT_DIR / "dataset_difficulty_donut.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def plot_intent_barh(intent_count: Counter) -> Path:
    sorted_intents = sorted(intent_count.items(), key=lambda x: x[1])
    keys = [k for k, _ in sorted_intents]
    counts = [v for _, v in sorted_intents]
    labels = [INTENT_LABELS.get(k, k) for k in keys]

    cmap = plt.cm.Greens
    cmin, cmax = min(counts), max(counts)
    span = cmax - cmin if cmax != cmin else 1
    norm = [(c - cmin) / span for c in counts]
    colors = [cmap(0.35 + 0.55 * n) for n in norm]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    bars = ax.barh(labels, counts, color=colors, edgecolor="white", linewidth=0.8)

    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            str(c),
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#1B4332",
        )

    ax.set_xlabel("Số câu hỏi", fontsize=12)
    ax.set_title("Phân bố theo nhóm ý định", fontsize=15, fontweight="bold", pad=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlim(0, max(counts) * 1.12)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()

    out = OUTPUT_DIR / "dataset_intent_barh.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def main() -> None:
    intent_count, diff_count = load_distribution()
    print(f"Total: {sum(intent_count.values())}")
    print(f"Difficulty: {dict(diff_count)}")

    p1 = plot_donut(diff_count)
    print(f"Saved: {p1.relative_to(ROOT)}")
    p2 = plot_intent_barh(intent_count)
    print(f"Saved: {p2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
