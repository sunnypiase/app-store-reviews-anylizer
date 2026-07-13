import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless server process — never open a GUI window
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.insights.schemas import Insight
from app.metrics.schemas import Metrics
from app.reviews.schemas import ReviewsSample

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=select_autoescape())

# Chart ink/chrome tokens (light surface only — the report prints to PDF).
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED = "#898781"
_BASELINE = "#c3c2b7"
_SURFACE = "#ffffff"
_SERIES_BLUE = "#2a78d6"
# Status colors for sentiment polarity — reserved roles, not series hues.
_SENTIMENT_COLORS = {"positive": "#0ca30c", "neutral": "#898781", "negative": "#d03b3b"}


def _fig_to_base64_png(fig: "plt.Figure") -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, facecolor=_SURFACE)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _bar_chart(labels: list[str], values: list[int], colors: list[str], title: str) -> str:
    """Values are printed on the bars, so the y-axis and grid would be redundant ink."""
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.set_facecolor(_SURFACE)
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.bar_label(bars, padding=3, color=_SECONDARY_INK, fontsize=9)
    ax.set_title(title, loc="left", color=_INK, fontsize=11, fontweight="bold", pad=12)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=_MUTED, length=0, labelsize=9)
    ax.set_ylim(0, max(max(values), 1) * 1.18)
    fig.tight_layout()
    return _fig_to_base64_png(fig)


def _rating_distribution_chart(metrics: Metrics) -> str:
    stars = [1, 2, 3, 4, 5]
    return _bar_chart(
        [f"{star}★" for star in stars],
        [metrics.rating_distribution[star].count for star in stars],
        [_SERIES_BLUE] * len(stars),
        "Rating distribution",
    )


def _sentiment_distribution_chart(insight: Insight) -> str:
    distribution = insight.sentiment_distribution
    return _bar_chart(
        ["Positive", "Neutral", "Negative"],
        [distribution.positive, distribution.neutral, distribution.negative],
        [
            _SENTIMENT_COLORS["positive"],
            _SENTIMENT_COLORS["neutral"],
            _SENTIMENT_COLORS["negative"],
        ],
        "Sentiment distribution",
    )


def render_report(sample: ReviewsSample, metrics: Metrics, insight: Insight) -> str:
    template = _env.get_template("report.html")
    total = max(insight.review_count, 1)
    distribution = insight.sentiment_distribution
    return template.render(
        sample=sample,
        metrics=metrics,
        insight=insight,
        reviews_by_id={review.id: review for review in sample.reviews},
        sentiment_percent={
            "positive": 100 * distribution.positive / total,
            "neutral": 100 * distribution.neutral / total,
            "negative": 100 * distribution.negative / total,
        },
        rating_chart=_rating_distribution_chart(metrics),
        sentiment_chart=_sentiment_distribution_chart(insight),
    )
