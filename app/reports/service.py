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


def _fig_to_base64_png(fig: "plt.Figure") -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _rating_distribution_chart(metrics: Metrics) -> str:
    stars = [1, 2, 3, 4, 5]
    counts = [metrics.rating_distribution[star].count for star in stars]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([str(star) for star in stars], counts, color="#4C72B0")
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Reviews")
    ax.set_title("Rating distribution")
    fig.tight_layout()
    return _fig_to_base64_png(fig)


def _sentiment_distribution_chart(insight: Insight) -> str:
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        insight.sentiment_distribution.positive,
        insight.sentiment_distribution.neutral,
        insight.sentiment_distribution.negative,
    ]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, values, color=["#55A868", "#8C8C8C", "#C44E52"])
    ax.set_ylabel("Reviews")
    ax.set_title("Sentiment distribution")
    fig.tight_layout()
    return _fig_to_base64_png(fig)


def render_report(sample: ReviewsSample, metrics: Metrics, insight: Insight) -> str:
    template = _env.get_template("report.html")
    return template.render(
        sample=sample,
        metrics=metrics,
        insight=insight,
        rating_chart=_rating_distribution_chart(metrics),
        sentiment_chart=_sentiment_distribution_chart(insight),
    )
