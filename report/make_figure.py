"""Figure 1: the detector's location and the information available at that point.

    python report/make_figure.py  ->  report/figure_detector_location.png / .pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "figure_detector_location"

INK = "#1f2430"
MUT = "#5b6478"
GREEN = "#2e8b57"
RED = "#c0392b"
FILLS = {"client": "#e8eefc", "proxy": "#fdf3e0", "app": "#e7f4ec"}
EDGES = {"client": "#4a6fa5", "proxy": "#c98a12", "app": "#2e8b57"}

VISIBLE = [
    "HTTP method, path, query string",
    "request headers, cookies, body",
    "the once-decoded parameter value",
    "timing and byte counts",
]
INVISIBLE = [
    "which output context the value reaches",
    "whether the browser executes it",
    "whether a user ever triggers the handler",
    "the URL fragment (never sent in the request)",
    "response body, unless also inspected",
]


def panel(ax, x, y, w, h, key, title, lines):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=FILLS[key], edgecolor=EDGES[key],
                               linewidth=1.6, zorder=1))
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="top",
            fontsize=11.5, fontweight="bold", color=INK, zorder=2)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.74 - i * 0.31, ln, ha="center", va="top",
                fontsize=8.5, color=MUT, zorder=2)


def flow(ax, x1, x2, y, label):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.5, color=INK))
    ax.text((x1 + x2) / 2, y + 0.12, label, ha="center", va="bottom",
            fontsize=8.3, color=MUT)


def bullet_block(ax, x, y, title, items, colour, mark):
    ax.text(x, y, title, fontsize=10.2, fontweight="bold", color=colour, va="top")
    for i, it in enumerate(items):
        ax.text(x, y - 0.40 - i * 0.33, mark, fontsize=9.5, color=colour, va="top")
        ax.text(x + 0.22, y - 0.40 - i * 0.33, it, fontsize=8.8, color=INK, va="top")


def main():
    W, H = 12.4, 7.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(0.25, H - 0.18, "Figure 1  Detector location and available information",
            fontsize=14, fontweight="bold", color=INK, va="top")
    ax.text(0.25, H - 0.58, "Local experiment, plaintext HTTP over 127.0.0.1. "
                            "The detector runs inline in the reverse proxy.",
            fontsize=9.2, color=MUT, va="top")

    # --- row of components -------------------------------------------------
    row_y, box_h = 4.92, 1.72
    panel(ax, 0.25, row_y, 3.1, box_h, "client", "Browser / load client",
          ["issues the request", "renders the response", "runs any script"])
    panel(ax, 4.55, row_y, 3.6, box_h, "proxy", "Reverse proxy  :8080",
          ["DETECTOR HERE", "char n-gram TF-IDF", "modes: off | observe | block"])
    panel(ax, 9.35, row_y, 2.8, box_h, "app", "Flask app  :5000",
          ["reflects the value into", "four output contexts"])

    mid = row_y + box_h * 0.62
    flow(ax, 3.42, 4.48, mid, "request")
    flow(ax, 8.22, 9.28, mid, "forwarded raw")

    # response path drawn BELOW the boxes so it never crosses their text
    resp_y = row_y - 0.48
    ax.annotate("", xy=(0.55, resp_y), xytext=(11.85, resp_y),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.2, color=MUT,
                                linestyle=(0, (5, 3))))
    for x in (0.55, 11.85):
        ax.plot([x, x], [resp_y, row_y], linestyle=(0, (5, 3)), linewidth=1.2, color=MUT)
    ax.text(6.20, resp_y - 0.10, "HTTP response — not inspected in this experiment",
            ha="center", va="top", fontsize=8.4, color=MUT)

    # --- what is and is not available at that point ------------------------
    cap_y = 3.88
    ax.plot([4.55, 8.15], [cap_y + 0.16, cap_y + 0.16], linewidth=1.2, color=EDGES["proxy"])
    ax.text(6.35, cap_y, "information available at the detection point",
            ha="center", va="top", fontsize=9.6, fontstyle="italic", color=EDGES["proxy"])

    bullet_block(ax, 0.60, 3.55, "Visible to the detector", VISIBLE, GREEN, "+")
    bullet_block(ax, 6.55, 3.55, "NOT visible to the detector", INVISIBLE, RED, "–")

    # --- closing band ------------------------------------------------------
    ax.add_patch(plt.Rectangle((0.25, 0.28), 11.9, 0.86, facecolor="#f4f6fa",
                               edgecolor="#d4dae6", linewidth=1.0))
    ax.text(6.20, 1.00,
            "Measured consequence: identical request content is scored identically, yet the "
            "same value is inert at /reflect-text and executes at /reflect-html.",
            ha="center", va="top", fontsize=9.0, color=INK)
    ax.text(6.20, 0.66,
            "Under TLS this inspection point must sit at or after authorised termination; "
            "a passive encrypted-traffic monitor sees none of the above.",
            ha="center", va="top", fontsize=8.6, color=MUT)

    fig.savefig(f"{OUT}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    sys.exit(main())
