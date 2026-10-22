"""The shared look of the figures and the carousel: one surface, three inks, one hue per
model. The two model hues are categorical slots 1 and 2 of a validated palette (adjacent
CVD delta E 24.7, normal-vision 33.6, both >= 3:1 on the surface)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WASH = "#f0efec"

MODEL_COLOR = {"gemini-3.5-flash-lite": "#2a78d6", "gemini-3.8-flash": "#eb6834"}
MODEL_NAME = {"gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
              "gemini-3.8-flash": "Gemini 3.8 Flash"}
FRAMING = {"informed-v1": "auction prompt", "shopper-v1": "shopping-assistant prompt",
           "informed-k13-v1": "auction prompt, k = 1/3"}


def apply() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def load_summary(results_dir: Path) -> dict:
    return json.loads((Path(results_dir) / "summary.json").read_text())


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def pct_ci(lo_hi, digits: int = 1) -> str:
    lo, hi = lo_hi
    return f"[{100 * lo:.{digits}f}, {100 * hi:.{digits}f}]"
