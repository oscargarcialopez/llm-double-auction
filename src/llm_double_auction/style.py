"""The shared look of the figures and the carousel: white, three inks, one hue per model,
IBM Plex Sans throughout (vendored in fonts/). The two model hues pass the dataviz palette
checks on white (lightness band, chroma floor, adjacent CVD delta E 10.3, normal-vision 20.1,
both >= 3:1)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONTS = Path(__file__).parent / "fonts"

SURFACE = "#ffffff"
INK = "#1a1a1a"
INK_2 = "#4d4d4d"
MUTED = "#8c8c8c"
GRID = "#e8e8e8"
AXIS = "#c8c8c8"
WASH = "#f0f0f0"

MODEL_COLOR = {"gemini-3.5-flash-lite": "#b8790e", "gemini-3.8-flash": "#00846f"}
MODEL_NAME = {"gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
              "gemini-3.8-flash": "Gemini 3.8 Flash"}
FRAMING = {"informed-v1": "auction prompt", "shopper-v1": "shopping-assistant prompt",
           "informed-k13-v1": "auction prompt, k = 1/3"}


def apply() -> None:
    for font in sorted(FONTS.glob("*.ttf")):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": ["IBM Plex Sans", "DejaVu Sans"],
        "mathtext.fontset": "cm",
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
