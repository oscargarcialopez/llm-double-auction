"""The LinkedIn carousel: nine 4:5 slides (1080 x 1350 px) laid out like talk slides, one PDF
plus PNG previews.

Every number on the slides comes from results/summary.json through facts.build(); theory
constants come from theory.py, and the two quotes are read from the cached calls.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

from . import facts as fx
from . import style
from . import theory as th
from .style import INK, INK_2, MUTED, SURFACE, WASH, plt

W, H_IN, DPI = 8.0, 10.0, 135
LEFT, RIGHT, TOP = 0.075, 0.925, 0.935
BOTTOM, FLOOR = 0.055, 0.05  # the footer's last baseline; the lowest any content may reach
FLASH, LITE = "gemini-3.8-flash", "gemini-3.5-flash-lite"
AUTHOR = "Óscar García López"
LINK = "Code, data and the full derivation are linked in the first comment."
REPO = "github.com/oscargarcialopez/llm-double-auction"
TITLE, BODY, SMALL, EQ = 23, 15, 12, 20
LOST = "#cfcfcf"  # the share no rule can save: one step darker than the wash
QUOTE_BG = "#f2f2f2"

# The story order: Flash in each prompt, then Flash-Lite.
ORDER = ["B", "D", "E", "A", "C"]
SHORT = {"B": "Flash", "D": "Flash as a shopping assistant",
         "E": "Flash with p = (b + 2s)/3", "A": "Flash-Lite",
         "C": "Flash-Lite as a shopping assistant"}

# Two sentences from Flash's thought summaries: (condition, role, value, opening words).
QUOTES = {"recall": ("D", "buyer", 26.38, "Okay, so I'm looking at a classic double auction"),
          "derive": ("E", "buyer", 10.82, "With some algebra")}


def sans(size: float, weight: str = "normal") -> FontProperties:
    return FontProperties(family="IBM Plex Sans", size=size, weight=weight)


def mono(size: float) -> FontProperties:
    return FontProperties(family="DejaVu Sans Mono", size=size)


def keep(formula: str) -> str:
    """Join a formula with no-break spaces so the word wrap never splits it."""
    return formula.replace(" ", "\u00a0")


def load_quotes(results_dir: Path) -> dict[str, str]:
    """The quoted sentences, verbatim from the cache but for markdown marks."""
    rows = [json.loads(line) for line in (Path(results_dir) / "raw" / "calls.jsonl").open()]
    out = {}
    for name, (cond, role, value, start) in QUOTES.items():
        text = next(r["thoughts"] for r in rows if r["condition"] == cond
                    and r["role"] == role and r["value"] == value and start in r["thoughts"])
        sentence = re.match(r"[^\n]*?\.(?=\s|$)", text[text.index(start):]).group(0)
        out[name] = re.sub(r"[`*]|\\(?=_)", "", sentence)
    return out


class Slide:
    """A slide laid out top to bottom: each block is drawn at the cursor, measured, and the
    cursor moves below it. Text is wrapped by measuring it and set on a baseline grid; `alt`
    collects the slide's text in reading order."""

    def __init__(self):
        self.fig = plt.figure(figsize=(W, H_IN), dpi=DPI)
        self.fig.patch.set_facecolor(SURFACE)
        self.renderer = self.fig.canvas.get_renderer()
        self.alt: list[str] = []
        self.y = TOP
        self.floor = FLOOR

    # --- measuring ---------------------------------------------------------------------

    def _w(self, s: str, prop: FontProperties) -> float:
        return self.renderer.get_text_width_height_descent(s, prop, ismath=False)[0]

    def _metrics(self, prop: FontProperties, leading: float) -> tuple[float, float, float]:
        """Cap height, descent and baseline step, as fractions of the slide height."""
        _, h, d = self.renderer.get_text_width_height_descent("Hg", prop, ismath=False)
        px = self.fig.bbox.height
        return (h - d) / px, d / px, leading * prop.get_size_in_points() * DPI / 72 / px

    def _lines(self, text: str, prop: FontProperties, width: float) -> list[str]:
        """Greedy word wrap to `width` (a fraction of the slide width)."""
        px, lines = width * self.fig.bbox.width, []
        for para in text.split("\n"):
            line = ""
            for word in para.split(" "):
                trial = f"{line} {word}" if line else word
                if line and self._w(trial, prop) > px:
                    lines.append(line)
                    line = word
                else:
                    line = trial
            lines.append(line)
        return lines

    def _set(self, lines: list[str], prop: FontProperties, color: str, x: float,
             leading: float) -> None:
        cap, desc, step = self._metrics(prop, leading)
        for i, line in enumerate(lines):
            self.fig.text(x, self.y - cap - i * step, line, fontproperties=prop, color=color,
                          va="baseline")
        self.y -= cap + (len(lines) - 1) * step + desc

    # --- blocks --------------------------------------------------------------------------

    def title(self, text: str, size: float = TITLE, gap: float = 0.035) -> None:
        prop = sans(size, "semibold")
        self._set(self._lines(text, prop, RIGHT - LEFT), prop, INK, LEFT, 1.18)
        self.y -= gap
        self.alt.append(text)

    def para(self, text: str, size: float = BODY, color: str = INK_2, gap: float = 0.022,
             x: float = LEFT, width: float | None = None, weight: str = "normal",
             leading: float = 1.45) -> None:
        prop = sans(size, weight)
        self._set(self._lines(text, prop, width or RIGHT - x), prop, color, x, leading)
        self.y -= gap
        self.alt.append(text)

    def bullet(self, text: str, size: float = BODY, gap: float = 0.014,
               mark: str = "•") -> None:
        cap, _, _ = self._metrics(sans(size), 1.45)
        self.fig.text(LEFT + 0.004, self.y - cap, mark, fontproperties=sans(size), color=INK_2,
                      va="baseline")
        self.para(text, size=size, x=LEFT + 0.035, gap=gap)

    def quote(self, label: str, text: str, gap: float = 0.026) -> None:
        """A sentence of model output in a grey box, with a small label above it."""
        self.para(label, size=11, color=MUTED, gap=0.012)
        pad_x, pad_y = 0.022, 0.016
        top = self.y
        self.y -= pad_y
        prop = mono(12)
        self._set(self._lines(text, prop, RIGHT - LEFT - 2 * pad_x), prop, INK, LEFT + pad_x,
                  1.5)
        self.y -= pad_y
        self.fig.add_artist(Rectangle((LEFT, self.y), RIGHT - LEFT, top - self.y,
                                      transform=self.fig.transFigure, facecolor=QUOTE_BG,
                                      edgecolor="none", zorder=0))
        self.y -= gap
        self.alt.append(f"“{text}”")

    def math(self, tex: str, size: float = EQ, gap: float = 0.024, x: float = LEFT) -> None:
        t = self.fig.text(x, self.y, tex, fontsize=size, color=INK, va="top",
                          math_fontfamily="cm")
        self.y -= t.get_window_extent(self.renderer).height / self.fig.bbox.height + gap
        self.alt.append(tex)

    def axes(self, height: float, left: float = LEFT, right: float = RIGHT,
             gap: float = 0.03):
        ax = self.fig.add_axes((left, self.y - height, right - left, height))
        ax.set_facecolor(SURFACE)
        self.y -= height + gap
        return ax

    def beside(self, height: float, text: str, ylabel: str, xlabel: str,
               gap: float = 0.03):
        """A square 0-100 € plot at the left with a paragraph to its right. The y label sits
        above the plot; the cursor moves below the taller of the two."""
        top, side, left = self.y, height * H_IN / W, LEFT + 0.055
        ax = self.fig.add_axes((left, top - 0.03 - height, side, height))
        ax.set_xlim(0, th.H)
        ax.set_ylim(0, th.H)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.tick_params(labelsize=11)
        ax.grid(False)
        ax.set_xlabel(xlabel, fontsize=11.5, labelpad=4)
        ax.text(0, 1.04, ylabel, fontsize=11.5, color=INK_2, transform=ax.transAxes,
                va="bottom")
        self.para(text, x=left + side + 0.055, gap=0)
        self.y = min(self.y, top - 0.03 - height - 0.065) - gap
        return ax

    def footer(self, text: str) -> None:
        """Small grey text whose last line sits on the bottom margin. It leaves the cursor
        where it was and raises the floor the content above has to clear."""
        prop = sans(SMALL)
        lines = self._lines(text, prop, RIGHT - LEFT)
        cap, _, step = self._metrics(prop, 1.45)
        content, self.y = self.y, BOTTOM + cap + (len(lines) - 1) * step
        self.floor = self.y + 0.02
        self._set(lines, prop, MUTED, LEFT, 1.45)
        self.y = content
        self.alt.append(text)


def bar_rows(ax, rows, thickness_px: float = 18):
    """Horizontal bars, one per row, with the label above each bar on a patch of white so
    gridlines pass behind it. Rows are (y, value, colour, label); returns the bar height in
    data units."""
    ax.figure.canvas.draw()
    height = thickness_px / (ax.bbox.height / np.ptp(ax.get_ylim()))
    for y, value, color, label in rows:
        ax.barh(y, value, height=height, color=color, lw=0)
        ax.text(0, y + height / 2 + 0.06, label, fontproperties=sans(13), color=INK,
                va="bottom", transform=ax.get_yaxis_transform(), zorder=5,
                bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none"})
    return height


# --- the slides ----------------------------------------------------------------------------


def slide_cover(s: dict, f: dict, q: dict) -> Slide:
    sl = Slide()
    sl.para(AUTHOR, size=SMALL, color=MUTED, gap=0.03)
    sl.title("Gemini agents vs. the Myerson–Satterthwaite bound", size=28, gap=0.03)
    sl.para("I put Gemini agents on both sides of a sealed-bid double auction, with values "
            "uniform on 0–100 €. When both sides bid strategically, no trading rule can "
            f"realise more than {f['frontier']} of the gains (Myerson and Satterthwaite, "
            "1983).", gap=0.06)
    rows = [("B", "Gemini 3.8 Flash"), ("A", "Gemini 3.5 Flash-Lite"),
            ("C", "Gemini 3.5 Flash-Lite as a shopping assistant")]
    ax = sl.axes(0.38)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.5, len(rows) - 0.1)
    ax.axis("off")
    ys = np.arange(len(rows))[::-1] * 1.0
    h = bar_rows(ax, [(y, s["conditions"][n]["efficiency"]["efficiency"],
                       style.MODEL_COLOR[s["conditions"][n]["model"]], label)
                      for y, (n, label) in zip(ys, rows, strict=True)])
    for y, (n, _) in zip(ys, rows, strict=True):
        e = s["conditions"][n]["efficiency"]["efficiency"]
        ax.text(e + 0.015, y, f[f"{n}.eff"], fontproperties=sans(15, "semibold"), color=INK,
                va="center")
    fr = s["theory"]["ms_frontier"]
    ax.plot([fr, fr], [ys[-1] - h, ys[0] + 0.6], color=INK, lw=1)
    ax.text(fr, ys[0] + 0.64, f"bound {f['frontier']}", fontproperties=sans(SMALL),
            color=INK_2, ha="center", va="bottom")
    sl.para("Share of the available gains each pair of agents realised.", size=SMALL,
            color=MUTED)
    sl.footer(f"{f['calls']} bids and asks, {f['cost_usd']} of API calls.\n{LINK}")
    return sl


def slide_theorem(s: dict, f: dict, q: dict) -> Slide:
    sl = Slide()
    sl.title("The theorem")
    sl.para("A buyer values an item at v and the seller's cost is c. Each knows only their "
            "own number. Myerson and Satterthwaite (1983) showed that no trading rule is at "
            "the same time", gap=0.02)
    for text in (f"efficient (every deal with {keep('v > c')} happens),",
                 "incentive compatible (it works when both sides bid strategically),",
                 "voluntary (nobody expects to lose by taking part),",
                 "and budget-balanced (no money comes from outside)."):
        sl.bullet(text)
    sl.y -= 0.015
    sl.para(f"With v and c uniform on {keep('[0, 100]')}, the best a rule with the last "
            "three properties can do is", gap=0.02)
    sl.math(r"$\dfrac{\mathrm{gains\ realised}}{\mathrm{gains\ available}}\;\leq\;"
            r"\dfrac{9/64}{1/6}\;=\;\dfrac{27}{32}\;\approx\;" + f["frontier"].replace(
                "%", r"\%") + "$", gap=0.035)
    ax = sl.beside(0.27, f"The missing {f['lost_at_frontier']} comes from private "
                   "information. Each side shades its number to get a better price, and some "
                   "deals worth making fall through. The best such rule trades only when "
                   f"{keep('v − c ≥ 25')}, above the solid line.", "buyer's value v (€)",
                   "seller's cost c (€)")
    h, t = th.H, th.eq_trade_gap()
    ax.fill([0, 0, h - t], [t, h, h], color=WASH, lw=0)
    ax.fill([0, 0, h - t, h], [0, t, h, h], color=LOST, lw=0)
    ax.plot([0, h], [0, h], color=INK_2, lw=0.8)
    ax.plot([0, h - t], [t, h], color=INK, lw=1.4)
    ax.text(6, 86, "trade", fontproperties=sans(12), color=INK)
    ax.text(45, 45 + t / 2, "worth making, but lost", fontproperties=sans(11),
            color=INK, rotation=45, rotation_mode="anchor", ha="center", va="center")
    ax.text(94, 8, "nothing to gain", fontproperties=sans(11), color=MUTED, ha="right")
    return sl


def slide_mechanism(s: dict, f: dict, q: dict) -> Slide:
    sl = Slide()
    sl.title("The auction")
    sl.para("Buyer and seller each submit a sealed number, a bid b and an ask s. If "
            f"{keep('b ≥ s')} they trade at {keep('p = (b + s)/2')}.", gap=0.03)
    sl.para("Against asks spread evenly above a lowest ask, the buyer's best bid is",
            gap=0.018)
    sl.math(r"$\max_{b}\;\int_{s_{\min}}^{b}\left(v-\dfrac{b+s}{2}\right)\mathrm{d}s"
            r"\;\;\Rightarrow\;\; b=\dfrac{2v+s_{\min}}{3}$", gap=0.03)
    sl.para("Doing the same for the seller and solving both gives the linear equilibrium of "
            "Chatterjee and Samuelson (1983)", gap=0.018)
    sl.math(r"$b(v)=\dfrac{2}{3}\,v+\dfrac{25}{3}\qquad s(c)=\dfrac{2}{3}\,c+25$",
            gap=0.035)
    ax = sl.beside(0.27, "Wherever a trade is possible, the buyer bids below their value "
                   "and the seller asks above their cost. They trade only when "
                   f"{keep('v − c ≥ 25')}, the solid line on the previous slide, so this "
                   "simple auction loses just the "
                   f"{f['lost_at_frontier']} that any such rule must lose.", "bid or ask (€)",
                   "value v or cost c (€)")
    x = np.linspace(0, th.H, 101)
    ax.plot(x, x, color=MUTED, lw=0.9, ls=(0, (1, 2.5)))
    ax.plot(x, th.eq_buyer_bid(x), color=INK, lw=1.4)
    ax.plot(x, th.eq_seller_ask(x), color=INK, lw=1.4)
    ax.text(12, th.eq_seller_ask(12) + 4, "$s(c)$", fontsize=14, color=INK, ha="center",
            va="bottom")
    ax.text(88, th.eq_buyer_bid(88) - 4, "$b(v)$", fontsize=14, color=INK, ha="center",
            va="top")
    ax.text(83, 89, "truthful", fontproperties=sans(11), color=MUTED, rotation=45,
            rotation_mode="anchor", ha="center", va="center")
    return sl


def slide_why(s: dict, f: dict, q: dict) -> Slide:
    sl = Slide()
    sl.title("Why 27/32")
    sl.para(f"Let {keep('g = v − c')}. For {keep('g ≥ 0')} its density is "
            f"{keep('(100 − g)/100²')}, so if only the deals with {keep('g ≥ t')} go through, "
            "the expected gains are", gap=0.018)
    sl.math(r"$G(t)=\int_t^{100} g\,\dfrac{100-g}{100^2}\,\mathrm{d}g"
            r"\;=\;\dfrac{(100-t)^2\,(100+2t)}{6\cdot 100^2}$", gap=0.04)
    ax = sl.axes(0.27, left=LEFT + 0.01, right=RIGHT - 0.01, gap=0.09)
    g = np.linspace(0, th.H, 401)
    dens = g * (th.H - g) / th.H**2
    t = th.eq_trade_gap()
    ax.fill_between(g, dens, where=g >= t, color=WASH, lw=0)
    ax.fill_between(g, dens, where=g <= t, color=LOST, lw=0)
    ax.plot(g, dens, color=INK, lw=1.6)
    ax.plot([t, t], [0, t * (th.H - t) / th.H**2], color=INK, lw=0.9)
    ax.text(60, 0.1, f"kept {f['frontier']}", fontproperties=sans(13), color=INK,
            ha="center")
    ax.annotate(f"lost {f['lost_at_frontier']}", xy=(15, 0.055), xytext=(2, 0.235),
                fontproperties=sans(13), color=INK, ha="left", va="top",
                arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8,
                            "shrinkA": 4, "shrinkB": 0})
    ax.set_xlim(0, th.H)
    ax.set_ylim(0, 0.27)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("g = v − c (€), each deal weighted by its gap", fontsize=11.5, labelpad=6)
    sl.para(f"With no threshold {keep('G(0) = ' + f['first_best_eur'])}. The equilibrium "
            f"threshold is {keep('t = 25')}, with {keep('G(25) = ' + f['eq_gains_eur'])}, so",
            gap=0.018)
    sl.math(r"$\dfrac{G(25)}{G(0)}\;=\;\left(\dfrac{3}{4}\right)^{2}\cdot\dfrac{3}{2}"
            r"\;=\;\dfrac{27}{32}$")
    return sl


def slide_bids(s: dict, f: dict, q: dict) -> Slide:
    sl = Slide()
    sl.title("Bids against the equilibrium")
    sl.para("Mean bid or ask per 5 € band of values or costs.", size=SMALL, color=MUTED,
            gap=0.045)
    series = [("B", FLASH, False), ("A", LITE, False), ("C", LITE, True)]
    top = sl.y
    for j, role in enumerate(("buyer", "seller")):
        sl.y = top
        left = LEFT + 0.05 + j * 0.45
        ax = sl.axes(0.31, left=left, right=left + 0.37, gap=0.06)
        x = np.linspace(0, th.H, 101)
        eq = th.eq_buyer_bid(x) if role == "buyer" else th.eq_seller_ask(x)
        ax.plot(x, x, color=MUTED, lw=0.9, ls=(0, (1, 2.5)))
        ax.plot(x, eq, color=INK, lw=1.4)
        for name, model, hollow in series:
            st = s["conditions"][name][role]["strata"]
            color = style.MODEL_COLOR[model]
            ax.plot([b["value_mean"] for b in st], [b["amount_mean"] for b in st],
                    ls="", marker="o", ms=6, mew=1.4 if hollow else 0.8,
                    mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE)
        ax.set_xlim(0, th.H)
        ax.set_ylim(0, th.H)
        ax.set_aspect("equal")
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.tick_params(labelsize=11)
        ax.set_xlabel("buyer's value (€)" if role == "buyer" else "seller's cost (€)",
                      fontsize=11.5, labelpad=4)
        ax.text(0, 1.04, "bid (€)" if role == "buyer" else "ask (€)", fontsize=11.5,
                color=INK_2, transform=ax.transAxes, va="bottom")
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=7, mfc=style.MODEL_COLOR[FLASH], mec=SURFACE,
                   label="Flash"),
        plt.Line2D([], [], ls="", marker="o", ms=7, mfc=style.MODEL_COLOR[LITE], mec=SURFACE,
                   label="Flash-Lite"),
        plt.Line2D([], [], ls="", marker="o", ms=7, mfc=SURFACE, mec=style.MODEL_COLOR[LITE],
                   mew=1.4, label="Flash-Lite as a shopping assistant"),
        plt.Line2D([], [], color=INK, lw=1.4, label="equilibrium"),
        plt.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2.5)),
                   label="bid = value, ask = cost"),
    ]
    sl.y -= 0.012
    sl.fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(LEFT - 0.008, sl.y),
                  ncol=3, prop=sans(11.5), frameon=False, handlelength=1.6,
                  columnspacing=1.4, handletextpad=0.5, borderaxespad=0)
    sl.alt += [h.get_label() for h in handles]
    sl.y -= 0.075
    sl.bullet(f"Flash put {f['flash_k_half.at_eq_trade_region']} tradeable bids and asks "
              "exactly on the equilibrium.")
    sl.bullet("Flash-Lite sellers ask "
              f"{keep(f['A.seller.slope'] + 'c + ' + f['A.seller.intercept'])} instead of "
              f"{keep('0.67c + 25')}.")
    sl.bullet("As a shopping assistant its asks follow "
              f"{keep(f['C.seller.slope'] + 'c + ' + f['C.seller.intercept'])}, and "
              f"{f['C.seller.ir_binding']} were below cost yet low enough to be accepted.")
    return sl


def slide_efficiency(s: dict, f: dict, q: dict) -> Slide:
    sl = Slide()
    sl.title("Efficiency")
    sl.para("Share of the available gains realised, every buyer call against every seller "
            "call, with 95% intervals.", size=SMALL, color=MUTED, gap=0.05)
    ax = sl.axes(0.55, left=LEFT, right=RIGHT - 0.12, gap=0.045)
    ys = np.array([5.4, 4.4, 3.4, 1.6, 0.6])  # a wider gap between the two models
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6.3)
    ax.axis("off")
    rows = [(y, s["conditions"][n]["efficiency"]["efficiency"],
             style.MODEL_COLOR[s["conditions"][n]["model"]], SHORT[n])
            for y, n in zip(ys, ORDER, strict=True)]
    bar_rows(ax, rows)
    for y, name in zip(ys, ORDER, strict=True):
        lo, hi = s["conditions"][name]["efficiency"]["efficiency_ci"]
        ax.plot([lo, hi], [y, y], color=INK, lw=1.3, solid_capstyle="butt")
        ax.text(1.03, y, f[f"{name}.eff"], fontproperties=sans(14, "semibold"), color=INK,
                va="center", transform=ax.get_yaxis_transform())
    fr = s["theory"]["ms_frontier"]
    ax.plot([fr, fr], [0.25, 6.0], color=INK, lw=1)
    ax.text(fr, 6.05, f"bound {f['frontier']}", fontproperties=sans(SMALL), color=INK_2,
            ha="center", va="bottom")
    sl.para(f"Equilibrium play on the same draws realises {f['B.eq_same_draws']}, the same "
            "as Flash.")
    return sl


def slide_recall(s: dict, f: dict, q: dict) -> Slide:
    sl = Slide()
    sl.title("Changing the price rule")
    sl.quote("Flash, as a shopping assistant buying a used camera lens", q["recall"])
    sl.para(f"Flash named Chatterjee–Samuelson in {f['flash_k_half.named']} calls. So I "
            f"changed the price to {keep('p = (b + 2s)/3')}, which moves the equilibrium to",
            gap=0.018)
    sl.math(r"$b(v)=\dfrac{3}{4}\,v+\dfrac{25}{3}\qquad s(c)=\dfrac{3}{5}\,c+\dfrac{100}{3}$",
            gap=0.032)
    sl.quote("Flash, with the new price rule", q["derive"])
    ax = sl.beside(0.22, f"{f['E.at_eq_trade_region']} tradeable bids and asks landed on the "
                   f"new equilibrium and {f['E.at_k_half']} on the old one. Derivatives or "
                   f"first-order conditions appear in {f['E.derivation_terms']} summaries, "
                   f"against {f['flash_k_half.derivation_terms']} before, and it realised "
                   f"{f['E.eff']} of the gains against {f['k13_eff']} in theory.", "bid (€)",
                   "buyer's value (€)")
    x = np.linspace(0, th.H, 101)
    ax.plot(x, th.eq_buyer_bid(x, k=1 / 3), color=INK, lw=1.4, label="new equilibrium")
    ax.plot(x, th.eq_buyer_bid(x), color=INK_2, lw=1.1, ls=(0, (4, 2.5)),
            label="old equilibrium")
    st = s["conditions"]["E"]["buyer"]["strata"]
    ax.plot([b["value_mean"] for b in st], [b["amount_mean"] for b in st], ls="", marker="o",
            ms=5.5, mfc=style.MODEL_COLOR[FLASH], mec=SURFACE, mew=0.8,
            label="Flash, mean bid")
    legend = ax.legend(loc="upper left", prop=sans(10.5), handlelength=1.8,
                       borderaxespad=0.1)
    sl.alt += [t.get_text() for t in legend.get_texts()]
    return sl


def slide_exploit(s: dict, f: dict, q: dict) -> Slide:
    sl = Slide()
    sl.title("Exploitability")
    sl.para("I fitted the exact best response to each agent's asks on half of the calls and "
            "scored it on the other half. The bars show what that rival buyer gains per "
            "auction over playing the equilibrium.", gap=0.05)
    ax = sl.axes(0.36, left=LEFT, right=RIGHT - 0.16, gap=0.085)
    ys = np.array([5.4, 4.4, 3.4, 1.6, 0.6])
    ax.set_xlim(-0.3, 0.95)
    ax.set_ylim(0, 6.2)
    rows = []
    for y, name in zip(ys, ORDER, strict=True):
        c = s["conditions"][name]
        rows.append((y, c["exploitability"]["vs_model_seller"]["br_gain_over_eq_play"],
                     style.MODEL_COLOR[c["model"]], SHORT[name]))
    h = bar_rows(ax, rows)
    for y, name in zip(ys, ORDER, strict=True):
        x = s["conditions"][name]["exploitability"]["vs_model_seller"]
        lo, hi = x["br_gain_over_eq_play_ci"]
        ax.plot([lo, hi], [y, y], color=INK, lw=1.3, solid_capstyle="butt")
        ax.text(1.04, y, f[f"{name}.exploit_seller.gain"] + " €",
                fontproperties=sans(14, "semibold"), color=INK, va="center",
                transform=ax.get_yaxis_transform())
    ax.plot([0, 0], [ys[-1] - h, ys[0] + h], color=INK_2, lw=0.9)
    ax.set_yticks([])
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else fx.signed(v, 1)))
    ax.tick_params(labelsize=11)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("€ per auction, with 95% intervals", fontsize=11.5, labelpad=6)
    worst_buyer = max((s["conditions"][n]["exploitability"]["vs_model_buyer"]
                       ["br_gain_over_eq_play"], n) for n in ORDER)[1]
    sl.para("A best response gains nothing against Flash. Against the shopping-assistant "
            f"Flash-Lite it gains {keep(f['C.exploit_seller.gain'] + ' €')} per auction "
            f"{keep(f['C.exploit_seller.gain_ci'])}, "
            f"{f['C.exploit_seller.gain_rel'].lstrip('+')} more than equilibrium play. As "
            "buyers, every agent was harder to exploit (at most "
            f"{keep(f[worst_buyer + '.exploit_buyer.gain'] + ' €')}).")
    return sl


def slide_takeaways(s: dict, f: dict, q: dict) -> Slide:
    sl = Slide()
    sl.title("What I'd check", gap=0.04)
    for i, text in enumerate((
        f"Benchmark trading agents against the frontier, {f['frontier']} here, rather than "
        "100%. A pair above it is not in equilibrium, and a smarter rival can exploit that.",
        "Use variants of famous games to tell recall from reasoning.",
        f"Test the framing you will ship. Flash-Lite fell from {f['A.eff']} to {f['C.eff']} "
        "as a shopping assistant.",
        "Measure exploitability with held-out best responses, and block dominated moves "
        "such as selling below cost.",
    ), start=1):
        sl.bullet(text, gap=0.04, mark=f"{i}.")
    sl.y -= 0.03
    sl.para("Code, data and the full derivation of 27/32:", gap=0.01)
    sl.para(REPO, color=INK, weight="medium", gap=0.016)
    sl.para("Everything reruns from the cached calls, without an API key.", size=SMALL,
            color=MUTED)
    sl.footer(AUTHOR)
    return sl


SLIDES = [slide_cover, slide_theorem, slide_mechanism, slide_why, slide_bids,
          slide_efficiency, slide_recall, slide_exploit, slide_takeaways]


def make_carousel(results_dir: Path, carousel_dir: Path) -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # font timestamps, harmless
    style.apply()
    summary = style.load_summary(results_dir)
    facts = fx.build(summary)
    quotes = load_quotes(results_dir)
    carousel_dir = Path(carousel_dir)
    carousel_dir.mkdir(parents=True, exist_ok=True)
    for old in carousel_dir.glob("slide-*.png"):
        old.unlink()
    pdf_path = carousel_dir / "carousel.pdf"
    text = []  # the slides as plain text: a text alternative, and input for `auction check`
    with PdfPages(pdf_path, metadata={"Title": "Gemini agents and the "
                                               "Myerson-Satterthwaite bound",
                                      "CreationDate": None}) as pdf:
        for i, make in enumerate(SLIDES, start=1):
            sl = make(summary, facts, quotes)
            if sl.y < sl.floor:
                print(f"  warning: slide {i} overflows (cursor at {sl.y:.3f}, floor "
                      f"{sl.floor:.3f})")
            sl.fig.savefig(carousel_dir / f"slide-{i:02d}.png", dpi=DPI)
            pdf.savefig(sl.fig)
            labels = [t.get_text() for ax in sl.fig.axes for t in ax.texts]
            text.append(f"## Slide {i}\n\n" + "\n\n".join(
                t for t in sl.alt + labels if t.strip()))
            plt.close(sl.fig)
    (carousel_dir / "slides.txt").write_text("\n\n".join(text) + "\n")
    print(f"-> {pdf_path} ({len(SLIDES)} slides)")
