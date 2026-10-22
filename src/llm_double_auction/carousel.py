"""The LinkedIn carousel: eight 4:5 slides (1080 x 1350 px), one PDF plus PNG previews.

Every number on the slides comes from results/summary.json through facts.build(); theory
constants come from theory.py.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

from . import facts as fx
from . import style
from . import theory as th
from .style import INK, INK_2, MUTED, SURFACE, WASH, plt

W, H_IN, DPI = 8.0, 10.0, 135
LEFT, RIGHT = 0.075, 0.925
TOTAL = 8
FLASH, LITE = "gemini-3.8-flash", "gemini-3.5-flash-lite"
FOOTER = "Gemini agents in a sealed-bid double auction · code & data in the first comment"

# The story order: Flash in each prompt, then Flash-Lite.
ORDER = ["B", "D", "E", "A", "C"]
SHORT = {"B": "Flash · auction prompt", "D": "Flash · shopping-assistant prompt",
         "E": "Flash · auction prompt, k = 1/3", "A": "Flash-Lite · auction prompt",
         "C": "Flash-Lite · shopping-assistant prompt"}


def wrap(text: str, size: float) -> str:
    """Wrap a paragraph to the slide's text width for a given font size (DejaVu Sans)."""
    width = int((RIGHT - LEFT) * W * 72 / (size * 0.56))
    return "\n".join(textwrap.fill(p, width) for p in text.split("\n"))


class Slide:
    """A slide laid out top to bottom: each block is placed at the cursor, measured, and
    the cursor moves below it."""

    def __init__(self, n: int, kicker: str):
        self.fig = plt.figure(figsize=(W, H_IN), dpi=DPI)
        self.fig.patch.set_facecolor(SURFACE)
        self.renderer = self.fig.canvas.get_renderer()
        self.fig.text(LEFT, 0.955, kicker.upper(), fontsize=11, color=MUTED, weight="bold",
                      va="top")
        self.fig.text(RIGHT, 0.955, f"{n} / {TOTAL}", fontsize=11, color=MUTED, ha="right",
                      va="top")
        self.fig.text(LEFT, 0.03, FOOTER, fontsize=9.5, color=MUTED, va="bottom")
        self.y = 0.905

    def _place(self, t, gap: float):
        bb = t.get_window_extent(self.renderer)
        self.y -= bb.height / self.fig.bbox.height + gap
        return t

    def title(self, text: str, size: float = 29, gap: float = 0.03):
        t = self.fig.text(LEFT, self.y, text, fontsize=size, weight="bold", color=INK,
                          va="top", linespacing=1.12)
        return self._place(t, gap)

    def _grid(self, size: float, weight: str = "normal", linespacing: float = 1.38):
        """First baseline, baseline step and x-height of a paragraph, as fractions of the
        slide height. Every line gets a box linespacing x size tall (DejaVu's ascent and
        descent add up to one em), with the baseline where matplotlib puts it on a line
        with both an ascender and a descender. Matplotlib itself centres each line's own
        glyphs in its box, so a line without descenders sits lower than its neighbours."""
        props = FontProperties(size=size, weight=weight)
        _, h, d = self.renderer.get_text_width_height_descent("lp", props, ismath=False)
        _, x_height, _ = self.renderer.get_text_width_height_descent("x", props, ismath=False)
        step = linespacing * size * DPI / 72
        px = self.fig.bbox.height
        return ((step - h) / 2 + h - d) / px, step / px, x_height / px

    def _lines(self, x: float, body: str, size: float, color: str,
               weight: str = "normal") -> float:
        """Draw a paragraph on the baseline grid and return its height."""
        first, step, _ = self._grid(size, weight)
        lines = body.split("\n")
        for i, line in enumerate(lines):
            self.fig.text(x, self.y - first - i * step, line, fontsize=size, color=color,
                          weight=weight, va="baseline")
        return len(lines) * step

    def text(self, text: str, size: float = 16.5, color: str = INK_2, gap: float = 0.02,
             x: float = LEFT, weight: str = "normal", raw: bool = False):
        body = text if raw else wrap(text, size)
        self.y -= self._lines(x, body, size, color, weight) + gap

    def math(self, tex: str, size: float = 25, gap: float = 0.025, x: float = LEFT):
        t = self.fig.text(x, self.y, tex, fontsize=size, color=INK, va="top",
                          math_fontfamily="cm")
        return self._place(t, gap)

    def bullet(self, text: str, size: float = 16.5, color: str | None = None,
               hollow: bool = False, gap: float = 0.016, marker: str = "o"):
        """A paragraph with a hanging marker: a model-coloured dot, or a numeral."""
        indent = 0.045
        baseline, _, x_height = self._grid(size)
        if marker == "o":
            self.fig.add_artist(plt.Line2D(
                [LEFT + 0.012], [self.y - baseline + x_height / 2], marker="o", ms=9,
                mfc=SURFACE if hollow else color, mec=color, mew=2,
                transform=self.fig.transFigure))
        else:
            self.fig.text(LEFT, self.y - baseline, marker, fontsize=size, color=INK,
                          weight="bold", va="baseline")
        width = int((RIGHT - LEFT - indent) * W * 72 / (size * 0.56))
        body = "\n".join(textwrap.fill(p, width) for p in text.split("\n"))
        self.y -= self._lines(LEFT + indent, body, size, INK_2) + gap

    def axes(self, height: float, left: float = LEFT, right: float = RIGHT,
             gap: float = 0.03):
        ax = self.fig.add_axes((left, self.y - height, right - left, height))
        ax.set_facecolor(SURFACE)
        self.y -= height + gap
        return ax


# --- the slides ----------------------------------------------------------------------------


def slide_hook(s: dict, f: dict) -> Slide:
    sl = Slide(1, "Agentic commerce × mechanism design")
    sl.title("When two AI agents\nhaggle strategically,\na 1983 theorem caps\n"
             f"what they can capture:\n{f['frontier']} of the value.", size=29)
    sl.text("How close does Gemini get?", size=19, color=INK, gap=0.035)
    rows = [("B", "Gemini 3.8 Flash"), ("A", "Gemini 3.5 Flash-Lite"),
            ("C", "Flash-Lite, prompted as a shopping assistant")]
    ax = sl.axes(0.36, gap=0.04)
    ys = np.arange(len(rows))[::-1] * 1.0
    for y, (name, label) in zip(ys, rows, strict=True):
        c = s["conditions"][name]
        e = c["efficiency"]["efficiency"]
        ax.barh(y, e, height=0.34, color=style.MODEL_COLOR[c["model"]], lw=0)
        ax.text(0, y + 0.3, label, fontsize=15, color=INK, va="bottom")
        ax.text(e + 0.015, y, f[f"{name}.eff"], fontsize=24, weight="bold",
                color=INK, va="center")
    fr = s["theory"]["ms_frontier"]
    ax.plot([fr, fr], [-0.45, len(rows) - 0.25], color=INK, lw=1.3, ls=(0, (4, 3)))
    ax.text(fr, len(rows) - 0.2, f"ceiling {f['frontier']}", fontsize=12, color=INK,
            ha="center", va="bottom")
    ax.set_xlim(0, 1.13)
    ax.set_ylim(-0.5, len(rows) + 0.1)
    ax.axis("off")
    sl.text(f"{f['calls']} sealed bids from two Gemini models, {f['cost_usd']} of API calls. "
            "The math, the experiment and what it means for agents that buy and sell: "
            "swipe →", size=15)
    return sl


def slide_theorem(s: dict, f: dict) -> Slide:
    sl = Slide(2, "The theorem")
    sl.title("Myerson–Satterthwaite\n(1983)")
    sl.text("A buyer values an item at v; the seller's cost is c. Each knows only its own "
            "number. No trading rule can be all four at once:", gap=0.025)
    items = [("1", "Efficient: every deal with v > c happens."),
             ("2", "Manipulation-proof: it still works when both sides bid strategically "
                   "(Bayes–Nash equilibrium)."),
             ("3", "Voluntary: nobody expects to lose by taking part."),
             ("4", "Self-funded: no money from outside.")]
    for num, text in items:
        sl.bullet(text, marker=num)
    sl.y -= 0.015
    sl.text("With v and c uniform on 0–100 €, a rule that keeps 2–4 realises at most",
            gap=0.022)
    sl.math(r"$\dfrac{\mathrm{gains\ realised}}{\mathrm{gains\ available}}\;\leq\;"
            r"\dfrac{9/64}{1/6}\;=\;\dfrac{27}{32}$", size=26)
    sl.text(f"= {f['frontier']}. The missing {f['lost_at_frontier']} is the price of "
            "private information, not a modelling failure: no prompt or model can win it "
            "back under these rules.", size=16.5)
    return sl


def slide_mechanism(s: dict, f: dict) -> Slide:
    sl = Slide(3, "The game")
    sl.title("Split-the-difference\ndouble auction")
    sl.text("Buyer bids b, seller asks s, sealed and at the same time. If b ≥ s they trade "
            "at p = (b + s)/2.", gap=0.03)
    sl.text(r"Against asks spread evenly above $s_{\min}$, the buyer's best bid is",
            gap=0.012)
    sl.math(r"$\max_{b}\;\int_{s_{\min}}^{b}\left(v-\dfrac{b+s}{2}\right)\mathrm{d}s"
            r"\;\;\Rightarrow\;\; b=\dfrac{2v+s_{\min}}{3}$", size=23)
    sl.text("The seller's problem mirrors it. Solving both (Chatterjee & Samuelson, 1983):",
            gap=0.012)
    sl.math(r"$b(v)=\dfrac{2}{3}\,v+\dfrac{25}{3}\qquad s(c)=\dfrac{2}{3}\,c+25$", size=25)
    sl.text("so they trade exactly when", gap=0.01)
    sl.math(r"$v-c\;\geq\;25$", size=27, gap=0.03)
    sl.text("Deals with a gap under 25 € never happen. They hold exactly the "
            f"{f['lost_at_frontier']} the theorem says no rule can save, so this simple "
            "auction is the best possible one.")
    return sl


def slide_why(s: dict, f: dict) -> Slide:
    sl = Slide(4, "Why 27/32")
    sl.title(f"Where the other\n{f['lost_at_frontier']} goes")
    sl.text("The gap d = v − c of a random pair has a triangular density. Trading only when "
            "d ≥ t realises", gap=0.012)
    sl.math(r"$G(t)=\int_t^{100} d\;\dfrac{100-d}{100^2}\;\mathrm{d}d"
            r"\;=\;\dfrac{(100-t)^2\,(100+2t)}{6\cdot 100^2}$", size=22, gap=0.02)
    ax = sl.axes(0.25, left=LEFT + 0.02, gap=0.06)
    d = np.linspace(0, th.H, 401)
    g = d * (th.H - d) / th.H**2
    t = th.eq_trade_gap()
    ax.fill_between(d, g, where=d >= t, color=WASH, lw=0)
    ax.fill_between(d, g, where=d <= t, facecolor=SURFACE, edgecolor=INK_2, hatch="////",
                    lw=0)
    ax.plot(d, g, color=INK, lw=2)
    ax.axvline(t, color=INK, lw=1)
    ax.text(62, 0.11, f"kept: {f['frontier']}", fontsize=15, color=INK, ha="center")
    ax.text(1, 0.255, f"lost: {f['lost_at_frontier']}", fontsize=15, color=INK,
            ha="left", va="top")
    ax.set_xlim(0, th.H)
    ax.set_ylim(0, 0.27)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=12)
    ax.set_xlabel("gap d = v − c (€), weighted by what the deal is worth", fontsize=12.5)
    nb = "\u00a0"  # keep each equation on one line
    sl.text(f"First best G(0){nb}={nb}{f['first_best_eur'].replace(' ', nb)}, "
            f"equilibrium G(25){nb}={nb}{f['eq_gains_eur'].replace(' ', nb)}:", gap=0.012)
    sl.math(r"$\dfrac{G(25)}{G(0)}\;=\;\left(\dfrac{3}{4}\right)^{2}\cdot\dfrac{3}{2}"
            r"\;=\;\dfrac{27}{32}$", size=26)
    return sl


def slide_bids(s: dict, f: dict) -> Slide:
    sl = Slide(5, "What the agents bid")
    sl.title("Flash plays the\nequilibrium to the cent")
    series = [("B", FLASH, False), ("A", LITE, False), ("C", LITE, True)]
    top = sl.y
    axes = []
    for j, role in enumerate(("buyer", "seller")):
        sl.y = top
        left = LEFT + j * 0.45 + 0.04
        ax = sl.axes(0.3, left=left, right=left + 0.39, gap=0.05)
        axes.append(ax)
        x = np.linspace(0, th.H, 101)
        eq = th.eq_buyer_bid(x) if role == "buyer" else th.eq_seller_ask(x)
        ax.plot(x, x, color=MUTED, lw=1, ls=(0, (1, 2)))
        ax.plot(x, eq, color=INK, lw=1.6)
        for name, model, hollow in series:
            st = s["conditions"][name][role]["strata"]
            ax.plot([q["value_mean"] for q in st], [q["amount_mean"] for q in st],
                    ls="", marker="o", ms=5.5, mew=1.4,
                    mfc=SURFACE if hollow else style.MODEL_COLOR[model],
                    mec=style.MODEL_COLOR[model])
        ax.set_xlim(0, th.H)
        ax.set_ylim(0, th.H)
        ax.set_aspect("equal")
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.tick_params(labelsize=11)
        ax.set_title("buyer's bid vs value" if role == "buyer" else "seller's ask vs cost",
                     fontsize=13, loc="left", color=INK)
    axes[0].text(97, 62, "equilibrium", fontsize=11, color=INK, ha="right", va="bottom",
                 rotation=33.7)
    axes[0].text(60, 75, "bid = value", fontsize=11, color=MUTED, ha="right", va="bottom",
                 rotation=45)
    sl.y += 0.012
    sl.text("Dots: mean per €5 band of values or costs, over all calls.", size=11.5,
            color=MUTED, gap=0.022)
    sl.bullet(f"Flash: {f['flash_k_half.at_eq_trade_region']} bids and asks that could "
              "trade sat on the equilibrium, to the cent. Its reasoning summaries named "
              f"Chatterjee–Samuelson in {f['flash_k_half.named']} calls, even with the game "
              "told as selling a used camera lens.", color=style.MODEL_COLOR[FLASH],
              size=15)
    sl.bullet(f"Flash-Lite shades too hard: sellers ask {f['A.seller.slope']}c + "
              f"{f['A.seller.intercept']} instead of 0.67c + 25.", color=style.MODEL_COLOR[LITE],
              size=15)
    sl.bullet(f"As a shopping assistant its asks barely follow its cost ({f['C.seller.slope']}c "
              f"+ {f['C.seller.intercept']}), and {f['C.seller.ir_binding']} were below its "
              "cost where a trade could happen.", color=style.MODEL_COLOR[LITE], hollow=True,
              size=15)
    return sl


def slide_efficiency(s: dict, f: dict) -> Slide:
    sl = Slide(6, "Efficiency")
    sl.title("Flash sits on the\ntheoretical frontier")
    sl.text("Share of the available gains each pair of agents realised (every buyer call "
            "against every seller call, 95% CI):", size=15, gap=0.02)
    ax = sl.axes(0.37, left=LEFT, right=RIGHT - 0.13, gap=0.045)
    ys = np.arange(len(ORDER))[::-1] * 1.0
    for y, name in zip(ys, ORDER, strict=True):
        c = s["conditions"][name]
        e = c["efficiency"]
        ax.barh(y, e["efficiency"], height=0.36, color=style.MODEL_COLOR[c["model"]], lw=0)
        lo, hi = e["efficiency_ci"]
        ax.plot([lo, hi], [y, y], color=INK, lw=1.6, solid_capstyle="butt")
        ax.text(0, y + 0.25, SHORT[name], fontsize=12.5, color=INK, va="bottom")
        ax.text(1.02, y, f[f"{name}.eff"], fontsize=17, weight="bold", color=INK,
                va="center", transform=ax.get_yaxis_transform())
    fr = s["theory"]["ms_frontier"]
    ax.plot([fr, fr], [-0.4, len(ORDER) - 0.35], color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(fr, len(ORDER) - 0.3, f"bound {f['frontier']}", fontsize=11.5, color=INK,
            ha="center", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(ORDER) + 0.05)
    ax.axis("off")
    sl.text("Recall or reasoning? With the price at p = (b + 2s)/3 the equilibrium is no "
            "longer the famous one. Flash solved the new game: "
            f"{f['E.at_eq_trade_region']} tradeable bids and asks at the new equilibrium, "
            f"{f['E.at_k_half']} at the textbook numbers, {f['E.eff']} realised against "
            f"{f['k13_eff']} in theory. Its reasoning summaries mention first-order conditions "
            f"or derivatives in {f['E.derivation_terms']} calls, against "
            f"{f['flash_k_half.derivation_terms']} in the textbook game.", size=15)
    return sl


def slide_exploit(s: dict, f: dict) -> Slide:
    sl = Slide(7, "Exploitability")
    sl.title("Can a smarter rival\nexploit them?")
    sl.text("I computed the exact best response to each agent's asks on half of the calls "
            "and scored it on the other half. Bars: the rival buyer's gain per auction over "
            "playing the equilibrium itself (95% CI).", size=15, gap=0.035)
    ax = sl.axes(0.35, left=LEFT, right=RIGHT - 0.2, gap=0.08)
    ys = np.arange(len(ORDER))[::-1] * 1.0
    for y, name in zip(ys, ORDER, strict=True):
        c = s["conditions"][name]
        x = c["exploitability"]["vs_model_seller"]
        ax.barh(y, x["br_gain_over_eq_play"], height=0.36, color=style.MODEL_COLOR[c["model"]],
                lw=0)
        lo, hi = x["br_gain_over_eq_play_ci"]
        ax.plot([lo, hi], [y, y], color=INK, lw=1.6, solid_capstyle="butt")
        ax.text(-0.3, y + 0.25, SHORT[name], fontsize=12.5, color=INK, va="bottom", zorder=5,
                bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none"})
        ax.text(1.03, y, f[f"{name}.exploit_seller.gain"] + " €", fontsize=16, weight="bold",
                color=INK, va="center", transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_xlim(-0.3, 0.95)
    ax.set_ylim(-0.5, len(ORDER) - 0.2)
    ax.set_yticks([])
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else fx.signed(v, 1)))
    ax.tick_params(labelsize=11)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("€ per auction", fontsize=12)
    worst_buyer = max((s["conditions"][n]["exploitability"]["vs_model_buyer"]
                       ["br_gain_over_eq_play"], n) for n in ORDER)[1]
    sl.text("Flash leaves nothing to take. The shopping-assistant Flash-Lite seller hands a "
            f"rival buyer {f['C.exploit_seller.gain']} € per auction "
            f"{f['C.exploit_seller.gain_ci']}, {f['C.exploit_seller.gain_rel'].lstrip('+')} "
            "more than equilibrium play. As buyers, all agents were harder to exploit (at "
            f"most {f[worst_buyer + '.exploit_buyer.gain']} €).", size=15)
    return sl


def slide_takeaways(s: dict, f: dict) -> Slide:
    sl = Slide(8, "If you build agents that trade")
    sl.title("Four tests before\nyou ship")
    items = [
        ("1", f"Benchmark against the frontier, not 100%. Here it is {f['frontier']}: an "
              "agent pair above it is not in equilibrium, so one side could do better by "
              "bidding differently, and a smarter rival will."),
        ("2", "Change the rules to tell recall from reasoning. Famous games get recited: "
              "Flash named Chatterjee–Samuelson every time, so the test that counted was the "
              "k = 1/3 variant, which it also solved."),
        ("3", f"Test the framing you will ship. Flash-Lite fell from {f['A.eff']} to "
              f"{f['C.eff']} as a shopping assistant; Flash stayed at {f['B.eff']}."),
        ("4", "Measure exploitability with held-out best responses, and block dominated "
              f"moves: the cheap seller asked below its own cost {f['C.seller.ir_binding']} "
              "times where a trade could happen."),
    ]
    for num, text in items:
        sl.bullet(text, marker=num, size=16, gap=0.03)
    sl.y -= 0.01
    sl.text("Code, data and the full derivation: link in the first comment.", size=16,
            color=INK, weight="bold")
    return sl


SLIDES = [slide_hook, slide_theorem, slide_mechanism, slide_why, slide_bids,
          slide_efficiency, slide_exploit, slide_takeaways]


def make_carousel(results_dir: Path, carousel_dir: Path) -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # font timestamps, harmless
    style.apply()
    summary = style.load_summary(results_dir)
    facts = fx.build(summary)
    carousel_dir = Path(carousel_dir)
    carousel_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = carousel_dir / "carousel.pdf"
    text = []  # the slides as plain text: a text alternative, and input for `auction check`
    with PdfPages(pdf_path, metadata={"Title": "Gemini agents and the "
                                               "Myerson-Satterthwaite bound"}) as pdf:
        for i, make in enumerate(SLIDES, start=1):
            sl = make(summary, facts)
            if sl.y < 0.06:
                print(f"  warning: slide {i} overflows (cursor at {sl.y:.3f})")
            sl.fig.savefig(carousel_dir / f"slide-{i:02d}.png", dpi=DPI)
            pdf.savefig(sl.fig)
            texts = sl.fig.texts + [t for ax in sl.fig.axes for t in ax.texts]
            text.append(f"## Slide {i}\n\n" + "\n\n".join(
                t.get_text().replace("\n", " ") for t in texts if t.get_text().strip()))
            plt.close(sl.fig)
    (carousel_dir / "slides.txt").write_text("\n\n".join(text) + "\n")
    print(f"-> {pdf_path} ({len(SLIDES)} slides)")
