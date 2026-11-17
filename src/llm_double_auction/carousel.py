"""The LinkedIn carousel: nine 4:5 pages (6 x 7.5 in, previews at 1080 x 1350 px) set like a
paper. Times-like STIX Two type (vendored in fonts/), numbered sections and equations,
captioned figures, ruled tables and the references; the figures use the steel blue and dark
red of the author's thesis, one per model. Typst sets the pages and matplotlib draws the
figures as SVG.

Every number comes from results/summary.json through facts.build(); theory constants come
from theory.py, and the two quotes are read from the cached calls. carousel/slides.txt holds
the pages as plain text: a text alternative, and the input for `auction check`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import typst
from matplotlib import font_manager

from . import facts as fx
from . import style
from . import theory as th
from .style import plt

TITLE = "Gemini agents vs the Myerson–Satterthwaite bound"
AUTHOR = "Óscar García López"
DATE = "November 2026"
REPO = "github.com/oscargarcialopez/llm-double-auction"
FLASH, LITE = "gemini-3.8-flash", "gemini-3.5-flash-lite"

INK, GREY, FILL = "#000000", "#555555", "#e2e2e2"
COLOR = {FLASH: "#4682b4", LITE: "#8b0000"}  # steel blue and dark red, as in the thesis
TEXT_W = 4.9  # inches: a 6 in page with 0.55 in margins
SIDE = 2.2  # inches: a square plot with text beside it

# The story order: Flash in each prompt, then Flash-Lite.
ORDER = ["B", "D", "E", "A", "C"]
GROUPS = [("Gemini 3.8 Flash", [("B", "auction prompt"), ("D", "shopping assistant"),
                                ("E", "price $(b + 2s)\\/3$")]),
          ("Gemini 3.5 Flash-Lite", [("A", "auction prompt"), ("C", "shopping assistant")])]

# Two sentences from Flash's thought summaries: (condition, role, value, opening words).
QUOTES = {"recall": ("D", "buyer", 26.38, "Okay, so I'm looking at a classic double auction"),
          "derive": ("E", "buyer", 10.82, "With some algebra")}

REFERENCES = [
    ("Chatterjee, K., & Samuelson, W. (1983). Bargaining under incomplete information.",
     "Operations Research", "31(5), 835–851."),
    ("Myerson, R. B., & Satterthwaite, M. A. (1983). Efficient mechanisms for bilateral "
     "trading.", "Journal of Economic Theory", "29(2), 265–281."),
]

PREAMBLE = r"""
#set document(title: "@TITLE", author: "@AUTHOR", date: none)
#set page(
  width: 6in, height: 7.5in,
  margin: (x: 0.55in, top: 0.74in, bottom: 0.62in),
  header: context if here().page() > 1 {
    set text(9.5pt, style: "italic")
    grid(columns: (1fr, auto), [@TITLE], [@AUTHOR])
    v(-3pt)
    line(length: 100%, stroke: 0.4pt)
  },
  footer: context align(center, text(10pt)[#here().page()]),
)
#set text(font: "STIX Two Text", size: 13.5pt, lang: "en", region: "gb")
#show math.equation: set text(font: "STIX Two Math")
#set par(justify: true, leading: 0.5em, spacing: 0.78em)
#set heading(numbering: "1.")
#show heading: set text(15pt, weight: "bold")
#show heading: set block(above: 0pt, below: 0.8em)
#show heading.where(level: 2): set text(12.5pt)
#show heading.where(level: 2): set block(above: 1.1em, below: 0.6em)
#set math.equation(numbering: "(1)")
#set enum(indent: 0.4em, body-indent: 0.6em, spacing: 0.5em)
#set list(indent: 0.4em, body-indent: 0.6em, spacing: 0.5em, marker: [•])
#show list: set par(justify: false)
#show enum: set par(justify: false)
#let cap(kind, n, body, above: 0.55em) = block(above: above, below: 0.85em, {
  set text(11.5pt)
  set par(justify: true, leading: 0.42em)
  [#strong[#kind #n.] #body]
})
#let mquote(body, source) = pad(x: 1.1em, block(above: 0.65em, below: 0.75em, {
  set text(12.5pt)
  set par(justify: false, leading: 0.42em)
  emph[“#body”]
  linebreak()
  align(right, text(11pt)[— #source])
}))
#let ruled(cols, aligns, header, ..rows) = align(center, {
  set text(11.5pt)
  set par(justify: false)
  table(
    columns: cols, align: aligns, stroke: none, inset: (x: 5pt, y: 3.4pt),
    table.hline(stroke: 0.9pt),
    table.header(..header),
    table.hline(stroke: 0.5pt),
    ..rows,
    table.hline(stroke: 0.9pt),
  )
})
"""

MATH_DOLLAR = re.compile(r"(?<!\\)\$")


def _esc(text: str) -> str:
    """Plain text as Typst markup: escape what would start markup; a literal $ is \\$."""
    text = text.replace("\\$", "\x00")
    text = re.sub(r"([\\#*_@<>\[\]`~/=+-])", r"\\\1", text)
    return text.replace("\x00", "\\$")


def T(src: str) -> str:
    """Typst markup from text with $inline math$ (Typst math syntax)."""
    parts = MATH_DOLLAR.split(src)
    return "".join(_esc(p) if i % 2 == 0 else f"#box[${p}$]" for i, p in enumerate(parts))


def P(src: str) -> str:
    """The same text, plain: math loses its Typst escapes."""
    parts = MATH_DOLLAR.split(src)
    return "".join(p.replace("\\$", "$") if i % 2 == 0 else
                   p.replace("\\/", "/").replace("^2", "²").replace("dif ", "d")
                   for i, p in enumerate(parts))


def head(h: str) -> str:
    """A table header cell; a newline in it is a line break."""
    return "[" + " \\ ".join(T(part) for part in h.split("\n")) + "]"


def usd(x: str) -> str:
    return x.replace("$", "\\$")


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


class Page:
    """One page: Typst markup blocks and the same content as plain text, in reading order."""

    def __init__(self):
        self.typ: list[str] = []
        self.txt: list[str] = []

    def raw(self, typ: str, txt: str | list[str] = ()) -> None:
        self.typ.append(typ)
        self.txt += [txt] if isinstance(txt, str) else list(txt)

    def h(self, src: str) -> None:
        self.raw(f"= {T(src)}", P(src))

    def p(self, src: str) -> None:
        self.raw(T(src), P(src))

    def eq(self, math: str, plain: str) -> None:
        self.raw(f"$ {math} $", plain)

    def bullets(self, srcs: list[str]) -> None:
        self.raw("\n".join(f"- {T(s)}" for s in srcs), [P(s) for s in srcs])

    def figure(self, name: str, n: int, caption: str, labels: list[str]) -> None:
        self.raw(f'#align(center, image("{name}"))\n#cap("Figure", {n})[{T(caption)}]',
                 [f"Figure {n}. {P(caption)}"] + labels)

    def beside(self, name: str, n: int, caption: str, text: str, labels: list[str]) -> None:
        """A square plot at the left; its caption and a paragraph at the right."""
        self.raw(f'#grid(columns: (auto, 1fr), column-gutter: 13pt,\n'
                 f'  image("{name}"),\n'
                 f'  [#set par(justify: false)\n#v(2pt)#cap("Figure", {n})[{T(caption)}]'
                 f'\n\n{T(text)}])',
                 [f"Figure {n}. {P(caption)}", P(text)] + labels)

    def markup(self) -> str:
        return "\n\n".join(self.typ)


# --- figures --------------------------------------------------------------------------------


def paper_style() -> None:
    for font in sorted(style.FONTS.glob("STIX2Text-*.otf")):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": ["STIX Two Text", "STIXGeneral"],
        "font.size": 12,
        "mathtext.fontset": "stix",
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.labelsize": 12,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": INK,
        "legend.framealpha": 1.0,
        "legend.fontsize": 10.5,
        "patch.linewidth": 0.6,
        "hatch.linewidth": 0.5,
        "svg.fonttype": "path",
        "svg.hashsalt": "llm-double-auction",
    })


def texts(fig) -> list[str]:
    """The words on a figure that are not tick numbers: annotations, legends, axis labels and
    category ticks."""
    out = []
    for ax in fig.axes:
        out += [t.get_text() for t in ax.texts]
        out += [ax.get_xlabel(), ax.get_ylabel()]
        out += [t.get_text() for t in ax.get_yticklabels() if re.search("[A-Za-z]", t.get_text())]
        if ax.get_legend():
            out += [t.get_text() for t in ax.get_legend().get_texts()]
    for leg in fig.legends:
        out += [t.get_text() for t in leg.get_texts()]
    return [t.replace("\n", " ") for t in out if t.strip()]


def save(fig, path: Path) -> list[str]:
    labels = texts(fig)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.03,
                metadata={"Date": None})
    plt.close(fig)
    return labels


def square(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlim(0, th.H)
    ax.set_ylim(0, th.H)
    ax.set_aspect("equal")
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel(ylabel, labelpad=2)


def fig_overview(s: dict, f: dict, path: Path) -> list[str]:
    rows = [("B", "Gemini 3.8 Flash"), ("A", "Gemini 3.5 Flash-Lite"),
            ("C", "Gemini 3.5 Flash-Lite,\nshopping assistant")]
    fig, ax = plt.subplots(figsize=(TEXT_W - 0.1, 1.85))
    ys = np.arange(len(rows))[::-1].astype(float)
    for y, (n, _) in zip(ys, rows, strict=True):
        c = s["conditions"][n]
        e = 100 * c["efficiency"]["efficiency"]
        color = COLOR[c["model"]]
        hollow = n == "C"
        ax.barh(y, e, height=0.55, color="white" if hollow else color, edgecolor=color,
                hatch="////" if hollow else None, lw=0.8)
        ax.text(e + 2.2, y, f[f"{n}.eff"], va="center", fontsize=12)
    fr = 100 * s["theory"]["ms_frontier"]
    ax.axvline(fr, color=INK, lw=0.9, ls=(0, (4, 2.5)))
    ax.text(fr, 1.03, f"bound {f['frontier']}", ha="center", va="bottom", fontsize=11.5,
            transform=ax.get_xaxis_transform())
    ax.set_yticks(ys, [label for _, label in rows])
    ax.tick_params(axis="y", length=0, labelsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("share of the available gains realised (%)", labelpad=3)
    return save(fig, path)


def fig_regions(path: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(SIDE, SIDE))
    h, t = th.H, th.eq_trade_gap()
    ax.fill([0, 0, h - t], [t, h, h], color=FILL, lw=0)
    ax.fill([0, 0, h - t, h], [0, t, h, h], facecolor="white", edgecolor=GREY, hatch="////",
            lw=0)
    ax.plot([0, h], [0, h], color=INK, lw=0.6)
    ax.plot([0, h - t], [t, h], color=INK, lw=1.3)
    ax.text(6, 86, "trade", fontsize=11.5)
    ax.text(47, 47 + t / 2, "lost", fontsize=11.5, rotation=45, rotation_mode="anchor",
            ha="center", va="center",
            bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})
    ax.text(95, 6, "no gains", fontsize=11.5, ha="right")
    square(ax, "seller's cost $c$ (€)", "buyer's value $v$ (€)")
    return save(fig, path)


def fig_strategies(path: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(SIDE, SIDE))
    x = np.linspace(0, th.H, 101)
    ax.plot(x, x, color=GREY, lw=0.8, ls=(0, (1, 2)))
    ax.plot(x, th.eq_buyer_bid(x), color=INK, lw=1.2)
    ax.plot(x, th.eq_seller_ask(x), color=INK, lw=1.2)
    ax.text(14, th.eq_seller_ask(14) + 5, "$s(c)$", fontsize=13, ha="center", va="bottom")
    ax.text(86, th.eq_buyer_bid(86) - 5, "$b(v)$", fontsize=13, ha="center", va="top")
    square(ax, "value $v$ or cost $c$ (€)", "bid or ask (€)")
    return save(fig, path)


def fig_gains(f: dict, path: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(TEXT_W - 0.1, 2.2))
    g = np.linspace(0, th.H, 401)
    dens = g * (th.H - g) / th.H**2
    t = th.eq_trade_gap()
    ax.fill_between(g, dens, where=g >= t, color=FILL, lw=0)
    ax.fill_between(g, dens, where=g <= t, facecolor="white", edgecolor=GREY, hatch="////",
                    lw=0)
    ax.plot(g, dens, color=INK, lw=1.1)
    ax.plot([t, t], [0, t * (th.H - t) / th.H**2], color=INK, lw=0.8)
    ax.text(58, 0.095, f"kept {f['frontier']}", ha="center", fontsize=12)
    ax.annotate(f"lost {f['lost_at_frontier']}", xy=(17, 0.07), xytext=(3, 0.215),
                fontsize=12, ha="left", va="top",
                arrowprops={"arrowstyle": "-", "lw": 0.6, "shrinkA": 2, "shrinkB": 0})
    ax.set_xlim(0, th.H)
    ax.set_ylim(0, 0.27)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", left=False, right=False, labelleft=False)
    ax.set_xlabel("gap $g = v - c$ (€)", labelpad=3)
    return save(fig, path)


def fig_bids(s: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W - 0.1, 3.15))
    x = np.linspace(0, th.H, 101)
    series = [("B", FLASH, False), ("A", LITE, False), ("C", LITE, True)]
    for ax, role in zip(axes, ("buyer", "seller"), strict=True):
        eq = th.eq_buyer_bid(x) if role == "buyer" else th.eq_seller_ask(x)
        ax.plot(x, x, color=GREY, lw=0.8, ls=(0, (1, 2)))
        ax.plot(x, eq, color=INK, lw=1.1)
        for name, model, hollow in series:
            st = s["conditions"][name][role]["strata"]
            color = COLOR[model]
            ax.plot([b["value_mean"] for b in st], [b["amount_mean"] for b in st], ls="",
                    marker="o", ms=4.2, mew=0.9, mfc="white" if hollow else color, mec=color)
        if role == "buyer":
            square(ax, "buyer's value (€)", "bid (€)")
        else:
            square(ax, "seller's cost (€)", "ask (€)")
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=5, mfc=COLOR[FLASH], mec=COLOR[FLASH],
                   label="Flash"),
        plt.Line2D([], [], ls="", marker="o", ms=5, mfc=COLOR[LITE], mec=COLOR[LITE],
                   label="Flash-Lite"),
        plt.Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=COLOR[LITE], mew=0.9,
                   label="Flash-Lite, shopping assistant"),
        plt.Line2D([], [], color=INK, lw=1.1, label="equilibrium"),
        plt.Line2D([], [], color=GREY, lw=0.8, ls=(0, (1, 2)), label="bid = value, ask = cost"),
    ]
    fig.subplots_adjust(wspace=0.4, bottom=0.31)
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.5, columnspacing=0.9, handletextpad=0.4, fontsize=10)
    return save(fig, path)


def fig_new_rule(s: dict, path: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(SIDE, SIDE))
    x = np.linspace(0, th.H, 101)
    ax.plot(x, th.eq_buyer_bid(x, k=1 / 3), color=INK, lw=1.1, label="new equilibrium")
    ax.plot(x, th.eq_buyer_bid(x), color=GREY, lw=0.9, ls=(0, (4, 2)), label="old equilibrium")
    st = s["conditions"]["E"]["buyer"]["strata"]
    ax.plot([b["value_mean"] for b in st], [b["amount_mean"] for b in st], ls="", marker="o",
            ms=3.8, mew=0.8, mfc=COLOR[FLASH], mec=COLOR[FLASH], label="Flash, mean bid")
    square(ax, "buyer's value (€)", "bid (€)")
    return save(fig, path)


# --- the pages ------------------------------------------------------------------------------


def page_title(s: dict, f: dict, q: dict, out: Path) -> Page:
    pg = Page()
    head, tail = TITLE.split(" the ")
    pg.raw("#align(center)[\n"
           f"  #v(2pt)\n  #text(20pt, weight: \"bold\", hyphenate: false)"
           f"[{T(head + ' the')} \\ {T(tail)}]\n  #v(10pt)\n"
           f"  #text(14pt)[{T(AUTHOR)}]\n  #v(-1pt)\n  #text(12pt)[{T(DATE)}]\n]",
           [AUTHOR, TITLE, DATE])
    abstract = (
        "I put Gemini agents on both sides of a sealed-bid double auction, with values "
        "uniform on 0–100 €. When both sides bid strategically, no trading rule can realise "
        f"more than {f['frontier']} of the gains (Myerson and Satterthwaite, 1983). Gemini "
        f"3.8 Flash realised {f['B.eff']}, as much as equilibrium play, and every bid and ask "
        "it made that could trade sat on the textbook equilibrium to the cent. Gemini 3.5 "
        f"Flash-Lite realised {f['A.eff']}, and {f['C.eff']} as a shopping assistant.")
    pg.raw("#v(6pt)\n#align(center, text(12.5pt, weight: \"bold\")[Abstract])\n#v(-3pt)\n"
           f"#pad(x: 1.2em)[#set text(12pt)\n#set par(leading: 0.45em)\n{T(abstract)}]",
           ["Abstract", P(abstract)])
    labels = fig_overview(s, f, out / "fig-1.svg")
    pg.raw("#v(4pt)")
    pg.figure("fig-1.svg", 1, "Share of the available gains each pair of agents realised. "
              "The dashed line is the Myerson–Satterthwaite bound.", labels)
    note = (f"{f['calls']} bids and asks, {usd(f['cost_usd'])} of API calls. Code, data and "
            "the full derivation are linked in the first comment.")
    pg.raw("#v(1fr)\n#block(width: 100%, {\n"
           "  line(length: 30%, stroke: 0.4pt)\n  v(-4pt)\n"
           f"  text(10.5pt)[{T(note)}]\n}})", P(note))
    return pg


def page_theorem(s: dict, f: dict, q: dict, out: Path) -> Page:
    pg = Page()
    pg.h("The theorem")
    pg.p("A buyer values an item at $v$ and the seller's cost is $c$. Each knows only their "
         "own number. Myerson and Satterthwaite (1983) showed that no trading rule is at the "
         "same time")
    items = [("efficient", "every deal with $v > c$ happens;"),
             ("incentive compatible", "it works when both sides bid strategically;"),
             ("voluntary", "nobody expects to lose by taking part;"),
             ("budget-balanced", "no money comes from outside.")]
    pg.raw('#enum(numbering: "(i)", ' + ", ".join(
        f"[#emph[{T(term)}]: {T(rest)}]" for term, rest in items) + ")",
        [f"{term}: {P(rest)}" for term, rest in items])
    pg.p("With $v$ and $c$ uniform on $[0, 100]$, the best a rule with the last three "
         "properties can do is")
    pg.eq(f'"gains realised" / "gains available" <= (9\\/64)/(1\\/6) = 27/32 approx '
          f"{f['frontier']}",
          f"gains realised / gains available ≤ (9/64)/(1/6) = 27/32 ≈ {f['frontier']}")
    labels = fig_regions(out / "fig-2.svg")
    pg.beside("fig-2.svg", 2, "Deals by cost and value. The best such rule trades only when "
              "$v − c ≥ 25$, above the solid line.",
              f"The missing {f['lost_at_frontier']} comes from private information. Each side "
              "shades its number to get a better price, and the deals between the two lines, "
              "worth making, fall through.", labels)
    return pg


def page_auction(s: dict, f: dict, q: dict, out: Path) -> Page:
    pg = Page()
    pg.h("The auction")
    pg.p("Buyer and seller each submit a sealed number, a bid $b$ and an ask $s$. If "
         "$b ≥ s$ they trade at $p = (b + s)\\/2$. Against asks spread evenly above a lowest "
         "ask, the buyer's best bid is")
    pg.eq("max_b integral_(s_min)^b (v - (b + s)/2) dif s ==> b = (2v + s_min)/3",
          "max over b of ∫ from s_min to b of (v − (b + s)/2) ds ⇒ b = (2v + s_min)/3")
    pg.p("Doing the same for the seller and solving both gives the linear equilibrium of "
         "Chatterjee and Samuelson (1983):")
    pg.eq("b(v) = 2/3 v + 25/3, quad s(c) = 2/3 c + 25",
          "b(v) = 2/3 v + 25/3, s(c) = 2/3 c + 25")
    labels = fig_strategies(out / "fig-3.svg")
    pg.beside("fig-3.svg", 3, "The equilibrium bid and ask. Dotted, bidding one's own number.",
              "Wherever a trade is possible, the buyer bids below their value and the seller "
              "asks above their cost. They trade only when $v − c ≥ 25$, so this simple "
              f"auction loses just the {f['lost_at_frontier']} that any such rule must lose.",
              labels)
    return pg


def page_why(s: dict, f: dict, q: dict, out: Path) -> Page:
    pg = Page()
    pg.h("Why 27/32")
    pg.p("Let $g = v − c$. For $g ≥ 0$ its density is $(100 − g)\\/100^2$, so if only the "
         "deals with $g ≥ t$ go through, the expected gains are")
    pg.eq("G(t) = integral_t^100 g (100 - g)/100^2 dif g = ((100 - t)^2 (100 + 2t))"
          "/(6 dot 100^2)",
          "G(t) = ∫ from t to 100 of g (100 − g)/100² dg = (100 − t)² (100 + 2t)/(6 · 100²)")
    labels = fig_gains(f, out / "fig-4.svg")
    pg.figure("fig-4.svg", 4, "Each deal weighted by its gap $g$. Trading only when "
              f"$g ≥ 25$ keeps {f['frontier']} of the gains.", labels)
    pg.p(f"With no threshold $G(0) =$ {f['first_best_eur']}. The equilibrium threshold is "
         f"$t = 25$, with $G(25) =$ {f['eq_gains_eur']}, so")
    pg.eq("(G(25))/(G(0)) = (3/4)^2 dot 3/2 = 27/32", "G(25)/G(0) = (3/4)² · 3/2 = 27/32")
    return pg


def page_bids(s: dict, f: dict, q: dict, out: Path) -> Page:
    pg = Page()
    pg.h("Bids against the equilibrium")
    pg.p("The solid line is the equilibrium of Section 2, and on the dotted line an agent "
         "would bid or ask its own number.")
    labels = fig_bids(s, out / "fig-5.svg")
    pg.figure("fig-5.svg", 5, "Mean bid or ask per 5 € band of values or costs.", labels)
    pg.bullets([
        f"Flash put {f['flash_k_half.at_eq_trade_region']} tradeable bids and asks exactly "
        "on the equilibrium.",
        f"Flash-Lite buyers bid ${f['A.buyer.slope']}v + {f['A.buyer.intercept']}$ instead of "
        f"$0.67v + 8.3$, and its sellers ask ${f['A.seller.slope']}c + "
        f"{f['A.seller.intercept']}$ instead of $0.67c + 25$: both sides shade too hard.",
        "As a shopping assistant its asks follow "
        f"${f['C.seller.slope']}c + {f['C.seller.intercept']}$, and "
        f"{f['C.seller.ir_binding']} were below cost yet low enough to be accepted.",
    ])
    return pg


def page_efficiency(s: dict, f: dict, q: dict, out: Path) -> Page:
    pg = Page()
    pg.h("Efficiency")
    pg.p("Every buyer call meets every seller call, and the intervals come from a bootstrap "
         "that resamples buyers and sellers separately.")
    caption = ("Share of the available gains realised, with 95% intervals, and what "
               "equilibrium play realises on the same draws.")
    header = ["", "Realised", "95% CI", "Equilibrium,\nsame draws"]
    rows, plain = [], [f"Table 1. {caption}", " | ".join(h.replace("\n", " ") for h in header)]
    for model, members in GROUPS:
        rows.append(f"table.cell(colspan: 4)[#emph[{T(model)}]]")
        plain.append(model)
        for name, label in members:
            cells = [label, f[f"{name}.eff"], f[f"{name}.eff_ci"], f[f"{name}.eq_same_draws"]]
            rows.append(f"[#h(0.9em){T(cells[0])}], " + ", ".join(f"[{T(c)}]" for c in cells[1:]))
            plain.append(" | ".join(P(c) for c in cells))
    pg.raw(f'#cap("Table", 1, above: 1.1em)[{T(caption)}]\n'
           "#ruled((auto, auto, auto, auto), (left, right, center, right),\n"
           f"  ({', '.join(head(h) for h in header)}),\n  " + ",\n  ".join(rows) + ")",
           plain)
    pg.p("Flash realises what equilibrium play realises on the same draws, in both prompts "
         f"and under both price rules. Flash-Lite trades in {f['A.trade_rate']} of the pairs, "
         f"against {f['A.eq_trade_rate']} in equilibrium, and misses {f['A.missed']} of the "
         f"gains; as a shopping assistant it trades in {f['C.trade_rate']}. No rule can beat "
         f"{f['frontier']} here, and with $p = (b + 2s)\\/3$ the equilibrium realises "
         f"{f['k13_eff']}.")
    note = ("On a 20 × 20 grid of band midpoints, "
            f"{f['grid_pairs_on_boundary']} pairs sit exactly on the trade boundary "
            "$v − c = 25$, and whether they trade moves the benchmark itself between "
            f"{f['grid_eq_eff_skip']} and {f['grid_eq_eff_trade']}. So values and costs are "
            "drawn uniformly within each 5 € band.")
    pg.raw(f"#emph[Why random draws.] {T(note)}", f"Why random draws. {P(note)}")
    return pg


def page_recall(s: dict, f: dict, q: dict, out: Path) -> Page:
    pg = Page()
    pg.h("Recall or reasoning?")
    src = "Gemini 3.8 Flash, as a shopping assistant"
    pg.raw(f"#mquote[{T(q['recall'])}][{T(src)}]", f"“{q['recall']}” ({src})")
    pg.p(f"Flash named Chatterjee–Samuelson in {f['flash_k_half.named']} calls, even with the "
         "game disguised as a used camera lens on a second-hand marketplace. So I changed the "
         "price to $p = (b + 2s)\\/3$, which moves the equilibrium to")
    pg.eq("b(v) = 3/4 v + 25/3, quad s(c) = 3/5 c + 100/3",
          "b(v) = 3/4 v + 25/3, s(c) = 3/5 c + 100/3")
    src = "Gemini 3.8 Flash, with the new price rule"
    pg.raw(f"#mquote[{T(q['derive'])}][{T(src)}]", f"“{q['derive']}” ({src})")
    labels = fig_new_rule(s, out / "fig-6.svg")
    pg.beside("fig-6.svg", 6, "Flash's mean bid per 5 € band with the new rule (dots), the "
              "new equilibrium (solid) and the old one (dashed).",
              f"{f['E.at_eq_trade_region']} tradeable bids and asks landed on the new "
              f"equilibrium and {f['E.at_k_half']} on the old one. Derivatives or first-order "
              f"conditions appear in {f['E.derivation_terms']} summaries, against "
              f"{f['flash_k_half.derivation_terms']} before, and it realised {f['E.eff']} of "
              f"the gains against {f['k13_eff']} in theory.", labels)
    return pg


def page_exploit(s: dict, f: dict, q: dict, out: Path) -> Page:
    pg = Page()
    pg.h("Exploitability")
    pg.p("For each agent I fitted the exact best response to its asks, and to its bids, on "
         "half of the calls and scored it on the other half. Against a finite sample the "
         "payoff is piecewise linear, so its maximum falls on a sampled value and enumerating "
         "them is exact.")
    caption = ("What the best-responding rival gains per auction over playing the "
               "equilibrium, in €, with 95% intervals.")
    header = ["", "Rival buyer,\nagainst its asks", "Rival seller,\nagainst its bids"]
    head_typ = ", ".join(head(h) for h in header)
    rows, plain = [], [f"Table 2. {caption}",
                       " | ".join(h.replace("\n", " ") for h in header)]
    for model, members in GROUPS:
        rows.append(f"table.cell(colspan: 3)[#emph[{T(model)}]]")
        plain.append(model)
        for name, label in members:
            cells = [label]
            for side in ("seller", "buyer"):
                cells.append(f"{f[f'{name}.exploit_{side}.gain']} "
                             f"{f[f'{name}.exploit_{side}.gain_ci']}")
            rows.append(f"[#h(0.9em){T(cells[0])}], " + ", ".join(f"[{T(c)}]" for c in cells[1:]))
            plain.append(" | ".join(P(c) for c in cells))
    pg.raw(f'#cap("Table", 2, above: 1.1em)[{T(caption)}]\n'
           "#ruled((auto, auto, auto), (left, center, center),\n"
           f"  ({head_typ}),\n  " + ",\n  ".join(rows) + ")", plain)
    worst_buyer = max((s["conditions"][n]["exploitability"]["vs_model_buyer"]
                       ["br_gain_over_eq_play"], n) for n in ORDER)[1]
    pg.p("A best response gains nothing against Flash. Against the shopping-assistant "
         f"Flash-Lite, the rival buyer earns {f['C.exploit_seller.rival_br_eur']} per auction "
         f"instead of {f['C.exploit_seller.rival_eq_eur']}: {f['C.exploit_seller.gain']} € "
         f"{f['C.exploit_seller.gain_ci']}, {f['C.exploit_seller.gain_rel'].lstrip('+')} more "
         "than equilibrium play. As buyers, every agent was harder to exploit: at most "
         f"{f[worst_buyer + '.exploit_buyer.gain']} €.")
    pg.p("The intervals hold the fitted strategy fixed and resample the scored calls. "
         "Refitting inside the bootstrap, as I first did, biased them down.")
    return pg


def page_takeaways(s: dict, f: dict, q: dict, out: Path) -> Page:
    pg = Page()
    pg.h("What I'd check")
    items = [
        f"Benchmark trading agents against the frontier, {f['frontier']} here, rather than "
        "100%. A pair above it is not in equilibrium, and a smarter rival can exploit that.",
        "Use variants of famous games to tell recall from reasoning.",
        f"Test the framing you will ship. Flash-Lite fell from {f['A.eff']} to {f['C.eff']} as "
        "a shopping assistant.",
        "Measure exploitability with held-out best responses, and block dominated moves such "
        "as selling below cost.",
    ]
    pg.raw("\n".join(f"+ {T(t)}" for t in items), [P(t) for t in items])
    limits = ("A famous game, thought summaries rather than reasoning, one prior and one "
              "shot, each model playing itself, and model and thinking level changed "
              "together. The shopping-assistant prompt and the new price rule were added "
              "after a 20-call pilot.")
    pg.raw(f"#emph[Limitations.] {T(limits)}", f"Limitations. {P(limits)}")
    pg.raw("#v(4pt)\n" + T("Code, data and the full derivation of 27/32:") +
           f"\n\n#align(center, strong[{T(REPO)}])\n\n#text(11pt)[" +
           T("Everything reruns from the cached calls, without an API key.") + "]",
           ["Code, data and the full derivation of 27/32:", REPO,
            "Everything reruns from the cached calls, without an API key."])
    refs = []
    for lead, journal, tail in REFERENCES:
        refs.append(f"[{T(lead)} #emph[{T(journal)}], {T(tail)}]")
    pg.raw("#heading(level: 2, numbering: none)[References]\n"
           "#block({\n  set text(10.5pt)\n  set par(justify: false, hanging-indent: 1.2em, "
           "leading: 0.42em, spacing: 0.6em)\n  " + "\n  parbreak()\n  ".join(refs) +
           "\n})")
    pg.refs = [f"{lead} {journal}, {tail}" for lead, journal, tail in REFERENCES]
    return pg


PAGES = [page_title, page_theorem, page_auction, page_why, page_bids, page_efficiency,
         page_recall, page_exploit, page_takeaways]


def make_carousel(results_dir: Path, carousel_dir: Path) -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # font timestamps, harmless
    paper_style()
    summary = style.load_summary(results_dir)
    facts = fx.build(summary)
    quotes = load_quotes(results_dir)
    out = Path(carousel_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("slide-*.png"), *out.glob("fig-*.svg")]:
        old.unlink()
    pages = [make(summary, facts, quotes, out) for make in PAGES]
    source = PREAMBLE.replace("@TITLE", TITLE).replace("@AUTHOR", AUTHOR)
    source += "\n" + "\n\n#pagebreak()\n\n".join(pg.markup() for pg in pages) + "\n"
    typ = out / "carousel.typ"
    typ.write_text(source)
    common = {"root": str(out), "font_paths": [str(style.FONTS)], "ignore_system_fonts": True}
    pdf_path = out / "carousel.pdf"
    typst.compile(str(typ), output=str(pdf_path), **common)
    typst.compile(str(typ), output=str(out / "slide-{p}.png"), format="png", ppi=180, **common)
    pngs = sorted(out.glob("slide-*.png"), key=lambda p: int(re.search(r"\d+", p.name)[0]))
    for i, png in enumerate(pngs, start=1):
        png.rename(out / f"slide-{i:02d}.png")
    if len(pngs) != len(PAGES):
        print(f"  warning: {len(pngs)} pages for {len(PAGES)} slides: something overflows")
    text = [f"## Slide {i}\n\n" + "\n\n".join(t for t in pg.txt if t.strip())
            for i, pg in enumerate(pages, start=1)]
    refs = getattr(pages[-1], "refs", [])
    (out / "slides.txt").write_text("\n\n".join(text) + "\n\n## References\n\n" +
                                    "\n".join(refs) + "\n")
    print(f"-> {pdf_path} ({len(pngs)} pages)")
