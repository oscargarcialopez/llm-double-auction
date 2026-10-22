"""The README figures, drawn from results/summary.json and results/calls.csv only."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import style
from . import theory as th
from .style import INK, INK_2, MUTED, SURFACE, WASH, plt

ROLES = ("buyer", "seller")


def eq_line(role: str, x, k: float):
    return th.eq_buyer_bid(x, k=k) if role == "buyer" else th.eq_seller_ask(x, k=k)


def no_trade_zone(role: str, k: float) -> tuple[float, float]:
    """Values (costs) whose equilibrium bid (ask) can never meet an equilibrium ask (bid)."""
    if role == "buyer":
        return 0.0, (th.eq_seller_ask(0.0, k=k) - th.eq_buyer_bid(0.0, k=k)) * (1 + k)
    return (th.eq_buyer_bid(th.H, k=k) - th.eq_seller_ask(0.0, k=k)) * (2 - k), th.H


def load_calls(results_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], list] = {}
    with (Path(results_dir) / "calls.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["valid"] == "True":
                out.setdefault((row["condition"], row["role"]), []).append(
                    (float(row["value"]), float(row["amount"])))
    return {key: np.array(val) for key, val in out.items()}


def signed(x: float, digits: int = 2) -> str:
    """+0.78 / −0.03, with a real minus sign."""
    return f"{x:+.{digits}f}".replace("-", "\u2212")


def line_text(slope: float, intercept: float, sym: str) -> str:
    sign = "+" if intercept >= 0 else "\u2212"
    return f"{slope:.2f}{sym} {sign} {abs(intercept):.1f}"


def condition_title(name: str, c: dict) -> str:
    return f"{name}  {style.MODEL_NAME[c['model']]} · {style.FRAMING[c['prompt_version']]}"


def bid_functions(summary: dict, calls: dict, out: Path) -> None:
    names = list(summary["conditions"])
    fig, axes = plt.subplots(len(names), 2, figsize=(7.6, 3.0 * len(names) + 1.2),
                             sharex=True, sharey=True)
    x = np.linspace(0, th.H, 201)
    for i, name in enumerate(names):
        c = summary["conditions"][name]
        k = c["k"]
        color = style.MODEL_COLOR[c["model"]]
        for j, role in enumerate(ROLES):
            ax = axes[i, j]
            lo, hi = no_trade_zone(role, k)
            ax.axvspan(lo, hi, color=WASH, lw=0, zorder=0)
            ax.plot(x, x, color=MUTED, lw=1.0, ls=(0, (1, 2)))
            if k != 0.5:
                ax.plot(x, eq_line(role, x, 0.5), color=INK_2, lw=1.1, ls=(0, (4, 3)))
            ax.plot(x, eq_line(role, x, k), color=INK, lw=1.3)
            pts = calls[(name, role)]
            ax.scatter(pts[:, 0], pts[:, 1], s=9, color=color, alpha=0.35, lw=0, zorder=3)
            st = c[role]["strata"]
            xm = np.array([s["value_mean"] for s in st])
            ym = np.array([s["amount_mean"] for s in st])
            band = np.array([s["amount_ci"] for s in st])
            ax.errorbar(xm, ym, yerr=[ym - band[:, 0], band[:, 1] - ym], fmt="o", ms=4.5,
                        color=color, mec=SURFACE, mew=1.0, elinewidth=1.2, capsize=0, zorder=4)
            d = c[role]
            fit = d["fit_trade_region"]
            sym = "v" if role == "buyer" else "c"
            n_rel = d["n_trade_region"]
            at_eq = round(d["share_at_eq_trade_region"] * n_rel)
            # buyers never bid far above their value and sellers never ask far below their
            # cost, so the top-left (bids) and bottom-right (asks) corners are free for text
            corner = (0.03, 0.97, "left", "top") if role == "buyer" else (0.97, 0.03, "right",
                                                                            "bottom")
            ax.text(corner[0], corner[1],
                    f"fit where trade is possible: {line_text(fit['slope'], fit['intercept'], sym)}"
                    f"\nequilibrium: {line_text(d['eq_slope'], d['eq_intercept'], sym)}"
                    f"\nat equilibrium to the cent: {at_eq} of {n_rel}",
                    transform=ax.transAxes, ha=corner[2], va=corner[3], fontsize=7.5,
                    color=INK_2, linespacing=1.4, zorder=5,
                    bbox={"boxstyle": "square,pad=0.25", "fc": SURFACE, "ec": "none",
                          "alpha": 0.85})
            ax.set_xlim(0, th.H)
            ax.set_ylim(0, th.H)
            ax.set_aspect("equal")
            if i == len(names) - 1:
                ax.set_xlabel("buyer's value v (€)" if role == "buyer" else "seller's cost c (€)")
            if j == 0:
                ax.set_ylabel("bid b (€)")
            else:
                ax.set_ylabel("ask s (€)")
        axes[i, 0].set_title(condition_title(name, c), loc="left", fontsize=10, pad=6)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=4, color=style.MODEL_COLOR[m], alpha=0.5,
                   label=f"one call, {style.MODEL_NAME[m]}") for m in style.MODEL_COLOR
    ] + [
        plt.Line2D([], [], ls="", marker="o", ms=5, color=INK_2, mec=SURFACE,
                   label="mean per €5 band, 95% CI"),
        plt.Line2D([], [], color=INK, lw=1.3, label="Bayes-Nash equilibrium of the auction"),
        plt.Line2D([], [], color=INK_2, lw=1.1, ls=(0, (4, 3)),
                   label="k = ½ formula (row E only)"),
        plt.Line2D([], [], color=MUTED, lw=1.0, ls=(0, (1, 2)), label="truthful (b = v, s = c)"),
        plt.Rectangle((0, 0), 1, 1, color=WASH, label="cannot trade against equilibrium play"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / fig.get_figheight()), w_pad=0.5)
    save(fig, out / "bid_functions")


def efficiency_figure(summary: dict, out: Path) -> None:
    names = list(summary["conditions"])
    fig, ax = plt.subplots(figsize=(9.2, 0.75 * len(names) + 1.6))
    ys = np.arange(len(names))[::-1]
    for y, name in zip(ys, names, strict=True):
        c = summary["conditions"][name]
        e = c["efficiency"]
        ax.barh(y, e["efficiency"], height=0.46, color=style.MODEL_COLOR[c["model"]], lw=0)
        lo, hi = e["efficiency_ci"]
        ax.plot([lo, hi], [y, y], color=INK, lw=1.4, solid_capstyle="butt")
        eq = e["eq_efficiency_same_draws"]
        ax.plot([eq, eq], [y - 0.3, y + 0.3], color=INK, lw=1.8)
        ax.text(1.03, y, f"{style.pct(e['efficiency'])}  {style.pct_ci(e['efficiency_ci'])}",
                va="center", ha="left", fontsize=9, color=INK, transform=ax.get_yaxis_transform())
    frontier = summary["theory"]["ms_frontier"]
    ax.axvline(frontier, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(frontier - 0.01, len(names) - 0.45, "Myerson–Satterthwaite bound: 27/32 = "
            f"{style.pct(frontier)}", ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys, [condition_title(n, summary["conditions"][n]).replace(" · ", "\n", 1)
                       for n in names], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(names) - 0.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("share of the first-best gains from trade realised (every buyer call "
                  "against every seller call)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D([], [], color=INK, lw=1.4, label="95% CI (stratified bootstrap)"),
        plt.Line2D([], [], color=INK, lw=1.8, marker="|", ms=10, ls="",
                   label="equilibrium play on the same draws"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out / "efficiency")


def exploitability_figure(summary: dict, out: Path) -> None:
    names = list(summary["conditions"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.62 * len(names) + 2.0), sharey=True)
    ys = np.arange(len(names))[::-1]
    titles = {"vs_model_buyer": "the model bids as the buyer;\na seller best-responds",
              "vs_model_seller": "the model asks as the seller;\na buyer best-responds"}
    lim = 0.0
    for ax, key in zip(axes, titles, strict=True):
        for y, name in zip(ys, names, strict=True):
            c = summary["conditions"][name]
            x = c["exploitability"][key]
            gain, (lo, hi) = x["br_gain_over_eq_play"], x["br_gain_over_eq_play_ci"]
            ax.barh(y, gain, height=0.46, color=style.MODEL_COLOR[c["model"]], lw=0)
            ax.plot([lo, hi], [y, y], color=INK, lw=1.4, solid_capstyle="butt")
            ax.plot(x["br_gain_in_sample"], y, marker="o", ms=5, mfc=SURFACE, mec=INK, mew=1.1,
                    ls="")
            ax.text(1.02, y, f"{signed(gain)} €\n[{signed(lo)}, {signed(hi)}]",
                    transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.5,
                    linespacing=1.3)
            lim = max(lim, hi, x["br_gain_in_sample"], -lo)
        ax.axvline(0, color=INK_2, lw=0.9)
        ax.set_title(titles[key], fontsize=9.5, loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    for ax in axes:
        ax.set_xlim(-0.35, lim * 1.1)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else signed(v, 1)))
    fig.supxlabel("rival's gain over playing the equilibrium itself (€ per auction)",
                  fontsize=10, color=INK_2, y=0.1)
    axes[0].set_yticks(ys, [f"{n}  {style.MODEL_NAME[summary['conditions'][n]['model']]}\n"
                            f"{style.FRAMING[summary['conditions'][n]['prompt_version']]}"
                            for n in names], fontsize=8.5)
    handles = [
        plt.Line2D([], [], color=INK, lw=1.4,
                   label="95% CI, best response fitted on the other half of the calls"),
        plt.Line2D([], [], marker="o", ms=5, mfc=SURFACE, mec=INK, ls="",
                   label="in-sample best response (optimistic)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 0.94, 1), w_pad=6)
    save(fig, out / "exploitability")


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    # no CreationDate, so rebuilding from the cache leaves the committed PDFs unchanged
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None})
    plt.close(fig)
    print(f"-> {stem.with_suffix('.png')}")


def make_figures(results_dir: Path, figures_dir: Path) -> None:
    style.apply()
    summary = style.load_summary(results_dir)
    calls = load_calls(results_dir)
    bid_functions(summary, calls, figures_dir)
    efficiency_figure(summary, figures_dir)
    exploitability_figure(summary, figures_dir)
