r"""Figure 6: the two gradient envelopes of Section 4.3 against the margin, and where they meet.

    python figures/paper_plots/fig6_envelopes.py [--out PATH]      # -> fig6_envelopes.png

Nothing is read: the figure is drawn from the formulas.  The envelopes are the largest sizes the two
gradients on a view can take: the braked cross-entropy envelope 2(1 - p_g)/tau of Equation (6) and
the invariance envelope 2 lambda / V of Equation (9), at the values of Section 4.2 (lambda = 2,
tau = 0.02, V = 4).

The x-axis is the margin s_h = <z_v, a_g> - <z_v, a_h> = tau ln(p_g / p_h) of Equation (8), with
the residual mass on one competitor h, the worst case of online Appendix D, so that
1 - p_g = 1 / (1 + exp(s / tau)): the CE envelope is the softmax's own curve, flat while the view
sits on the wrong side, bending at s = 0 (p_g = p_h, the view classified) and falling exponentially
after.  The envelopes meet at s* = tau ln(V / (lambda tau) - 1) = 0.02 ln 99 = 0.092, the stopping
point p_g = 0.99.  The fill under the lower envelope is gray (either gradient can be there);
between the two it takes the colour of the larger, the term that can steer.  The y-axis is
logarithmic; the lettering is 7 pt at one column.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path("paper_plots_out") / "fig6_envelopes.png"
LAM, TAU, V = 2.0, 0.02, 4
CE, INV = "#f2541b", "#2f6fdb"
CE_INK, INV_INK, INK, MUTED = "#c03c0c", "#1f4fa8", "#1f2328", "#5b6573"
FS = 7.0
SIZE = (3.30, 1.85)
S_LO, S_HI = -0.12, 0.22


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    plt.rcParams.update({"font.size": FS, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
                         "mathtext.fontset": "dejavusans"})
    s = np.linspace(S_LO, S_HI, 600)
    resid = 1 / (1 + np.exp(s / TAU))              # 1 - p_g, the worst case of one competitor
    ce = 2 * resid / TAU
    yc = 2 * LAM / V
    inv = np.full_like(s, yc)
    s_star = TAU * np.log(V / (LAM * TAU) - 1)     # 0.092
    ylo, yhi = 1e-3, 1e3

    fig, ax = plt.subplots(figsize=SIZE, dpi=300)
    ax.set_yscale("log")
    ax.set_xlim(S_LO, S_HI); ax.set_ylim(ylo, yhi)
    lo, hi = np.minimum(ce, inv), np.maximum(ce, inv)
    ax.fill_between(s, ylo, lo, color="#9aa3b0", alpha=0.18, lw=0, zorder=1)
    ax.fill_between(s, lo, hi, where=ce >= inv, color=CE, alpha=0.20, lw=0, zorder=1, interpolate=True)
    ax.fill_between(s, lo, hi, where=ce <= inv, color=INV, alpha=0.20, lw=0, zorder=1, interpolate=True)
    ax.plot(s, ce, color=CE, lw=1.4, zorder=3, solid_capstyle="round")
    ax.plot(s, inv, color=INV, lw=1.4, zorder=3, solid_capstyle="round")

    # s = 0: the view classified (p_g = p_h), the dot of Figure 5a
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=2)
    ax.text(0.004, 4e2, "classified", color=MUTED, ha="left", va="center", fontsize=FS)
    # where the envelopes meet: the stopping point
    ax.axvline(s_star, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=2)
    ax.plot([s_star], [yc], "o", ms=4.2, mfc="white", mec=INK, mew=0.9, zorder=4)
    ax.annotate(r"stopping point, $p_g = 0.99$", (s_star, yc), xytext=(-5, -7), textcoords="offset points",
                ha="right", va="top", color=INK, fontsize=FS)

    # direct labels on the curves, and which term can steer in each wedge
    ax.text(S_LO + 0.005, 2 / TAU * 1.5, r"CE: $2(1-p_g)/\tau$", color=CE_INK, ha="left", va="bottom", fontsize=FS)
    ax.text(S_HI - 0.003, yc * 1.3, r"invariance: $2\lambda/V$", color=INV_INK, ha="right", va="bottom", fontsize=FS)
    ax.text(-0.065, 9, "CE steers", color=CE_INK, ha="center", va="center", fontsize=FS)
    ax.text(0.192, 0.09, "invariance\nsteers", color=INV_INK, ha="center", va="center", fontsize=FS,
            linespacing=1.0)                       # two lines, inside the blue wedge where it is widest

    # the stopping point as a balance: from the left CE pushes the margin up, from the right the
    # invariance pull erodes it and CE grows again; both arrows end at s*
    ya = 3.2e-3
    for x0, x1, txt, col, ink, ha in [(-0.10, s_star - 0.006, "CE pushes", CE, CE_INK, "center"),
                                      (0.192, s_star + 0.006, "invariance pulls", INV, INV_INK, "center")]:
        ax.annotate("", xy=(x1, ya), xytext=(x0, ya),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, mutation_scale=7, shrinkA=0, shrinkB=0),
                    zorder=4)
        ax.text((x0 + x1) / 2, ya / 1.25, txt, color=ink, ha=ha, va="top", fontsize=FS)

    ax.set_xlabel(r"margin $s_h = \langle z_v, a_g\rangle - \langle z_v, a_h\rangle$", fontsize=FS, color=INK)
    ax.set_ylabel("largest gradient size", fontsize=FS, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS, colors=INK, length=2.5)
    ax.tick_params(which="minor", length=1.5)
    fig.tight_layout(pad=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}: {SIZE[0]} x {SIZE[1]} in at 300 dpi; s* = {s_star:.4f}, "
          f"CE at s = 0: {2 * 0.5 / TAU:.0f}, at s*: {2 / TAU / (1 + np.exp(s_star / TAU)):.3f}")


if __name__ == "__main__":
    main()
