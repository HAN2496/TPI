"""Fig. 1 of the CoPL paper: pipeline overview with the two adaptation paths.

Rectangles, arrows and text only; outputs fig_overview.{svg,pdf,png} next to this script.
    python docs/copl/my_papers/figures/make_overview.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "mathtext.fontset": "dejavusans"})
W, H = 7.16, 3.1
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
C_EDGE, ACC, WARM, BAD, GREY = "#1d1f24", "#1f5f8b", "#8a4b08", "#8b2b2b", "#5d6270"


def box(x, y, w, h, text, fc="white", ec=C_EDGE, lw=0.8, fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)
    return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2, r=x + w, t=y + h)


def arrow(p, q, lw=0.9, color=C_EDGE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, shrinkA=1, shrinkB=1))


def polyline(points, lw=0.9, color=C_EDGE):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="round")
    arrow(points[-2], points[-1], lw=lw, color=color)


def label(x, y, s, fs=5.6, color=GREY, ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color)


def panel(x, y, w, h, title, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.08", fc=fc, ec="none"))
    ax.text(x + 0.08, y + h - 0.12, title, ha="left", va="center", fontsize=7.4, weight="bold", color=ACC)


# ----------------------------------------------------------------------------- (A) population training
panel(0.08, 0.1, 3.2, H - 0.2, "(A)  Population training (offline)", "#e4eef6")
for ux in (0.45, 0.95, 1.45):                                   # three stars: one user, its own items
    ax.plot(ux, 2.25, "o", color=ACC, ms=6)
    for j in range(3):
        ix = ux - 0.16 + 0.16 * j
        ax.plot(ix, 1.85, "s", color=GREY, ms=3.2)
        ax.plot([ux, ix], [2.25, 1.85], color=GREY, lw=0.6)
for a, b in [(0.45, 0.79), (1.11, 1.29), (0.61, 1.45)]:        # item-item bridges across stars
    ax.plot([a, b], [1.76, 1.76], color=WARM, lw=0.8, ls="--")
label(0.95, 2.48, "users $u$ (one star each)", fs=5.6, color=ACC)
label(1.0, 1.56, "items $i$; the only cross-user edges\nare the item–item kNN graph $A_{ii}$ (dashed)", fs=5.4, color=WARM)
enc = box(1.95, 1.75, 1.2, 0.6, "encoder $\\phi$\nGabor scattering (fixed)\n$z_i=\\phi(x_i)$, kNN graph", fs=5.8)
arrow((1.6, 2.0), (enc["x"], 2.0))
gcf = box(0.25, 0.75, 1.3, 0.62, "graph propagation\n(LightGCN on $A_+,A_-,A_{ii}$)\nuser $e_u$, item $e_i$", fs=5.8, ec=ACC, lw=1.0)
rm = box(1.75, 0.75, 1.4, 0.62, "reward model $r(x;e)$ (CNN)\nBCE on $(x_i, e_u, y_i)$\n+ Dirichlet mixtures of $e_u$\ncalibrated, ensemble of 3", fs=5.6, ec=ACC, lw=1.0)
arrow((0.3, 1.78), (0.3, gcf["t"]))
arrow((enc["cx"], enc["y"]), (rm["cx"], rm["t"]))
arrow((gcf["r"], gcf["cy"]), (rm["x"], rm["cy"]))
label((gcf["r"] + rm["x"]) / 2, gcf["cy"] + 0.09, "$e_u$", fs=5.8, color=C_EDGE)
label(1.65, 0.4, "Sec. III", fs=5.2)

# ----------------------------------------------------------------------------- (B) adaptation
panel(3.42, 0.1, 3.66, H - 0.2, "(B)  New occupant: which mixture of population users?", "#e3f1e6")
ctx = box(3.58, 2.1, 1.1, 0.55, "context labels\n$(x_{\\star i}, y_{\\star i})_{i\\leq t}$", fs=5.9)
vote = box(4.9, 2.15, 2.0, 0.6,
           "CoPL vote (Sec. IV)\nattach $x_{\\star i}$ to kNN items, $v_i$\n$c_u=\\Sigma_{i\\in I_u}s_i v_i$   grows with $n_u$", fs=5.6, ec=BAD, fc="#fbeaea")
ll = box(4.9, 1.3, 2.0, 0.6,
         "likelihood posterior (Sec. V)\n$\\ell_u=\\Sigma_i \\log p(y_{\\star i}\\mid x_{\\star i}, e_u)$\n$w_u\\propto\\exp(\\ell_u/\\tau)$", fs=5.6, ec=ACC, fc="#eaf2f8", lw=1.0)
mix = box(3.58, 0.55, 1.4, 0.55, "mixed embedding\n$\\hat e_\\star=\\Sigma_u w_u e_u$", fs=5.9)
pred = box(5.3, 0.55, 1.6, 0.55, "prediction for pending $j$\n$\\hat p_j=\\sigma(r(x_{\\star j};\\hat e_\\star))$", fs=5.9)
arrow((ctx["r"], ctx["cy"]), (vote["x"], vote["cy"]), color=BAD)
arrow((ctx["r"], ctx["cy"] - 0.1), (ll["x"], ll["cy"]), color=ACC)
label(5.9, 2.03, "degree bias: $w_u \\to$ largest user", fs=5.4, color=BAD)
label(5.9, 1.19, "no graph; $\\tau=1$ exact, $\\tau>1$ tempered", fs=5.4, color=ACC)
# vote -> mixed embedding: down the gap between the context column and the rule boxes
polyline([(vote["x"] + 0.06, vote["y"]), (vote["x"] + 0.06, 2.02), (4.79, 2.02), (4.79, mix["t"] + 0.1),
          (mix["r"] - 0.3, mix["t"])], lw=0.7, color=BAD)
arrow((ll["x"] + 0.35, ll["y"]), (mix["r"] - 0.1, mix["t"]), color=ACC)
arrow((mix["r"], mix["cy"]), (pred["x"], pred["cy"]))
# population embeddings and reward model enter the likelihood rule: along the panel margin
polyline([(rm["r"] + 0.02, rm["cy"]), (3.5, rm["cy"]), (3.5, ll["cy"] + 0.12), (ll["x"], ll["cy"] + 0.12)], lw=0.7, color=GREY)
label(4.15, ll["cy"] + 0.21, "$\\{e_u\\}_u$ and $r$", fs=5.4, color=GREY)
label(5.3, 0.4, "Sec. V, VI", fs=5.2)

for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"fig_overview.{ext}", dpi=300 if ext == "png" else None)
print("written", OUT / "fig_overview.{svg,pdf,png}")
