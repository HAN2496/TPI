"""Overview figure of the hierarchical Bayesian personalization framework (T-IV paper, Fig. 1).

Three panels that mirror the three contributions:
  (A) population posterior from many evaluators' good/bad feedback  (offline)
  (B) cold start + sequential personalization of a new evaluator     (online)
  (C) quantified trust: epistemic / aleatoric decomposition, reliability
Outputs fig_framework.{svg,pdf,png} next to this script.  Only rectangles, arrows and text,
so it can be re-drawn or edited in a vector tool.

    python docs/fully_bayesian/my_papers/figures/make_framework.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "mathtext.fontset": "dejavusans"})

W, H = 7.16, 3.4                                    # IEEE two-column width
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

C_POP, C_NEW, C_TRUST = "#e4eef6", "#e3f1e6", "#fbeedc"
C_EDGE, ACC, WARM, GREY = "#1d1f24", "#1f5f8b", "#8a4b08", "#5d6270"


def box(x, y, w, h, text, fc="white", ec=C_EDGE, lw=0.8, fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)
    return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2, r=x + w, t=y + h)


def arrow(p, q, lw=0.9, color=C_EDGE, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=lw, color=color, shrinkA=1, shrinkB=1))


def polyline(points, lw=1.1, color=ACC):
    """Elbow connector: plain segments, arrow head on the last one."""
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="round")
    arrow(points[-2], points[-1], lw=lw, color=color)


def label(x, y, s, fs=5.6, color=GREY, ha="center", va="center", rot=0):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, rotation=rot)


def panel(x, w, title, fc):
    ax.add_patch(FancyBboxPatch((x, 0.1), w, H - 0.3, boxstyle="round,pad=0.0,rounding_size=0.08", fc=fc, ec="none"))
    ax.text(x + 0.08, H - 0.3, title, ha="left", va="center", fontsize=7.4, weight="bold", color=ACC)


# ----------------------------------------------------------------------------- panels
x0, w0 = 0.08, 2.46
x1, w1 = 2.66, 2.22
x2, w2 = 5.0, 2.08
panel(x0, w0, "(A)  Population posterior  (offline)", C_POP)
panel(x1, w1, "(B)  New evaluator  (online)", C_NEW)
panel(x2, w2, "(C)  Quantified trust", C_TRUST)

# ---- (A) rows: data (top), model (middle), Gibbs + roles (lower), free band at the bottom for the connector
# evaluator cards
cx, cy, cw, ch = x0 + 0.1, 2.3, 1.16, 0.5
for k in range(3):
    off = 0.05 * (2 - k)
    ax.add_patch(FancyBboxPatch((cx + off, cy - off), cw, ch, boxstyle="round,pad=0.02,rounding_size=0.06", fc="white", ec=C_EDGE, lw=0.8))
ax.text(cx + cw / 2, cy + ch / 2 + 0.03, "evaluators $u=1,\\dots,U$\nwindows $\\tau_{ui}$, labels $y_{ui}$\ngood / bad, once per episode", ha="center", va="center", fontsize=5.8, linespacing=1.3)
fm = box(x0 + 1.4, cy - 0.1, 1.04, 0.6, "feature map $\\phi$\nISO-inspired bank,\npruned, standardized\n$z_{ui}=\\phi(\\tau_{ui})\\in\\mathbb{R}^{d}$", fs=5.8)
arrow((cx + cw + 0.02, cy + ch / 2), (fm["x"], cy + ch / 2))

hm = box(x0 + 0.1, 1.42, 2.34, 0.66,
         "hierarchical logistic model\n"
         "$y_{ui}\\sim\\mathrm{Bern}(\\sigma(\\theta_u^{\\top}z_{ui}))$,  $\\theta_u=\\gamma\\odot\\tilde{\\theta}_u$\n"
         "$\\tilde{\\theta}_u\\sim\\mathcal{N}(\\mu,\\Sigma)$,  $(\\mu,\\Sigma)\\sim\\mathrm{NIW}$,  $\\gamma$: spike-and-slab", fs=5.9)
arrow((fm["cx"], fm["y"]), (fm["cx"], hm["t"]))
label(fm["cx"] + 0.2, (fm["y"] + hm["t"]) / 2, "$z_{ui},\\,y_{ui}$", fs=5.8, color=C_EDGE)

gb = box(x0 + 0.1, 0.72, 1.3, 0.56,
         "Pólya–Gamma Gibbs\n$M$ posterior particles\n$\\{\\mu^{(m)},\\Sigma^{(m)},\\gamma^{(m)}\\}_{m=1}^{M}$", fs=5.9, ec=ACC, lw=1.0)
arrow((gb["cx"], hm["y"]), (gb["cx"], gb["t"]))
fr = box(x0 + 1.5, 0.72, 0.94, 0.56, "required features\n& sensors\ncommon / specific / inactive", fs=5.6, fc="#fff8ec", ec=WARM)
arrow((gb["r"], gb["cy"]), (fr["x"], fr["cy"]))
label((gb["r"] + fr["x"]) / 2, gb["t"] + 0.07, "PIP, $q_j$", fs=5.2, color=C_EDGE)
label(gb["cx"], gb["y"] - 0.08, "Sec. IV-A, B", fs=5.2)
label(fr["cx"], fr["y"] - 0.08, "Sec. V  (contribution 3)", fs=5.2)

# ---- (B)
cs = box(x1 + 0.16, 2.26, w1 - 0.26, 0.6,
         "cold start  ($t=0$)\n$\\theta_{\\star}^{(m)}\\sim\\mathcal{N}(\\mu^{(m)},\\Sigma^{(m)})$\npopulation posterior predictive", fs=5.9, ec=ACC, lw=1.0)
# particles connector: Gibbs bottom -> free band -> panel-B margin -> cold start
band_y = 0.4
polyline([(gb["cx"], gb["y"]), (gb["cx"], band_y), (x1 + 0.06, band_y), (x1 + 0.06, cs["cy"]), (cs["x"], cs["cy"])])
label((gb["cx"] + x1) / 2, band_y + 0.07, "posterior particles $\\{\\mu^{(m)},\\Sigma^{(m)}\\}_m$ = cold-start prior", fs=5.4, color=ACC)

fb = box(x1 + 0.16, 1.5, 0.8, 0.54, "feedback stream\n$y_{\\star 1},y_{\\star 2},\\dots,y_{\\star t}$\n(context, in order)", fs=5.6)
up = box(x1 + 1.08, 1.46, 1.04, 0.62, "sequential update\nPG sweeps of $\\theta_{\\star}^{(m)}$\nper particle, $(\\mu,\\Sigma)^{(m)}$\nfixed, warm-started", fs=5.6)
arrow((up["cx"], cs["y"]), (up["cx"], up["t"]))
arrow((fb["r"], fb["cy"]), (up["x"], up["cy"]))

pr = box(x1 + 0.16, 0.62, w1 - 0.26, 0.62,
         "prediction for a pending episode $j$\n"
         "$p^{(m)}_{j}=\\sigma(z_j^{\\top}\\theta_{\\star}^{(m)})$,   $\\hat p_j=\\frac{1}{M}\\sum_m p^{(m)}_j$\n"
         "the posterior is kept as particles, not a point", fs=5.8)
arrow((up["cx"], up["y"]), (up["cx"], pr["t"]))
arrow((fb["cx"], fb["y"]), (fb["cx"], pr["t"]), style="<|-", lw=0.7, color=GREY)
label(fb["cx"] + 0.04, (fb["y"] + pr["t"]) / 2, "next label\n$t\\to t+1$", fs=5.2, ha="left")
label(x1 + w1 / 2, pr["y"] - 0.08, "Sec. IV-C, D  (contribution 1)", fs=5.2)

# ---- (C)
dc = box(x2 + 0.1, 2.26, w2 - 0.2, 0.62,
         "per-episode decomposition\n$\\hat p(1-\\hat p)=\\mathbb{E}_m[p^{(m)}(1-p^{(m)})]+\\mathrm{Var}_m(p^{(m)})$\n"
         "aleatoric (ambiguity)  +  epistemic (not yet learned)", fs=5.6)
# particle predictions connector: prediction box right -> panel gap -> decomposition box left
gx = (x1 + w1 + x2) / 2
polyline([(pr["r"], pr["cy"]), (gx, pr["cy"]), (gx, dc["cy"]), (dc["x"], dc["cy"])])
label(gx + 0.02, (pr["cy"] + dc["cy"]) / 2, "$\\{p^{(m)}_j\\}_{m}$", fs=5.6, color=ACC, rot=90, ha="center")

# decay sketch
sx, sy, sw, sh = x2 + 0.16, 1.36, 0.92, 0.68
ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.02,rounding_size=0.04", fc="white", ec=C_EDGE, lw=0.6))
tt = np.linspace(0, 1, 60)
ax.plot(sx + 0.1 + 0.74 * tt, sy + 0.14 + 0.2 * np.exp(-3.2 * tt), color=ACC, lw=1.2)
ax.plot(sx + 0.1 + 0.74 * tt, sy + 0.5 + 0 * tt, color="#8b2b2b", lw=1.2, ls="--")
label(sx + 0.6, sy + 0.27, "epistemic", fs=5.3, color=ACC)
label(sx + 0.47, sy + 0.58, "aleatoric", fs=5.3, color="#8b2b2b")
label(sx + sw / 2, sy + 0.05, "context size $t$", fs=5.0, color=C_EDGE)
rl = box(x2 + 1.16, 1.36, 0.82, 0.68, "evaluator-level\nreliability interval\n$[\\mathrm{AUROC}_{lo},\\mathrm{AUROC}_{hi}]$\nposterior × bootstrap", fs=5.4)
arrow((dc["x"] + 0.35, dc["y"]), (sx + sw / 2, sy + sh))
arrow((dc["r"] - 0.35, dc["y"]), (rl["cx"], rl["t"]))

out = box(x2 + 0.1, 0.56, w2 - 0.2, 0.56,
          "to the controller: $\\hat p_j$ and how far to trust it\n"
          "high epistemic → ask for feedback;  high aleatoric →\nborderline, do not switch hard;  unreliable → do not act", fs=5.4, fc="#fff8ec", ec=WARM)
arrow((sx + sw / 2, sy), (sx + sw / 2, out["t"]))
arrow((rl["cx"], rl["y"]), (rl["cx"], out["t"]))
label(x2 + w2 / 2, out["y"] - 0.08, "Sec. IV-E  (contribution 2)", fs=5.2)

for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"fig_framework.{ext}", dpi=300 if ext == "png" else None)
print("written", OUT / "fig_framework.{svg,pdf,png}")
