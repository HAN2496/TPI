import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from .model import sigmoid


def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_population_summary(pop, phi, out_dir):
    """Feature-level summary using effective coefficients gamma * theta."""
    d = pop.d
    mu = pop.mu_samples                                          # [M, d]
    Sjj = pop.Sigma_samples[:, np.arange(d), np.arange(d)]       # [M, d]
    rows = []
    for j, name in enumerate(pop.feature_names):
        m, s = float(mu[:, j].mean()), float(mu[:, j].std())
        lo, hi = np.percentile(mu[:, j], [2.5, 97.5])
        rows.append({"feature": name, "group": phi.groups[j],
                     "inclusion_probability": float(pop.gamma_pip[j]),
                     "mu_mean": m, "mu_sd": s, "mu_ci_lo": float(lo), "mu_ci_hi": float(hi),
                     "mu_z": abs(m) / s if s > 0 else 0.0,
                     "sigma_jj": float(Sjj[:, j].mean()),
                     "between_user_sd": float(np.sqrt(Sjj[:, j]).mean()),
                     "s_coef": float((mu[:, j] ** 2 + Sjj[:, j]).mean())})

    # 그룹(=센서)별 S_coef (eq 17-18): 합 = Σ_j S_coef, 정규화 = 합 / d_g
    groups = {}
    for r in rows:
        if r["group"] == "bias":
            continue
        g = groups.setdefault(r["group"], {"sum": 0.0, "d": 0})
        g["sum"] += r["s_coef"]; g["d"] += 1
    group_rows = [{"group": g, "s_coef_group": v["sum"], "s_coef_group_norm": v["sum"] / v["d"], "d_g": v["d"]}
                  for g, v in groups.items()]

    with open(out_dir / "population.json", "w", encoding="utf-8") as f:
        json.dump({"features": rows, "groups": group_rows}, f, ensure_ascii=False, indent=2)

    lines = ["Population posterior per feature  (effective coefficient = gamma * slab coefficient)",
             "=" * 122, "",
             f"{'feature':<32} {'PIP':>7} {'mu':>8} {'sd':>7} {'95% CI':>19} {'|mu|/sd':>8} {'btw-user sd':>12} {'S_coef':>9}"]
    for r in sorted(rows, key=lambda r: -r["s_coef"]):
        lines.append(f"{r['feature']:<32} {r['inclusion_probability']:>7.3f} "
                     f"{r['mu_mean']:>+8.3f} {r['mu_sd']:>7.3f} "
                     f"[{r['mu_ci_lo']:>+7.3f}, {r['mu_ci_hi']:>+7.3f}] {r['mu_z']:>8.2f} "
                     f"{r['between_user_sd']:>12.3f} {r['s_coef']:>9.3f}")
    lines += ["", "Sensor group score (eq 17-18)  S_coef_group = sum_j S_coef,  norm = / d_g", "-" * 56,
              f"{'group':<24} {'S_coef_group':>14} {'norm':>10} {'d_g':>5}"]
    for r in sorted(group_rows, key=lambda r: -r["s_coef_group_norm"]):
        lines.append(f"{r['group']:<24} {r['s_coef_group']:>14.3f} {r['s_coef_group_norm']:>10.3f} {r['d_g']:>5}")
    (out_dir / "population.txt").write_text("\n".join(lines), encoding="utf-8")


def write_spike_slab_summary(pop, out_dir):
    unit_rows = []
    for index, (name, cols, pip) in enumerate(zip(
        pop.gamma_unit_names, pop.gamma_unit_columns, pop.gamma_unit_pip
    )):
        switches = int(np.count_nonzero(np.diff(pop.gamma_unit_samples[:, index])))
        unit_rows.append({
            "unit": name,
            "features": [pop.feature_names[j] for j in cols],
            "inclusion_probability": float(pip),
            "posterior_switches": switches,
            "median_probability_selected": bool(pip >= 0.5),
        })

    feature_rows = [
        {
            "feature": name,
            "group": pop.feature_groups[j],
            "inclusion_probability": float(pop.gamma_pip[j]),
        }
        for j, name in enumerate(pop.feature_names)
    ]
    report = {
        "unit": pop.spike_slab_unit,
        "prior": {"a": pop.spike_slab_a, "b": pop.spike_slab_b},
        "posterior_pi_mean": pop.pi_bar,
        "units": sorted(
            unit_rows, key=lambda row: -row["inclusion_probability"]
        ),
        "features": feature_rows,
    }
    with open(out_dir / "spike_slab.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        f"Spike-and-slab posterior inclusion ({pop.spike_slab_unit}-level)",
        "=" * 78,
        f"pi ~ Beta({pop.spike_slab_a:g}, {pop.spike_slab_b:g}), "
        f"posterior E[pi|D]={pop.pi_bar:.4f}",
        "PIP = posterior mean of gamma; PIP >= 0.5 marks the median-probability model",
        "",
        f"{'unit':<40} {'PIP':>9} {'switches':>10} {'n_feat':>8}  selected",
    ]
    for row in report["units"]:
        mark = "*" if row["median_probability_selected"] else ""
        lines.append(
            f"{row['unit']:<40} {row['inclusion_probability']:>9.4f} "
            f"{row['posterior_switches']:>10} {len(row['features']):>8}  {mark}"
        )
    (out_dir / "spike_slab.txt").write_text("\n".join(lines), encoding="utf-8")


def save_trust_metrics(drivers, out_path):
    lines = ["=" * 64, "Trust-Annotated Evaluation Metrics", "=" * 64, "",
             "Method:  posterior x bootstrap joint 95% CI of AUROC",
             "Gate:    trustworthy iff (CI_lo > 0.5) AND (CI_width < 0.15)", ""]
    for d in drivers:
        t = d["trust_final"]
        lines.append(f"[{d['name']}]  n_holdout={t['n']}  pos={t['n_pos']}  neg={t['n_neg']}")
        for label, key in [("prior", "trust_prior"), ("final", "trust_final"), ("peak ", "trust_peak")]:
            t = d[key]
            verdict = "trust " if t["trustworthy"] else "REJECT"
            if np.isnan(t["mean"]):
                lines.append(f"  {label}  AUROC=  N/A    ({t['reason']})")
            else:
                extra = f"  ({t['reason']})" if t["reason"] else ""
                lines.append(f"  {label}  AUROC={t['mean']:.4f}  CI=[{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]"
                             f"  width={t['width']:.3f}  [{verdict}]{extra}")
        lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def write_train_user_profiles(model, pipe, out_dir, Phis, ys, names):
    lines = ["Train user posterior profiles  (Gibbs)", "=" * 60, ""]
    profiles = []
    for name, Phi, y in zip(names, Phis, ys):
        if name not in model.user_names:
            continue
        i = model.user_names.index(name)
        theta, std = model.theta_means[i], model.theta_stds[i]
        probs = sigmoid(Phi @ theta)
        auroc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
        rows = pipe.group_contributions(theta, Phi, cov=np.diag(std ** 2), top_k=10)
        profiles.append({"name": name, "train_auroc": float(auroc),
                         "top_components": [{"group": r[0], "signed": r[1], "abs": r[2], "uncertainty": r[3]} for r in rows]})
        lines.append(f"[{name}] train AUROC={auroc:.4f}")
        for group, signed, abs_val, unc in rows:
            direction = "positive" if signed > 0 else "negative"
            unc_str = f"  unc={unc:.4f}" if unc is not None else ""
            lines.append(f"- {group:<35} {signed:+.4f}  abs={abs_val:.4f}{unc_str}  ({direction})")
        lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_user_profiles.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(out_dir / "train_user_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
