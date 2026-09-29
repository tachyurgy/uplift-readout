"""Render results/*.json into the static report in site/.

Every number shown on the page is read from results/*.json here; the template
has no hand-typed figures. Charts are drawn with matplotlib as SVG and inlined,
with the light-mode hexes swapped for CSS variables so they follow the page
theme.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SITE = ROOT / "site"

# light-mode hexes used while drawing; each maps to a CSS variable at inline time
C = {"ink": "#1b1b1a", "ink2": "#5b5a55", "grid": "#e4e3de", "s1": "#2a78d6", "s2": "#eb6834",
     "s3": "#1baf7a", "s4": "#eda100", "neutral": "#9a9890"}
VARS = {v: f"var(--c-{k})" for k, v in C.items()}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "text.color": C["ink"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "lines.linewidth": 2, "svg.fonttype": "none", "legend.frameon": False, "legend.fontsize": 10,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
})
LEARNER_COLORS = {"T-learner": C["s1"], "X-learner": C["s2"], "S-learner": C["s3"],
                  "Transformed outcome": C["s4"], "Random": C["neutral"]}


def svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    s = re.sub(r'width="[\d.]+pt" height="[\d.]+pt"', 'width="100%" height="auto"', s, count=1)
    for hexv, var in VARS.items():
        s = s.replace(hexv, var)
    return s


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def pp(x, d=2):
    return f"{100 * x:.{d}f}pp"


# ------------------------------------------------------------------ uplift summaries
def summarise_uplift(u):
    grid = np.array(u["grid"])
    out = {}
    for y in ("visit", "conversion"):
        learners = list(u["folds"][0]["outcomes"][y].keys())
        per = {}
        for name in learners:
            curves = np.array([f["outcomes"][y][name]["curve"] for f in u["folds"]])
            qini = np.array([f["outcomes"][y][name]["qini"] for f in u["folds"]])
            auuc = np.array([f["outcomes"][y][name]["auuc"] for f in u["folds"]])
            per[name] = {"curves": curves, "qini_mean": qini.mean(), "qini_sd": qini.std(ddof=1),
                         "auuc_mean": auuc.mean(), "auuc_sd": auuc.std(ddof=1), "qini": qini}
        rand_auuc = np.array([f["outcomes"][y]["Random"]["curve"][-1] / 2 for f in u["folds"]])
        best = max((n for n in learners if n != "Random"), key=lambda n: per[n]["qini_mean"])
        out[y] = {"grid": grid, "learners": per, "best": best, "diag_auuc_mean": rand_auuc.mean()}
    return out


def captured_share(curves, grid, phi):
    """Share of the all-treated incremental outcomes captured by treating top phi."""
    i = int(np.argmin(np.abs(grid - phi)))
    return curves[:, i] / curves[:, -1]


def break_even_policy(curves, grid, ratios):
    """For cost/value ratio r (cost per treated user / value per outcome), the
    fraction to treat that maximises net = curve(phi) - r * phi, per fold."""
    rows = []
    for r in ratios:
        net = curves - r * grid[None, :]
        best = net.argmax(axis=1)
        rows.append({"ratio": r, "phi": grid[best], "net": net[np.arange(len(best)), best],
                     "net_all": net[:, -1]})
    return rows


# ------------------------------------------------------------------ charts
def chart_hero(S, cost_ratio):
    s = S["conversion"]
    g = s["grid"]
    best = s["best"]
    cur = s["learners"][best]["curves"]
    share = cur / cur[:, [-1]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(100 * g, 100 * share.min(0), 100 * share.max(0), color=C["s1"], alpha=0.18, lw=0)
    ax.plot(100 * g, 100 * share.mean(0), color=C["s1"], label=f"Target by model ({best})")
    ax.plot([0, 100], [0, 100], color=C["neutral"], lw=1.5, ls="--", label="Target at random")
    for phi in (0.2, 0.4):
        i = int(np.argmin(np.abs(g - phi)))
        v = 100 * share.mean(0)[i]
        ax.plot([100 * phi], [v], "o", ms=7, color=C["s1"], mec="white", mew=2)
        ax.annotate(f"top {int(phi*100)}% of users\n→ {v:.0f}% of the extra conversions", (100 * phi, v),
                    xytext=(12, -34), textcoords="offset points", fontsize=10, color=C["ink"])
    ax.set_xlabel("Share of users shown the campaign (ranked by predicted uplift)")
    ax.set_ylabel("Share of all incremental conversions")
    ax.set_xlim(0, 100); ax.set_ylim(0, 105)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%"); ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.legend(loc="lower right")
    return svg(fig)


def chart_estimators(r):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    names = list(r["variance_reduction"]["visit"].keys())
    for ax, y in zip(axes, ("visit", "conversion")):
        vr = r["variance_reduction"][y]
        for i, n in enumerate(names):
            e = vr[n]
            col = C["neutral"] if n == "difference in means" else (C["s2"] if n.startswith("AIPW") else C["s1"])
            ax.plot([100 * e["ci_low"], 100 * e["ci_high"]], [i, i], color=col, lw=3, solid_capstyle="round")
            ax.plot([100 * e["estimate"]], [i], "o", color=col, ms=6, mec="white", mew=1.5)
        ax.set_title("Visit rate" if y == "visit" else "Conversion rate", loc="left", color=C["ink"])
        ax.set_xlabel("Effect of treatment (percentage points)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([n.replace(", ", ",\n") for n in names])
    axes[0].invert_yaxis()
    fig.tight_layout()
    return svg(fig)


def chart_balance(r):
    b = r["balance"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(b))
    null = 1.96 * r["balance_smd_null_sd"]
    ax.axhspan(-null, null, color=C["neutral"], alpha=0.25, lw=0)
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.plot(x - 0.12, [v["smd"] for v in b], "o", color=C["s2"], ms=7, label="Raw treatment vs control", mec="white", mew=1.5)
    ax.plot(x + 0.12, [v["smd_weighted"] for v in b], "o", color=C["s1"], ms=7, label="After propensity weighting", mec="white", mew=1.5)
    ax.set_xticks(x); ax.set_xticklabels([v["feature"] for v in b])
    ax.set_ylabel("Standardized mean difference")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    return svg(fig)


def chart_hte(r):
    tests = sorted([h for h in r["hte_tests"] if h["outcome"] == "conversion"], key=lambda h: -h["Q"])[:3]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    overall = r["aipw"]["conversion"]["ate"]["estimate"]
    for ax, t in zip(axes, tests):
        bins = [h for h in r["hte_bins"] if h["feature"] == t["feature"] and h["outcome"] == "conversion"]
        xs = np.arange(len(bins))
        for i, h in enumerate(bins):
            col = C["s1"] if h["p_holm"] < 0.05 else C["neutral"]
            ax.plot([i, i], [100 * h["ci_low"], 100 * h["ci_high"]], color=col, lw=3, solid_capstyle="round")
            ax.plot([i], [100 * h["estimate"]], "o", color=col, ms=6, mec="white", mew=1.5)
        ax.axhline(100 * overall, color=C["s2"], lw=1.2, ls="--")
        ax.set_title(f"by {t['feature']} bin", loc="left", color=C["ink"])
        ax.set_xticks(xs); ax.set_xticklabels([str(i + 1) for i in xs])
        ax.set_xlabel("low → high")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Conversion effect (pp)")
    fig.tight_layout()
    return svg(fig)


def chart_power(p):
    rows = p["power"]["conversion"]["table"]
    x = [100 * r["rel_mde"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(x, [r["n_5050"] for r in rows], "-o", color=C["s1"], ms=5, label="50/50 split")
    ax.plot(x, [r["n_8515"] for r in rows], "-o", color=C["s2"], ms=5, label="85/15 split (as run)")
    ax.axhline(p["n_total"], color=C["neutral"], lw=1.2, ls="--")
    ax.text(x[-1], p["n_total"] * 1.15, f"this dataset: {p['n_total']/1e6:.1f}M users", ha="right", fontsize=10, color=C["ink2"])
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}%" for v in x])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(lambda v, _: f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k")
    ax.set_xlabel("Smallest relative lift in conversion rate you want to detect")
    ax.set_ylabel("Users needed (80% power)")
    ax.legend(loc="lower left")
    return svg(fig)


def chart_peeking(p):
    rows = p["peeking_fpr_by_looks"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = [r["looks"] for r in rows]
    ax.plot(x, [100 * r["fpr"] for r in rows], "-o", color=C["s2"], ms=6, label="Stop at first p < 0.05")
    ax.axhline(5, color=C["neutral"], lw=1.2, ls="--")
    ax.text(x[-1], 6.2, "promised 5%", ha="right", fontsize=10, color=C["ink2"])
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x]); ax.minorticks_off()
    ax.set_xlabel("Number of times the dashboard is checked (A/A test, no real effect)")
    ax.set_ylabel("False positive rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    return svg(fig)


def chart_qini(S, y):
    s = S[y]
    g = 100 * s["grid"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    scale = 1000 if y == "conversion" else 100
    for name in ("T-learner", "X-learner", "S-learner", "Transformed outcome", "Random"):
        cur = s["learners"][name]["curves"] * scale
        ax.fill_between(g, cur.min(0), cur.max(0), color=LEARNER_COLORS[name], alpha=0.15, lw=0)
        ax.plot(g, cur.mean(0), color=LEARNER_COLORS[name], lw=2 if name != "Random" else 1.5,
                ls="--" if name == "Random" else "-", label=name)
    ax.set_xlabel("Share of users treated, ranked by predicted uplift")
    ax.set_ylabel("Incremental conversions per 1,000 users" if y == "conversion" else "Incremental visits per 100 users")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlim(0, 100); ax.set_ylim(0, None)
    ax.legend(loc="lower right")
    return svg(fig)


def chart_calibration(u, S):
    y = "conversion"
    best = S[y]["best"]
    cal = np.array([[(d["pred"], d["obs"], d["se"]) for d in f["outcomes"][y][best]["calibration"]] for f in u["folds"]])
    pred, obs = cal[:, :, 0].mean(0), cal[:, :, 1].mean(0)
    se = np.sqrt((cal[:, :, 2] ** 2).sum(0)) / cal.shape[0]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    xs = np.arange(1, 11)
    ax.bar(xs - 0.18, 100 * pred, width=0.34, color=C["s1"], label="Predicted", edgecolor="none")
    ax.bar(xs + 0.18, 100 * obs, width=0.34, color=C["s3"], label="Measured on held-out users", edgecolor="none")
    ax.errorbar(xs + 0.18, 100 * obs, yerr=196 * se, fmt="none", ecolor=C["ink2"], elinewidth=1.2, capsize=0)
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_xticks(xs); ax.set_xlabel("Predicted-uplift decile (1 = lowest)")
    ax.set_ylabel("Conversion effect (pp)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    return svg(fig), {"pred": pred, "obs": obs, "se": se}


def chart_policy(rows):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = [r["ratio"] * 1e4 for r in rows]
    phi = np.array([r["phi"] for r in rows]) * 100
    ax.fill_between(x, phi.min(1), phi.max(1), color=C["s1"], alpha=0.18, lw=0, step="mid")
    ax.step(x, np.median(phi, 1), where="mid", color=C["s1"])
    ax.set_xlabel("Cost per targeted user ÷ value of one conversion (× 10,000)")
    ax.set_ylabel("Best share of users to target")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylim(0, 105)
    return svg(fig)


# ------------------------------------------------------------------ build
def main():
    r = json.loads((RES / "readout.json").read_text())
    p = json.loads((RES / "power.json").read_text())
    u = json.loads((RES / "uplift.json").read_text())
    S = summarise_uplift(u)
    g = S["conversion"]["grid"]
    best = S["conversion"]["best"]
    cur_best = S["conversion"]["learners"][best]["curves"]
    ate_conv = r["aipw"]["conversion"]["ate"]["estimate"]

    # policy: ratios expressed against the average effect so the break-even is interpretable
    ratios = np.round(np.linspace(0, 3 * ate_conv, 61), 8)
    pol = break_even_policy(cur_best, g, ratios)
    # the stated scenario: cost per targeted user equals 1.5x the average incremental
    # conversion value per user, i.e. blanket targeting loses money
    scen = {}
    for mult in (0.5, 1.0, 1.5):
        rr = break_even_policy(cur_best, g, [mult * ate_conv])[0]
        scen[str(mult)] = {"phi_median": float(np.median(rr["phi"])), "phi_min": float(rr["phi"].min()),
                           "phi_max": float(rr["phi"].max()),
                           "net_best": float(rr["net"].mean()), "net_best_sd": float(rr["net"].std(ddof=1)),
                           "net_all": float(rr["net_all"].mean())}
    top = {int(100 * phi): {"mean": float(captured_share(cur_best, g, phi).mean()),
                            "min": float(captured_share(cur_best, g, phi).min()),
                            "max": float(captured_share(cur_best, g, phi).max())} for phi in (0.1, 0.2, 0.3, 0.4, 0.5)}
    cal_svg, cal = chart_calibration(u, S)
    # the lowest predicted decile's measured effect
    low_decile = {"obs": float(cal["obs"][0]), "ci": [float(cal["obs"][0] - 1.96 * cal["se"][0]),
                                                        float(cal["obs"][0] + 1.96 * cal["se"][0])]}
    charts = {
        "hero": chart_hero(S, None),
        "estimators": chart_estimators(r),
        "balance": chart_balance(r),
        "hte": chart_hte(r),
        "power": chart_power(p),
        "peeking": chart_peeking(p),
        "qini_conv": chart_qini(S, "conversion"),
        "qini_visit": chart_qini(S, "visit"),
        "calibration": cal_svg,
        "policy": chart_policy(pol),
    }
    derived = {"best": best, "top": top, "scenarios": scen, "low_decile": low_decile,
               "ate_conv_per_1000": 1000 * ate_conv,
               "per_learner": {y: {n: {k: float(v[k]) for k in ("qini_mean", "qini_sd", "auuc_mean", "auuc_sd")}
                                   for n, v in S[y]["learners"].items()} for y in S},
               "best_visit": S["visit"]["best"],
               "cal": {k: [float(x) for x in v] for k, v in cal.items()}}
    (RES / "derived.json").write_text(json.dumps(derived, indent=1))

    env = Environment(loader=FileSystemLoader(str(ROOT / "templates")), autoescape=False)
    env.filters.update(pct=pct, pp=pp, fmtint=lambda x: f"{int(x):,}", f=lambda x, d=3: f"{x:.{d}f}",
                       sci=lambda x: f"{x:.1e}", m=lambda x: f"{x/1e6:.2f}M")
    html = env.get_template("report.html.j2").render(r=r, p=p, u=u, d=derived, charts=charts)
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(html)
    print("wrote site/index.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
