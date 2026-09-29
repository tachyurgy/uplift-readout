"""Part 1: experiment readout on the full 13.98M rows.

Everything that touches all rows runs as DuckDB aggregates over the parquet
file; only per-arm sufficient statistics come back into Python. The one
exception is the cross-fitted ML covariate (CUPAC), which streams the table in
chunks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
from scipy import stats as sps

from uplift_readout import stats

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "criteo.parquet"
OUT = ROOT / "results"
FEATS = [f"f{i}" for i in range(12)]
OUTCOMES = ["visit", "conversion"]
DESIGN_SHARE = 0.85


def con():
    c = duckdb.connect()
    c.execute("SET memory_limit='3GB'; SET threads=4")
    c.execute(f"CREATE VIEW d AS SELECT * FROM read_parquet('{PARQUET}')")
    return c


def arm_moments(c, y: str, covs: list[str], table: str = "d") -> dict:
    """Per-arm n, mean/var of y, mean of covariates, Cov(X,X), Cov(X,y) via SQL."""
    cols = covs + [y]
    exprs = ["count(*) AS n"] + [f"avg({a}) AS m_{a}" for a in cols]
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i:]]
    exprs += [f"covar_samp({a},{b}) AS c_{a}__{b}" for a, b in pairs]
    rows = c.execute(f"SELECT treatment, {', '.join(exprs)} FROM {table} GROUP BY treatment").fetchall()
    names = ["treatment", "n"] + [f"m_{a}" for a in cols] + [f"c_{a}__{b}" for a, b in pairs]
    out = {}
    for r in rows:
        rec = dict(zip(names, r))
        p = len(covs)
        cxx = np.zeros((p, p))
        for i, a in enumerate(covs):
            for j, b in enumerate(covs):
                key = f"c_{a}__{b}" if f"c_{a}__{b}" in rec else f"c_{b}__{a}"
                cxx[i, j] = rec[key]
        out["t" if rec["treatment"] == 1 else "c"] = {
            "n": rec["n"],
            "mean_y": rec[f"m_{y}"],
            "var_y": rec[f"c_{y}__{y}"],
            "mean_x": np.array([rec[f"m_{a}"] for a in covs]),
            "cov_xx": cxx,
            "cov_xy": np.array([rec[f"c_{a}__{y}"] for a in covs]),
        }
    return out


def cupac_scores(c) -> None:
    """Cross-fitted LightGBM prediction of each outcome from pre-treatment features.

    Rows carry a deterministic fold (hash of source row id). A model trained on
    a 1.5M-row sample of fold A (treatment ignored) scores fold B and vice
    versa, so no row's covariate ever saw its own outcome. Scores are written to
    data/cupac.parquet keyed by rid.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    dst = ROOT / "data" / "cupac.parquet"
    if dst.exists():
        return
    writer = None
    for fold in (0, 1):
        tr = c.execute(f"SELECT {','.join(FEATS)}, visit, conversion FROM d WHERE fold={fold} "
                       f"USING SAMPLE 1500000 ROWS (reservoir, 42)").fetchnumpy()
        Xtr = np.column_stack([tr[f] for f in FEATS]).astype(np.float32)
        models = {}
        for y in OUTCOMES:
            m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=63,
                                   min_child_samples=500, n_jobs=4, verbose=-1, random_state=fold)
            models[y] = m.fit(Xtr, tr[y])
        del tr, Xtr
        c.execute(f"SELECT rid, {','.join(FEATS)} FROM d WHERE fold={1 - fold}")
        while True:
            batch = c.fetch_df_chunk(20)
            if len(batch) == 0:
                break
            X = batch[FEATS].to_numpy(np.float32)
            tbl = pa.table({"rid": batch["rid"].to_numpy(),
                            "s_visit": models["visit"].predict_proba(X)[:, 1],
                            "s_conversion": models["conversion"].predict_proba(X)[:, 1]})
            writer = writer or pq.ParquetWriter(dst, tbl.schema)
            writer.write_table(tbl)
    writer.close()


def main():
    t0 = time.time()
    c = con()
    res: dict = {"source_rows": c.execute("SELECT count(*) FROM d").fetchone()[0]}

    # ---------------------------------------------------------------- SRM
    nt, nc = c.execute("SELECT sum(treatment), count(*) - sum(treatment) FROM d").fetchone()
    res["srm"] = stats.srm_test(int(nt), int(nc), DESIGN_SHARE).to_dict()
    # sensitivity: what the test would say against a naive 50/50 expectation
    res["srm_vs_5050_p"] = stats.srm_test(int(nt), int(nc), 0.5).p_value

    # cross-fitted nuisance scores from scripts/nuisance.py (propensity, outcome models, AIPW psi)
    c.execute(f"CREATE VIEW dn AS SELECT * FROM d JOIN read_parquet('{ROOT / 'data' / 'nuisance.parquet'}') USING (rid)")
    assert c.execute("SELECT count(*) FROM dn").fetchone()[0] == res["source_rows"]

    # ---------------------------------------------------------------- balance
    bal = []
    for f in FEATS:
        mt, vt, mc, vc = c.execute(
            f"SELECT avg({f}) FILTER (treatment=1), var_samp({f}) FILTER (treatment=1), "
            f"avg({f}) FILTER (treatment=0), var_samp({f}) FILTER (treatment=0) FROM d").fetchone()
        smd = (mt - mc) / np.sqrt((vt + vc) / 2)
        e = stats.diff_in_means(mt, vt, nt, mc, vc, nc)
        # after inverse-propensity weighting with the cross-fitted e(x)
        wt, wc = c.execute(
            f"SELECT sum({f} / e_hat) FILTER (treatment=1) / sum(1 / e_hat) FILTER (treatment=1), "
            f"sum({f} / (1 - e_hat)) FILTER (treatment=0) / sum(1 / (1 - e_hat)) FILTER (treatment=0) FROM dn").fetchone()
        bal.append({"feature": f, "mean_t": mt, "mean_c": mc, "smd": float(smd), "p_value": e.p_value,
                    "smd_weighted": float((wt - wc) / np.sqrt((vt + vc) / 2))})
    res["balance"] = bal
    res["balance_max_abs_smd"] = max(abs(b["smd"]) for b in bal)
    res["balance_max_abs_smd_weighted"] = max(abs(b["smd_weighted"]) for b in bal)
    res["balance_n_sig_holm"] = int(sum(stats.holm([b["p_value"] for b in bal]) < 0.05))
    # what chance alone would produce: SMD standard error for these arm sizes
    res["balance_smd_null_sd"] = float(np.sqrt(1 / nt + 1 / nc))

    # propensity diagnostics: out-of-fold AUC of e(x) for predicting assignment
    sm = c.execute("SELECT e_hat, treatment FROM dn USING SAMPLE 3000000 ROWS (reservoir, 5)").fetchnumpy()
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(sm["treatment"], sm["e_hat"])
    rng = np.random.default_rng(0)
    null_auc = [roc_auc_score(rng.permutation(sm["treatment"]), sm["e_hat"]) for _ in range(20)]
    res["propensity"] = {
        "auc_out_of_fold": float(auc), "auc_permutation_max": float(max(null_auc)),
        "auc_permutation_sd": float(np.std(null_auc)),
        "quantiles": dict(zip(["p01", "p05", "p25", "p50", "p75", "p95", "p99"],
                              [float(x) for x in np.quantile(sm["e_hat"], [.01, .05, .25, .5, .75, .95, .99])])),
        "share_outside_80_90": float(((sm["e_hat"] < 0.8) | (sm["e_hat"] > 0.9)).mean()),
    }
    del sm

    # ---------------------------------------------------------------- ATE
    ate = {}
    for y in OUTCOMES:
        kt, kc = c.execute(f"SELECT sum({y}) FILTER (treatment=1), sum({y}) FILTER (treatment=0) FROM d").fetchone()
        pt, pc = kt / nt, kc / nc
        vt, vc = pt * (1 - pt) * nt / (nt - 1), pc * (1 - pc) * nc / (nc - 1)
        a = stats.diff_in_means(pt, vt, nt, pc, vc, nc)
        r = stats.relative_lift(pt, vt, nt, pc, vc, nc)
        b = stats.bootstrap_binary_ate(int(kt), int(nt), int(kc), int(nc), n_boot=10_000, seed=1)
        ate[y] = {"k_t": int(kt), "k_c": int(kc), "rate_t": pt, "rate_c": pc,
                  "ate": a.to_dict(), "relative_lift": r.to_dict(), "bootstrap": b}
    res["ate"] = ate

    # exposure: noncompliance. Control users cannot be exposed; the Wald ratio is the
    # effect on users who actually saw an ad (valid under exclusion restriction).
    et, ec = c.execute("SELECT avg(exposure) FILTER (treatment=1), avg(exposure) FILTER (treatment=0) FROM d").fetchone()
    res["exposure"] = {"rate_t": et, "rate_c": ec}
    for y in OUTCOMES:
        a = ate[y]["ate"]
        res["exposure"][f"wald_{y}"] = {"estimate": a["estimate"] / (et - ec),
                                        "ci_low": a["ci_low"] / (et - ec), "ci_high": a["ci_high"] / (et - ec)}

    # ---------------------------------------------------------------- AIPW (doubly robust)
    aipw = {}
    for y in OUTCOMES:
        # psi1/psi0 are the AIPW scores for each potential-outcome mean
        n, m1, m0, v1, v0, c10, mpsi, vpsi = c.execute(f"""
            WITH s AS (SELECT mu1_{y} + treatment * ({y} - mu1_{y}) / e_hat AS p1,
                              mu0_{y} + (1 - treatment) * ({y} - mu0_{y}) / (1 - e_hat) AS p0,
                              psi_{y} AS psi FROM dn)
            SELECT count(*), avg(p1), avg(p0), var_samp(p1), var_samp(p0), covar_samp(p1, p0),
                   avg(psi), var_samp(psi) FROM s""").fetchone()
        eff = stats.effect(mpsi, np.sqrt(vpsi / n))
        # relative lift m1/m0 - 1, delta method on log ratio
        se_log = np.sqrt((v1 / m1**2 + v0 / m0**2 - 2 * c10 / (m1 * m0)) / n)
        z = 1.959963984540054
        lr = np.log(m1 / m0)
        aipw[y] = {"ate": eff.to_dict(), "mean_y1": m1, "mean_y0": m0,
                   "relative_lift": {"estimate": m1 / m0 - 1, "ci_low": float(np.exp(lr - z * se_log) - 1),
                                     "ci_high": float(np.exp(lr + z * se_log) - 1)}}
        # effect on the exposed (Wald) using the doubly robust ITT
        aipw[y]["wald_exposed"] = {k: eff.to_dict()[k] / (et - ec) for k in ("estimate", "ci_low", "ci_high")}
    res["aipw"] = aipw

    # ---------------------------------------------------------------- variance reduction
    cupac_scores(c)
    c.execute(f"CREATE OR REPLACE VIEW ds AS SELECT d.*, s.s_visit, s.s_conversion FROM "
              f"d JOIN read_parquet('{ROOT / 'data' / 'cupac.parquet'}') s USING (rid)")
    vr = {}
    for y in OUTCOMES:
        raw = ate[y]["ate"]
        lin_m = arm_moments(c, y, FEATS)
        cuped = stats.cuped_from_moments(lin_m, per_arm_theta=False)
        lin = stats.cuped_from_moments(lin_m, per_arm_theta=True)
        cm = arm_moments(c, y, [f"s_{y}"], table="ds")
        cupac = stats.cuped_from_moments(cm, per_arm_theta=False)
        rows = {}
        for name, e in [("difference in means", raw), ("CUPED, 12 linear features", cuped.to_dict()),
                        ("Lin interacted regression", lin.to_dict()), ("CUPAC, cross-fitted GBM score", cupac.to_dict()),
                        ("AIPW, cross-fitted GBM", aipw[y]["ate"])]:
            rows[name] = {**e, "variance_ratio": (e["se"] / raw["se"]) ** 2,
                          "ci_width_ratio": (e["ci_high"] - e["ci_low"]) / (raw["ci_high"] - raw["ci_low"])}
        vr[y] = rows
    res["variance_reduction"] = vr

    # ---------------------------------------------------------------- heterogeneity by feature decile
    # Segment effect = mean AIPW score within the segment (valid because psi is
    # an unbiased per-row signal of the individual effect given X).
    hte = []
    for f in FEATS:
        qs = c.execute(f"SELECT quantile_cont({f}, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) FROM d").fetchone()[0]
        cuts = sorted(set(round(q, 12) for q in qs))
        case = "CASE " + " ".join(f"WHEN {f} <= {q!r} THEN {i}" for i, q in enumerate(cuts)) + f" ELSE {len(cuts)} END"
        rows = c.execute(
            f"SELECT {case} AS bin, min({f}), max({f}), count(*), "
            f"avg(psi_visit), var_samp(psi_visit), avg(psi_conversion), var_samp(psi_conversion), "
            f"avg(conversion) FILTER (treatment=0) "
            f"FROM dn GROUP BY bin ORDER BY bin").fetchall()
        for b, lo, hi, n, mv, vv, mcv, vcv, rc in rows:
            for y, m, v in (("visit", mv, vv), ("conversion", mcv, vcv)):
                e = stats.effect(m, np.sqrt(v / n))
                hte.append({"feature": f, "bin": int(b), "lo": lo, "hi": hi, "n": n,
                            "share": n / (nt + nc), "outcome": y, "conv_rate_c": rc, **e.to_dict()})
    pv = [h["p_value"] for h in hte]
    for h, ph, pb in zip(hte, stats.holm(pv), stats.benjamini_hochberg(pv)):
        h["p_holm"], h["p_bh"] = float(ph), float(pb)
    # a heterogeneity test per feature: chi-square across bins of (tau_b - tau_pooled)
    het = []
    for f in FEATS:
        for y in OUTCOMES:
            hs = [h for h in hte if h["feature"] == f and h["outcome"] == y]
            if len(hs) < 2:
                continue   # feature has one dominant value; no split possible
            est, se = np.array([h["estimate"] for h in hs]), np.array([h["se"] for h in hs])
            w = 1 / se**2
            pooled = (w * est).sum() / w.sum()
            q = float((w * (est - pooled) ** 2).sum())
            het.append({"feature": f, "outcome": y, "bins": len(hs), "Q": q,
                        "p_value": float(sps.chi2.sf(q, len(hs) - 1)),
                        "range": [float(est.min()), float(est.max())]})
    hp = [h["p_value"] for h in het]
    for h, ph in zip(het, stats.holm(hp)):
        h["p_holm"] = float(ph)
    res["hte_bins"] = hte
    res["hte_tests"] = het
    res["hte_summary"] = {
        "n_tests": len(hte),
        "n_sig_raw": sum(h["p_value"] < 0.05 for h in hte),
        "n_sig_holm": sum(h["p_holm"] < 0.05 for h in hte),
        "n_sig_bh": sum(h["p_bh"] < 0.05 for h in hte),
        "n_negative_sig_holm": sum(h["p_holm"] < 0.05 and h["estimate"] < 0 for h in hte),
        "n_het_features_holm": sum(h["p_holm"] < 0.05 for h in het),
        "n_het_tests": len(het),
        "features_with_bins": sorted({h["feature"] for h in het}),
    }
    res["runtime_s"] = time.time() - t0
    OUT.mkdir(exist_ok=True)
    (OUT / "readout.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("srm", "balance_max_abs_smd", "hte_summary", "runtime_s")}, indent=1, default=float))
    for y in OUTCOMES:
        print(y, {k: round(v["variance_ratio"], 4) for k, v in vr[y].items()})


if __name__ == "__main__":
    main()
