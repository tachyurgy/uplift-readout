"""Part 2: power, MDE and the cost of peeking.

Baselines come from the control arm measured in results/readout.json, so the
calculator is anchored to this campaign rather than to made-up rates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

from uplift_readout import sequential, stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def main():
    r = json.loads((OUT / "readout.json").read_text())
    n_total = r["srm"]["n_treat"] + r["srm"]["n_control"]
    res: dict = {"n_total": n_total}

    # ------------------------------------------------ MDE / sample size
    rel_grid = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
    calc = {}
    for y in ("visit", "conversion"):
        pc = r["ate"][y]["rate_c"]
        rows = []
        for rel in rel_grid:
            rows.append({"rel_mde": rel,
                         "n_5050": stats.sample_size_two_prop(pc, rel, 0.5),
                         "n_8515": stats.sample_size_two_prop(pc, rel, 0.85)})
        observed_rel = r["aipw"][y]["relative_lift"]["estimate"]
        calc[y] = {
            "baseline": pc,
            "table": rows,
            "mde_abs_this_n_8515": stats.mde_two_prop(pc, n_total, 0.85),
            "mde_abs_this_n_5050": stats.mde_two_prop(pc, n_total, 0.5),
            "observed_rel_lift": observed_rel,
            "n_for_observed_8515": stats.sample_size_two_prop(pc, observed_rel, 0.85),
            "n_for_observed_5050": stats.sample_size_two_prop(pc, observed_rel, 0.5),
            "cuped_variance_ratio": r["variance_reduction"][y]["CUPAC, cross-fitted GBM score"]["variance_ratio"],
        }
        calc[y]["mde_rel_this_n_8515"] = calc[y]["mde_abs_this_n_8515"] / pc
        calc[y]["n_for_observed_8515_cupac"] = int(np.ceil(calc[y]["n_for_observed_8515"] * calc[y]["cuped_variance_ratio"]))
    # the imbalance penalty: variance factor of 85/15 vs 50/50 = (1/.85 + 1/.15) / 4
    res["allocation_penalty"] = (1 / 0.85 + 1 / 0.15) / 4
    res["power"] = calc

    # ------------------------------------------------ peeking
    n_sims = 100_000
    looks_grid = [1, 2, 3, 5, 10, 20, 50, 100]
    fpr_by_looks = []
    for k in looks_grid:
        z = sequential.simulate_z_paths(n_sims, k, 0.0, seed=100 + k)
        rej = sequential.naive_peeking_reject(z)
        fpr_by_looks.append({"looks": k, "fpr": float(rej.mean()),
                             "se": float(np.sqrt(rej.mean() * (1 - rej.mean()) / n_sims))})
        del z
    res["peeking_fpr_by_looks"] = fpr_by_looks

    K = 20
    drift = sps.norm.ppf(0.975) + sps.norm.ppf(0.8)      # 80% power at the planned horizon
    tau2 = (drift / np.sqrt(K)) ** 2                     # mSPRT prior scaled to the planned effect
    bounds = sequential.spending_boundaries(K, n_calib=400_000)
    z0 = sequential.simulate_z_paths(n_sims, K, 0.0, seed=7)
    z1 = sequential.simulate_z_paths(n_sims, K, drift, seed=8)
    crit = sps.norm.ppf(0.975)
    fixed = np.full(K, np.inf); fixed[-1] = crit
    naive = np.full(K, crit)
    k_idx = np.arange(1, K + 1)
    msprt_bound = np.sqrt((2 * (1 + k_idx * tau2) / tau2) * (np.log(20) - 0.5 * np.log(1 / (1 + k_idx * tau2))) / k_idx)
    methods = {
        "Fixed horizon (look once)": fixed,
        "Naive peeking (z > 1.96 at any look)": naive,
        "O'Brien-Fleming alpha spending": bounds,
        "mSPRT (always-valid)": msprt_bound,
    }
    seq = []
    for name, b in methods.items():
        h0 = sequential.spending_reject(z0, b)
        h1 = sequential.spending_reject(z1, b)
        stop = sequential.first_crossing_look(z1, b)
        exp_n = np.where(stop > 0, stop, K).mean() / K
        seq.append({"method": name, "fpr": float(h0.mean()), "power": float(h1.mean()),
                    "expected_sample_fraction_h1": float(exp_n), "bounds": [float(x) for x in b]})
    # sanity: the closed-form mSPRT boundary agrees with the likelihood-ratio rule
    assert abs(sequential.msprt_reject(z0, tau2).mean() - seq[3]["fpr"]) < 1e-9
    # what always-valid monitoring costs: let the mSPRT run past the planned horizon
    ext = []
    for mult in (1.0, 1.5, 2.0, 3.0):
        Km = int(round(K * mult))
        # same per-look drift as the planned design, so the planned horizon is look K
        z1m = sequential.simulate_z_paths(n_sims, Km, drift * np.sqrt(Km / K), seed=9)
        km = np.arange(1, Km + 1)
        bm = np.sqrt((2 * (1 + km * tau2) / tau2) * (np.log(20) - 0.5 * np.log(1 / (1 + km * tau2))) / km)
        stop = sequential.first_crossing_look(z1m, bm)
        ext.append({"max_sample_multiple": mult, "power": float((stop > 0).mean()),
                    "expected_sample_multiple": float(np.where(stop > 0, stop, Km).mean() / K)})
        del z1m
    res["msprt_extended"] = ext
    res["sequential"] = {"looks": K, "n_sims": n_sims, "drift": float(drift), "tau2": float(tau2), "methods": seq}
    (OUT / "power.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "power"}, indent=1)[:3000])
    for y in calc:
        print(y, {k: v for k, v in calc[y].items() if k != "table"})


if __name__ == "__main__":
    main()
