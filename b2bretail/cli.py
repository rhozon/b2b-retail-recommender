"""Run the whole case: python -m b2bretail.cli [--raw-dir DIR] [--out results] [--site DIR] [--quick]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import assortment, data, export, impute, lake, metrics, recs, sasrec, segment, seqmodel, tracking, uplift
from .split import CUTOFF, VAL_CUTOFF, cold_share, make_split, targets


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_scores(sc: np.ndarray, users: np.ndarray, tg: list, split, new_only: bool, k: int = 10):
    excl = recs.seen(split, users) if new_only else None
    tops = metrics.top_k(sc, k, excl)
    return tops, metrics.per_user(tops, tg, k)


def tune(sales: pd.DataFrame, quick: bool) -> dict:
    """Pick settings on the validation window (the 61 days before the test window)."""
    sv = make_split(sales, VAL_CUTOFF, end=CUTOFF)
    users, tg = targets(sv, new_only=True)
    best = {}
    res = {}
    for nb in (50, 200):
        _, (_, nd) = evaluate_scores(recs.item_knn(sv, users, nb), users, tg, sv, True)
        res[f"knn_{nb}"] = float(np.mean(nd))
    best["neighbours"] = max((50, 200), key=lambda nb: res[f"knn_{nb}"])
    grid = [(32, 10.0), (64, 10.0), (64, 40.0)] if not quick else [(16, 10.0)]
    for fac, alpha in grid:
        X, Y = recs.als(sv.R, factors=fac, alpha=alpha, iters=10 if quick else 15)
        _, (_, nd) = evaluate_scores(recs.als_scores(X, Y, users), users, tg, sv, True)
        res[f"als_{fac}_{alpha:g}"] = float(np.mean(nd))
    best["als"] = max(grid, key=lambda g: res[f"als_{g[0]}_{g[1]:g}"])
    seqs = seqmodel.basket_sequences(sv.train, sv.users, sv.items)
    curve = {}

    def on_epoch(model, ep):
        sc = seqmodel.scores(model, seqs, users)
        _, (_, nd) = evaluate_scores(sc, users, tg, sv, True)
        curve[ep] = float(np.mean(nd))
        log(f"  sequence model epoch {ep}: validation NDCG@10 {curve[ep]:.4f}")

    seqmodel.fit(seqs, len(sv.items), epochs=2 if quick else 30, on_epoch=on_epoch)
    best["epochs"] = max(curve, key=curve.get)
    res.update({f"seq_epoch_{e}": v for e, v in curve.items()})

    def curve_for(name, fitter, score, n_epochs):
        c = {}

        def cb(model, ep):
            _, (_, nd) = evaluate_scores(score(model), users, tg, sv, True)
            c[ep] = float(np.mean(nd))
            log(f"  {name} epoch {ep}: validation NDCG@10 {c[ep]:.4f}")

        fitter(cb, n_epochs)
        return c

    iseqs = sasrec.item_sequences(seqs)
    c1 = curve_for("SASRec (items)", lambda cb, n: sasrec.fit(iseqs, len(sv.items), epochs=n, max_len=100, on_epoch=cb),
                   lambda m: sasrec.scores(m, iseqs, users), 2 if quick else 12)
    c2 = curve_for("SASRec (baskets)", lambda cb, n: sasrec.fit_baskets(seqs, len(sv.items), epochs=n, on_epoch=cb),
                   lambda m: seqmodel.scores(m, seqs, users), 2 if quick else 40)
    best["sasrec_items_epochs"] = max(c1, key=c1.get)
    best["sasrec_baskets_epochs"] = max(c2, key=c2.get)
    res.update({f"sasrec_items_epoch_{e}": v for e, v in c1.items()})
    res.update({f"sasrec_baskets_epoch_{e}": v for e, v in c2.items()})
    return {"best": best, "validation_ndcg": res}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(data.RAW_DIR))
    ap.add_argument("--out", default="results")
    ap.add_argument("--site", default=None, help="folder for the simulator data (assets/case10)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reuse-uplift", action="store_true", help="keep the promotion results already in --out")
    ap.add_argument("--lake", default=None, help="folder for the bronze, silver and gold Delta tables")
    ap.add_argument("--track", default=None, help="MLflow tracking URI, e.g. sqlite:///mlflow.db")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    log("loading")
    raw = data.load_raw(a.raw_dir)
    q = data.quality_report(raw)
    dup = raw[raw.duplicated(keep="first")]
    overlap = dup["invoice_date"].between("2010-12-01", "2010-12-10").sum()
    q = pd.concat([q, pd.DataFrame([{"rule": "exact duplicates inside the 1-9 Dec 2010 sheet overlap",
                                       "lines": int(overlap), "share": float(overlap / len(raw))}])])
    q.to_csv(out / "quality.csv", index=False)
    sales = data.clean_sales(raw)
    desc = data.descriptions(sales)
    summary["data"] = {"lines": int(len(raw)), "sale_lines": int(len(sales)),
                       "first": str(sales["invoice_date"].min().date()), "last": str(sales["invoice_date"].max().date()),
                       "customers": int(sales["customer_id"].nunique()), "products": int(sales["stock_code"].nunique()),
                       "invoices": int(sales["invoice"].nunique()),
                       "missing_customer_line_share": float(sales["customer_id"].isna().mean()),
                       "missing_customer_revenue_share": float(sales.loc[sales["customer_id"].isna(), "revenue"].sum()
                                                               / sales["revenue"].sum())}

    log("tuning on the validation window")
    tuned = tune(sales, a.quick)
    summary["tuning"] = tuned
    b = tuned["best"]

    log("fitting on everything before the cutoff")
    sp = make_split(sales, CUTOFF)
    summary["split"] = {"cutoff": str(CUTOFF.date()), "train_customers": int(len(sp.users)),
                        "train_products": int(len(sp.items)), "cold_revenue_share": cold_share(sp)}
    S = recs.item_similarity(sp.R, b["neighbours"])
    fac, alpha = b["als"]
    X, Y = recs.als(sp.R, factors=fac, alpha=alpha, iters=10 if a.quick else 15)
    seqs = seqmodel.basket_sequences(sp.train, sp.users, sp.items)
    model = seqmodel.fit(seqs, len(sp.items), epochs=b["epochs"])
    iseqs = sasrec.item_sequences(seqs)
    sas_i = sasrec.fit(iseqs, len(sp.items), epochs=b["sasrec_items_epochs"], max_len=100)
    sas_b = sasrec.fit_baskets(seqs, len(sp.items), epochs=b["sasrec_baskets_epochs"])

    rows, per = [], {}
    for mode, new_only in (("new to the customer", True), ("all products", False)):
        users, tg = targets(sp, new_only)
        sc = {"Popularity (last 90 days)": recs.popularity(sp, users),
              "Item-to-item (co-purchase)": recs.item_knn(sp, users, S=S),
              "Implicit ALS": recs.als_scores(X, Y, users),
              "Sequential GRU (PyTorch)": seqmodel.scores(model, seqs, users),
              "SASRec, product by product": sasrec.scores(sas_i, iseqs, users),
              "SASRec, basket by basket": seqmodel.scores(sas_b, seqs, users)}
        if not new_only:
            sc["Buy again"] = recs.buy_again(sp, users)
        for name, s in sc.items():
            tops, (rec, nd) = evaluate_scores(s, users, tg, sp, new_only)
            per[(mode, name)] = (users, tops, rec, nd)
            r_m, r_lo, r_hi = metrics.bootstrap_ci(rec)
            n_m, n_lo, n_hi = metrics.bootstrap_ci(nd)
            rows.append({"mode": mode, "model": name, "customers": len(users), "recall@10": r_m, "recall_lo": r_lo,
                         "recall_hi": r_hi, "ndcg@10": n_m, "ndcg_lo": n_lo, "ndcg_hi": n_hi})
            log(f"  {mode:22s} {name:28s} NDCG@10 {n_m:.4f} [{n_lo:.4f}, {n_hi:.4f}]")
    rm = pd.DataFrame(rows)
    rm.to_csv(out / "recs_metrics.csv", index=False)
    comp = {}
    for mode in rm["mode"].unique():
        m = rm[rm["mode"] == mode].sort_values("ndcg@10", ascending=False)
        first, second = m["model"].iat[0], m["model"].iat[1]
        d = per[(mode, first)][3] - per[(mode, second)][3]
        dm, dlo, dhi = metrics.bootstrap_ci(d)
        comp[mode] = {"best": first, "runner_up": second, "ndcg_diff": dm, "diff_lo": dlo, "diff_hi": dhi}
    summary["recs"] = comp

    log("segments")
    f = segment.customer_features(sp.train, CUTOFF)
    seg, prof, info = segment.fit_segments(f)
    prof.to_csv(out / "segment_profile.csv")
    seg.to_csv(out / "segments.csv")
    summary["segments"] = {"k": info["k"], "silhouette": info["silhouette"]}

    log("imputation of missing customer ids")
    hist = sp.train
    hold = sp.test
    ev = impute.evaluate(hist, hold)
    unknown = sales[sales["customer_id"].isna()]
    known_inv = sales.dropna(subset=["customer_id"]).groupby("invoice").agg(lines=("stock_code", "size"),
                                                                            price=("price", "median"))
    unk_inv = unknown.groupby("invoice").agg(lines=("stock_code", "size"), price=("price", "median"))
    ev["missing_invoices"] = int(len(unk_inv))
    ev["missing_median_lines"] = float(unk_inv["lines"].median())
    ev["known_median_lines"] = float(known_inv["lines"].median())
    ev["missing_median_price"] = float(unk_inv["price"].median())
    ev["known_median_price"] = float(known_inv["price"].median())
    thr = [c for c in ev["confidence_curve"] if c["coverage"] == 0.25][0]["threshold"]
    att = impute.attribute(hist, unknown[unknown["invoice_date"] >= CUTOFF], thr)
    ev["window_missing_invoices"] = int(len(att))
    ev["window_confident"] = int(att["confident"].sum())
    (out / "imputation.json").write_text(json.dumps(ev, indent=2))
    summary["imputation"] = {k: v for k, v in ev.items() if k != "confidence_curve"}

    log("assortment")
    fam = assortment.families(sp.items, desc)
    fam.to_csv(out / "families.csv", index=False)
    family = fam.set_index("stock_code")["family"]
    overall = sp.train.groupby("stock_code")["revenue"].sum().sort_values(ascending=False)
    arows, items_rows = [], []
    for name in prof.index:
        members = seg.index[seg == name]
        exp = assortment.expected_revenue(sp.train, members, CUTOFF)
        chosen = assortment.choose(exp, family, k=30, per_family=4)
        plans = {"Optimised (family cap)": chosen,
                 "Segment top sellers": list(exp.sort_values(ascending=False).index[:30]),
                 "Overall top sellers": list(overall.index[:30])}
        for plan, codes in plans.items():
            cv = assortment.coverage(sp.test, members, codes, family)
            arows.append({"segment": name, "plan": plan, **cv})
        for rank, c in enumerate(chosen, 1):
            items_rows.append({"segment": name, "rank": rank, "stock_code": c, "description": desc.get(c, ""),
                               "family": family.get(c, ""), "expected_revenue_180d": float(exp.get(c, 0.0))})
    pd.DataFrame(arows).to_csv(out / "assortment.csv", index=False)
    pd.DataFrame(items_rows).to_csv(out / "assortment_items.csv", index=False)

    log("promotion targeting (semi-synthetic)")
    y0 = sp.test.groupby("customer_id")["revenue"].sum()
    if a.reuse_uplift and (out / "uplift_curves.csv").exists():
        # the promotion step does not depend on the recommenders; reuse its files
        curves, areas = pd.read_csv(out / "uplift_curves.csv"), pd.read_csv(out / "uplift_area.csv")
    else:
        curves, areas = uplift.run(f, y0, reps=3 if a.quick else 10)
        curves.to_csv(out / "uplift_curves.csv", index=False)
        areas.to_csv(out / "uplift_area.csv", index=False)
        sens = uplift.sensitivity(f, y0, reps=2 if a.quick else 5)
        sens.to_csv(out / "uplift_sensitivity.csv", index=False)
    summary["uplift"] = areas.set_index("policy")["mean"].to_dict()

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if a.lake:
        log("writing Delta tables")
        summary["lake"] = lake.write_all(Path(a.lake), raw, sales, f, seg, rm)
    if a.track:
        log("logging the run to MLflow")
        summary["mlflow_run"] = tracking.log_run(a.track, summary, rm, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if a.site:
        log("exporting simulator data")
        export.site_data(Path(a.site), sp, desc, seg, prof, f, S, per, rm, pd.read_csv(out / "assortment.csv"),
                         pd.DataFrame(items_rows), curves, areas, summary)
    log("done")


if __name__ == "__main__":
    main()
