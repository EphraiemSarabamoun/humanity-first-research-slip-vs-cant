"""analyze_slip.py — can a linear probe on pre-answer hidden states separate
'cannot solve' from 'solved but execution-slipped'?

Labels (greedy-wrong items only): slip = at least one of K samples correct; cannot = none.
Per layer: 5-fold OOF logistic-regression AUC for slip-vs-cannot, plus correct-vs-wrong
(sanity) and a shuffle-label null at the best layer. Writes curve.csv, eval_points.jsonl,
analysis_summary.txt (via recompute.py), figures.
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SEED = 42


def probe_scores(X, y, seed=SEED):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    # PCA fits per-fold on ~80% of rows; keep k safely below the smallest train fold
    k = min(50, X.shape[1], int(X.shape[0] * 0.8) - 2)
    clf = make_pipeline(StandardScaler(), PCA(k, random_state=seed), LogisticRegression(max_iter=2000))
    return cross_val_predict(clf, X, y, cv=skf, method="decision_function")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out_dir", default="results/real"); args = ap.parse_args()
    out = Path(args.out_dir)
    d = np.load(out / "slip_data.npz")
    layers = [int(x) for x in d["layers"]]
    gc = d["greedy_correct"].astype(int); sc = d["samp_correct"].astype(int)
    wrong = gc == 0
    slip = wrong & (sc > 0); cant = wrong & (sc == 0)
    sc_idx = np.where(slip | cant)[0]
    y_sc = slip[sc_idx].astype(int)  # 1 = slip, 0 = cannot
    print("[labels] correct=%d wrong=%d slip=%d cannot=%d" % (gc.sum(), wrong.sum(), slip.sum(), cant.sum()), flush=True)

    rows = []
    per_layer_sc, per_layer_cw = {}, {}
    sc_scores_by_layer, cw_scores_by_layer = {}, {}
    for li in layers:
        H = d["h_L%02d" % li].astype(np.float64)
        # slip vs cannot
        if y_sc.sum() >= 5 and (len(y_sc)-y_sc.sum()) >= 5:
            s = probe_scores(H[sc_idx], y_sc)
            auc = roc_auc_score(y_sc, s); per_layer_sc[li] = auc; sc_scores_by_layer[li] = s
        else:
            per_layer_sc[li] = float("nan")
        # correct vs wrong (sanity)
        cw = probe_scores(H, gc); cwa = roc_auc_score(gc, cw); per_layer_cw[li] = cwa; cw_scores_by_layer[li] = cw
        rows.append(("slip_vs_cant_auc_L%02d" % li, per_layer_sc[li], len(sc_idx)))
        rows.append(("correct_vs_wrong_auc_L%02d" % li, cwa, len(gc)))
        print("[L%02d] slip_vs_cant=%.3f correct_vs_wrong=%.3f" % (li, per_layer_sc[li], cwa), flush=True)

    valid = {li: a for li, a in per_layer_sc.items() if a == a}
    best = max(valid, key=valid.get) if valid else layers[len(layers)//2]
    # eval_points for best layer
    ep = []
    for k, i in enumerate(sc_idx):
        ep.append({"section": "slip_vs_cant", "eval_order": k, "score": float(sc_scores_by_layer[best][k]), "label": int(y_sc[k])})
    for k in range(len(gc)):
        ep.append({"section": "correct_vs_wrong", "eval_order": k, "score": float(cw_scores_by_layer[best][k]), "label": int(gc[k])})
    with open(out / "eval_points.jsonl", "w") as f:
        for e in ep: f.write(json.dumps(e) + "\n")
    # shuffle null at best layer for slip_vs_cant
    Hbest = d["h_L%02d" % best].astype(np.float64)[sc_idx]
    rng = np.random.default_rng(SEED); nulls = []
    if y_sc.sum() >= 5 and (len(y_sc)-y_sc.sum()) >= 5:
        for _ in range(100):
            yp = rng.permutation(y_sc)
            nulls.append(roc_auc_score(yp, probe_scores(Hbest, yp, seed=int(rng.integers(1 << 30)))))
    import csv as _csv
    with open(out / "null_dist.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["key", "draw", "auc"])
        for i, v in enumerate(nulls): w.writerow(["slip_vs_cant_null", i, "%.6f" % v])
    # curve.csv
    with open(out / "curve.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["metric", "value", "n"])
        for sec, v, n in rows: w.writerow([sec, "%.6f" % v if v == v else "nan", n])
        w.writerow(["best_layer", best, len(sc_idx)])
        w.writerow(["n_correct", int(gc.sum()), len(gc)])
        w.writerow(["n_wrong", int(wrong.sum()), len(gc)])
        w.writerow(["n_slip", int(slip.sum()), len(gc)])
        w.writerow(["n_cannot", int(cant.sum()), len(gc)])
        w.writerow(["greedy_accuracy", "%.6f" % gc.mean(), len(gc)])
        if nulls: w.writerow(["slip_vs_cant_null_mean", "%.6f" % float(np.mean(nulls)), len(nulls)])
    with open(out / "analysis_summary.txt", "w") as f:
        subprocess.run(["python3", "recompute.py"], cwd=str(out), stdout=f, check=True)
    make_figs(per_layer_sc, per_layer_cw, layers, best, nulls, sc_scores_by_layer, y_sc,
              int(gc.sum()), int(slip.sum()), int(cant.sum()), out)
    print("[analyze] best_layer=%d slip_vs_cant_AUC=%.3f" % (best, per_layer_sc.get(best, float("nan"))), flush=True)


def make_figs(sc, cw, layers, best, nulls, sc_scores, y_sc, n_correct, n_slip, n_cant, out):
    # Fig1: slip_vs_cant AUC across layers + chance + null band
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(layers, [sc[li] for li in layers], "o-", color="#d95f02", label="slip vs cannot")
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
    if nulls:
        lo, hi = np.percentile(nulls, [2.5, 97.5])
        ax.axhspan(lo, hi, color="#fdae6b", alpha=0.35, label="shuffle null (95%%)")
    ax.set_xlabel("residual-stream layer"); ax.set_ylabel("5-fold OOF AUC"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Pre-answer separability of slip vs cannot-solve (n_slip=%d, n_cannot=%d)" % (n_slip, n_cant)); ax.legend()
    fig.tight_layout(); fig.savefig(out/"figure_main.png", dpi=150); plt.close(fig)
    # Fig2: correct vs wrong AUC across layers
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(layers, [cw[li] for li in layers], "s-", color="#1b9e77", label="correct vs wrong (sanity)")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("residual-stream layer"); ax.set_ylabel("5-fold OOF AUC"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Pre-answer correct-vs-wrong decodability (sanity check)"); ax.legend()
    fig.tight_layout(); fig.savefig(out/"figure_correct_vs_wrong.png", dpi=150); plt.close(fig)
    # Fig3: label counts
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.bar([0,1,2], [n_correct, n_slip, n_cant], color=["#2c7fb8","#d95f02","#7570b3"])
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["correct", "slip\n(wrong, can solve)", "cannot\n(wrong, never)"]); ax.set_ylabel("count")
    ax.set_title("GSM8K outcome labels (Qwen2.5-3B, K=6 samples)")
    fig.tight_layout(); fig.savefig(out/"figure_counts.png", dpi=150); plt.close(fig)
    # Fig4: best-layer slip_vs_cant vs null distribution
    fig, ax = plt.subplots(figsize=(6, 4.2))
    if nulls:
        ax.hist(nulls, bins=20, color="#bdbdbd", label="shuffle null")
    ax.axvline(sc[best], color="#d95f02", lw=2, label="observed AUC=%.3f (L%d)" % (sc[best], best))
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("AUC"); ax.set_ylabel("count"); ax.set_title("Best-layer slip-vs-cannot AUC vs shuffle null"); ax.legend()
    fig.tight_layout(); fig.savefig(out/"figure_null.png", dpi=150); plt.close(fig)
    print("[figures] wrote 4", flush=True)


if __name__ == "__main__":
    main()
