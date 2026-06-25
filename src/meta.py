import csv, json
from pathlib import Path
OUT = Path("results/real"); GIT = "bea9a6f3c1aea99791fe9fe07c567efdd5d8551c"
rows = list(csv.DictReader(open(OUT/"curve.csv")))

def sidecar(name, columns, ac):
    (OUT/(name+".md")).write_text(
"""# %s — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py (data from src/extract_slip.py)
Command: python3 src/analyze.py --out_dir results/real
Git-Commit: %s
Seeds: 42 (GSM8K sampling in extract_slip.py; greedy decoding deterministic; K=6 sampled generations at temperature 0.8 top-p 0.95; 5-fold StratifiedKFold probe; per-fold PCA(<=50); 100 shuffle-null draws; 2000-resample percentile bootstrap)
Source-Data: GSM8K (openai/gsm8k, main, test), 800 problems via datasets; Qwen2.5-3B-Instruct, RTX 5090, 2026-06-24, torch 2.12 cu130; pre-answer hidden states (last token before the '####' answer marker) at layers 14/20/25/31/36 in results/real/slip_data.npz
Analysis-Command: %s
Columns:
%s
""" % (name, GIT, ac, columns))

sidecar("curve.csv",
        "  metric (slip_vs_cant_auc_L<layer> = 5-fold OOF AUC separating slip from cannot at that layer; correct_vs_wrong_auc_L<layer> = sanity probe AUC; best_layer = layer with max slip-vs-cant AUC; n_correct/n_wrong/n_slip/n_cannot = label counts; greedy_accuracy; slip_vs_cant_null_mean = mean shuffle-label AUC);\n"
        "  value (AUC 0-1, count, or layer index); n (items entering that metric: 136 for slip-vs-cant, 800 for correct-vs-wrong and counts, 100 for the null)",
        "cd results/real && python3 recompute.py | diff - analysis_summary.txt  (empty); the headline slip-vs-cant and correct-vs-wrong AUCs are reproduced from eval_points.jsonl, the null from null_dist.csv; per-layer rows here back the figures")
sidecar("eval_points.jsonl",
        "  section (slip_vs_cant or correct_vs_wrong, both at the best layer); eval_order (position); score (5-fold OOF logistic-regression decision-function value); label (slip_vs_cant: 1=slip 0=cannot; correct_vs_wrong: 1=greedy-correct 0=wrong)",
        "cd results/real && python3 recompute.py  (each section AUC = rank-based AUC with 95% bootstrap CI)")
sidecar("null_dist.csv",
        "  key (slip_vs_cant_null); draw (0-99); auc (AUC of the slip-vs-cant probe at the best layer after shuffling the labels, unitless 0-1)",
        "cd results/real && python3 recompute.py  (mean and 95% interval of these are the shuffle null)")

def w(stem, pred, desc):
    rs=[r for r in rows if pred(r["metric"])]
    with open(OUT/(stem+".csv"),"w",newline="") as f:
        wr=csv.DictWriter(f, fieldnames=["metric","value","n"]); wr.writeheader()
        for r in rs: wr.writerow(r)
    (OUT/(stem+".md")).write_text("# %s.csv / %s.png\n\n%s\n\nSource: curve.csv (slice). Generated-By: src/analyze.py + src/meta.py. Git-Commit: %s\n"%(stem,stem,desc,GIT))

w("figure_main", lambda m: m.startswith("slip_vs_cant_auc_L") or m in ("slip_vs_cant_null_mean","best_layer"), "Slip-vs-cannot pre-answer probe AUC across layers, with shuffle-null and best layer.")
w("figure_correct_vs_wrong", lambda m: m.startswith("correct_vs_wrong_auc_L"), "Correct-vs-wrong pre-answer probe AUC across layers (sanity).")
w("figure_counts", lambda m: m in ("n_correct","n_wrong","n_slip","n_cannot","greedy_accuracy"), "GSM8K outcome label counts (correct / slip / cannot).")
w("figure_null", lambda m: m in ("best_layer","slip_vs_cant_null_mean") or m.startswith("slip_vs_cant_auc_L"), "Best-layer slip-vs-cannot AUC versus shuffle-null distribution.")

(OUT/"sources.json").write_text(json.dumps({"metrics":{"*":{"csv":"curve.csv"}},"per_example":["eval_points.jsonl"]}, indent=2))
print("wrote sidecars + per-figure csv/md + sources.json")
