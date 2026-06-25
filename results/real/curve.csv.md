# curve.csv — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py (data from src/extract_slip.py)
Command: python3 src/analyze.py --out_dir results/real
Git-Commit: bea9a6f3c1aea99791fe9fe07c567efdd5d8551c
Seeds: 42 (GSM8K sampling in extract_slip.py; greedy decoding deterministic; K=6 sampled generations at temperature 0.8 top-p 0.95; 5-fold StratifiedKFold probe; per-fold PCA(<=50); 100 shuffle-null draws; 2000-resample percentile bootstrap)
Source-Data: GSM8K (openai/gsm8k, main, test), 800 problems via datasets; Qwen2.5-3B-Instruct, RTX 5090, 2026-06-24, torch 2.12 cu130; pre-answer hidden states (last token before the '####' answer marker) at layers 14/20/25/31/36 in results/real/slip_data.npz
Analysis-Command: cd results/real && python3 recompute.py | diff - analysis_summary.txt  (empty); the headline slip-vs-cant and correct-vs-wrong AUCs are reproduced from eval_points.jsonl, the null from null_dist.csv; per-layer rows here back the figures
Columns:
  metric (slip_vs_cant_auc_L<layer> = 5-fold OOF AUC separating slip from cannot at that layer; correct_vs_wrong_auc_L<layer> = sanity probe AUC; best_layer = layer with max slip-vs-cant AUC; n_correct/n_wrong/n_slip/n_cannot = label counts; greedy_accuracy; slip_vs_cant_null_mean = mean shuffle-label AUC);
  value (AUC 0-1, count, or layer index); n (items entering that metric: 136 for slip-vs-cant, 800 for correct-vs-wrong and counts, 100 for the null)
