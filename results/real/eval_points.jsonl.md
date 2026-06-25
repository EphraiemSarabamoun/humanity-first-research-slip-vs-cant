# eval_points.jsonl — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py (data from src/extract_slip.py)
Command: python3 src/analyze.py --out_dir results/real
Git-Commit: bea9a6f3c1aea99791fe9fe07c567efdd5d8551c
Seeds: 42 (GSM8K sampling in extract_slip.py; greedy decoding deterministic; K=6 sampled generations at temperature 0.8 top-p 0.95; 5-fold StratifiedKFold probe; per-fold PCA(<=50); 100 shuffle-null draws; 2000-resample percentile bootstrap)
Source-Data: GSM8K (openai/gsm8k, main, test), 800 problems via datasets; Qwen2.5-3B-Instruct, RTX 5090, 2026-06-24, torch 2.12 cu130; pre-answer hidden states (last token before the '####' answer marker) at layers 14/20/25/31/36 in results/real/slip_data.npz
Analysis-Command: cd results/real && python3 recompute.py  (each section AUC = rank-based AUC with 95% bootstrap CI)
Columns:
  section (slip_vs_cant or correct_vs_wrong, both at the best layer); eval_order (position); score (5-fold OOF logistic-regression decision-function value); label (slip_vs_cant: 1=slip 0=cannot; correct_vs_wrong: 1=greedy-correct 0=wrong)
