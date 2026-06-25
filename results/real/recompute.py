"""recompute.py — reproduce analysis_summary.txt from per-example data alone (stdlib).
Reads eval_points.jsonl ({section, eval_order, score, label}) and null_dist.csv
({key, auc}); reports AUC + bootstrap CI per probe section and the shuffle-null
mean/interval. Gate: cd results/real && python3 recompute.py | diff - analysis_summary.txt."""
import json, csv, random
from collections import defaultdict
BOOT_N, BOOT_SEED = 2000, 42
SECTIONS = [
    ("Pre-answer probe: slip vs cannot-solve (5-fold OOF AUC, best layer)", "auc", "slip_vs_cant"),
    ("Pre-answer probe: shuffle-label null for slip vs cannot-solve", "null", "slip_vs_cant_null"),
    ("Pre-answer probe: correct vs wrong (5-fold OOF AUC, best layer; sanity)", "auc", "correct_vs_wrong"),
]
def auc_rank(scores, labels):
    pairs = sorted(zip(scores, labels)); n = len(pairs); ranks = [0.0]*n; i = 0
    while i < n:
        j = i
        while j+1 < n and pairs[j+1][0] == pairs[i][0]: j += 1
        avg = (i+j)/2.0 + 1.0
        for k in range(i, j+1): ranks[k] = avg
        i = j+1
    npos = sum(1 for _, l in pairs if l == 1); nneg = n - npos
    if npos == 0 or nneg == 0: return float("nan")
    sp = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (sp - npos*(npos+1)/2.0)/(npos*nneg)
def pct(s, q):
    if not s: return float("nan")
    p = q/100.0*(len(s)-1); lo = int(p); f = p-lo
    return s[lo]*(1-f)+s[lo+1]*f if lo+1 < len(s) else s[lo]
def auc_ci(scores, labels):
    point = auc_rank(scores, labels); rng = random.Random(BOOT_SEED); n = len(scores); boots = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(n) for _ in range(n)]
        sl = [labels[i] for i in idx]
        if sum(sl) in (0, len(sl)): continue
        boots.append(auc_rank([scores[i] for i in idx], sl))
    boots.sort(); return point, pct(boots, 2.5), pct(boots, 97.5), n
def main():
    ep = defaultdict(list)
    for line in open("eval_points.jsonl"):
        line = line.strip()
        if line:
            r = json.loads(line); ep[r["section"]].append(r)
    for k in ep: ep[k].sort(key=lambda r: r["eval_order"])
    nulls = defaultdict(list)
    try:
        for r in csv.DictReader(open("null_dist.csv")): nulls[r["key"]].append(float(r["auc"]))
    except FileNotFoundError:
        pass
    L = ["# Linear separability of failure modes in pre-answer hidden states",
         "", "Model: Qwen2.5-3B-Instruct on GSM8K. Pre-answer hidden state = last token before the '####' answer marker.",
         "Labels (greedy-wrong only): slip = >=1 of K=6 samples correct (capability present); cannot = 0/K correct.",
         "Probe: logistic regression on PCA-256 of the residual, 5-fold OOF, best layer by AUC. Bootstrap 2000, seed 42. Chance AUC = 0.5.", ""]
    for title, kind, key in SECTIONS:
        if kind == "auc":
            rows = ep[key]; p, lo, hi, n = auc_ci([r["score"] for r in rows], [r["label"] for r in rows])
            L.append("## %s" % title); L.append("  AUC = %.4f  (95%% CI %.4f-%.4f, n=%d)" % (p, lo, hi, n)); L.append("")
        else:
            v = sorted(nulls.get(key, []))
            if v:
                m = sum(v)/len(v)
                L.append("## %s" % title); L.append("  AUC mean = %.4f  (95%% null interval %.4f-%.4f, n=%d)" % (m, pct(v,2.5), pct(v,97.5), len(v))); L.append("")
            else:
                L.append("## %s" % title); L.append("  AUC mean = nan  (95%% null interval nan-nan, n=0)"); L.append("")
    print("\n".join(L).rstrip("\n"))
if __name__ == "__main__": main()
