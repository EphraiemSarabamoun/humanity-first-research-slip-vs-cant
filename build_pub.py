import json
from pathlib import Path
R = Path("results/real")
gate = json.load(open(R/"gate_report.json")); panel = json.load(open(R/"reviewer_panel.json")); cost = json.load(open(R/"cost_report.json"))
p = {
 "plain": "When a model gets a math problem wrong, it can be because it genuinely cannot do it, or because it can do it but slipped on this attempt. We asked whether the model's internal state, just before it gives an answer, tells these two cases apart. We labelled wrong answers by sampling the model several times (slip = it got it right at least once, cannot = never), then trained a simple detector on the internal state. The clear finding is that the pre-answer state strongly signals whether the answer will be right at all. The finer slip-versus-cannot distinction is only weakly present: it looks significant at first, but once we correct for having picked the best of several internal layers, it is borderline (p about 0.04) and likely mixed up with how hard the problem is. Approve publishes it as an honest, carefully-caveated result; Reject shelves it.",
 "cost": {"api_equivalent_usd": cost.get("api_equivalent_usd"), "sessions": cost.get("sessions")},
 "paper_title": "Pre-Answer States Encode Correctness but Only Marginally Separate 'Cannot Solve' from 'Slipped': A Selection-Corrected Probe on GSM8K",
 "headline_claim": "On Qwen2.5-3B-Instruct on GSM8K, the pre-answer hidden state robustly encodes correctness (correct-vs-wrong probe AUC 0.76, n=800). Within wrong answers, a linear probe separates self-consistency-labeled 'slipped' from 'cannot solve' at AUC 0.6547 (best of 5 layers, n=136), but under a selection-matched permutation null (best layer re-selected per shuffle) this gives only p=0.040 and sits below the null's 97.5th percentile (0.659), so the failure-mode separation is marginal and selection-sensitive, not an established effect, and is likely entangled with problem difficulty.",
 "weakest_part": "The slip-vs-cannot separation, the paper's titular question, is marginal and I now say so. The reviewer panel correctly flagged that my first-draft significance ('above the shuffle null') used a null computed at the single best-of-five-layers position, not selection-matched; when I rebuild the null to re-select the best layer on each shuffle, the observed 0.6547 gives p=0.040 and reaches only ~the 96th percentile, below the selection-matched 97.5th percentile (0.659), so the effect does not clear the corrected upper bound. Beyond selection: (1) the 'cannot' class is small (37 of 136 wrong items), giving a wide CI; (2) most important conceptually, the self-consistency label conflates failure mode with problem difficulty and model confidence, so the probe may read a graded difficulty/confidence axis rather than a categorical slip-vs-cannot representation, and I did not run the difficulty-matched control that would separate them; (3) the pre-answer position is heuristic (last token before '####'); (4) one model, one task. The robust, non-marginal part is the correct-vs-wrong signal (AUC 0.76). I reframed title, abstract, and claims from 'modest significant positive' to 'marginal, selection-sensitive' in response to the panel.",
 "gate_report": gate, "reviewer_panel": panel,
 "not_checked": [
   "a difficulty-matched control to separate a slip-vs-cannot representation from a generic difficulty/confidence axis (the key missing experiment)",
   "a pre-registered single probe layer (to avoid layer-selection inflation entirely)",
   "more 'cannot' items (larger N or a harder task) to narrow the wide interval",
   "more samples per problem (K>6) to denoise the cannot label",
   "larger models and tasks beyond GSM8K; a causal intervention on the putative direction"]
}
open(R/"pub.json", "w").write(json.dumps(p, indent=1))
print("pub.json bytes", len(json.dumps(p)), "cost", p["cost"]["api_equivalent_usd"])
