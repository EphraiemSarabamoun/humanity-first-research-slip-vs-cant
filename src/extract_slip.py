"""extract_slip.py — pre-answer hidden states + self-consistency failure labels on GSM8K.

For each GSM8K problem we (1) greedy-generate a CoT ending in '#### <int>', capture the
residual-stream hidden state at the last token BEFORE the '####' marker (the pre-answer
state) at several layers, and parse the greedy answer; (2) draw K sampled generations
to estimate whether the model CAN solve the problem. A wrong greedy answer is labeled
'slip' if any sample is correct (capability present, this run slipped) and 'cannot' if
no sample is correct (consistent incapacity). The analysis probes the pre-answer states
to separate slip from cannot. Python 3.10. Qwen2.5-7B-Instruct.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-3B-Instruct"  # faster + more error cases -> better power for the small slip/cannot set
SYS = ("Solve the math problem. Show brief step-by-step reasoning, then on the final "
       "line write exactly '#### ' followed by the integer answer and nothing else.")
ANS_RE = re.compile(r"####\s*(-?[0-9][0-9,]*)")


def gold_of(ans_field):
    m = re.search(r"####\s*(-?[0-9][0-9,]*)", ans_field)
    return int(m.group(1).replace(",", "")) if m else None


def parse_pred(text):
    ms = ANS_RE.findall(text)
    if ms:
        return int(ms[-1].replace(",", ""))
    # fallback: last integer in the text
    nums = re.findall(r"-?[0-9][0-9,]*", text)
    return int(nums[-1].replace(",", "")) if nums else None


def pick_layers(nl):
    return sorted(set(max(1, min(nl, int(round(f*nl)))) for f in (0.4, 0.55, 0.7, 0.85, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/real")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(ds))[:args.n]
    items = [{"q": ds[int(i)]["question"], "gold": gold_of(ds[int(i)]["answer"])} for i in idx]
    items = [it for it in items if it["gold"] is not None]
    print("[gsm8k] %d items" % len(items), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    nl = model.config.num_hidden_layers
    layers = pick_layers(nl)
    print("[layers]", layers, "of", nl, flush=True)

    prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": it["q"]}],
                                       tokenize=False, add_generation_prompt=True) for it in items]

    @torch.no_grad()
    def gen(ps, do_sample, max_new=400, batch=16, temp=0.8):
        outs = []
        for i in range(0, len(ps), batch):
            enc = tok(ps[i:i+batch], return_tensors="pt", padding=True, truncation=True, max_length=900).to(model.device)
            kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            if do_sample:
                kw.update(do_sample=True, temperature=temp, top_p=0.95)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            g = model.generate(**enc, **kw)
            outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    # 1) greedy
    print("[greedy] generating", flush=True)
    greedy = gen(prompts, do_sample=False)
    pred_g = [parse_pred(r) for r in greedy]
    gcorrect = np.array([1 if (pred_g[i] is not None and pred_g[i] == items[i]["gold"]) else 0 for i in range(len(items))])
    print("[greedy] acc=%.3f parseable=%d/%d" % (gcorrect.mean(), sum(p is not None for p in pred_g), len(items)), flush=True)

    # 2) K samples for capability label
    samp_correct = np.zeros(len(items), dtype=int)
    for s in range(args.k):
        seeded = gen(prompts, do_sample=True)
        for i, r in enumerate(seeded):
            p = parse_pred(r)
            if p is not None and p == items[i]["gold"]:
                samp_correct[i] += 1
        print("[sample %d/%d] cumulative any-correct=%d" % (s+1, args.k, int((samp_correct > 0).sum())), flush=True)

    # 3) pre-answer hidden states: re-encode prompt + reasoning-before-#### , last-token state
    @torch.no_grad()
    def pre_answer_states(batch=8):
        acts = {li: [] for li in layers}
        kept = []
        for start in range(0, len(items), batch):
            texts, sidx = [], []
            for i in range(start, min(start+batch, len(items))):
                r = greedy[i]
                m = re.search(r"####", r)
                pre = r[:m.start()] if m else r  # reasoning before the answer marker
                texts.append(prompts[i] + pre); sidx.append(i)
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1100).to(model.device)
            out = model(**enc, output_hidden_states=True)
            for li in layers:
                h = out.hidden_states[li][:, -1, :].float().cpu().numpy()  # left pad -> last token
                acts[li].append(h)
            kept.extend(sidx)
        return {li: np.concatenate(acts[li], 0) for li in layers}, kept

    print("[states] extracting pre-answer hidden states", flush=True)
    acts, kept = pre_answer_states()
    np.savez_compressed(out / "slip_data.npz",
                        gold=np.array([it["gold"] for it in items]),
                        greedy_correct=gcorrect, samp_correct=samp_correct, k=args.k,
                        layers=np.array(layers),
                        eval_order=np.arange(len(items)),
                        **{("h_L%02d" % li): acts[li] for li in layers})
    # quick label summary
    wrong = gcorrect == 0
    slip = wrong & (samp_correct > 0)
    cant = wrong & (samp_correct == 0)
    print("[labels] correct=%d wrong=%d  slip=%d cannot=%d" % (
        int(gcorrect.sum()), int(wrong.sum()), int(slip.sum()), int(cant.sum())), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
