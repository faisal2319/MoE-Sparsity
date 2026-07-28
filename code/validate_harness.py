"""Port of their TaskLoss (taskloss-eval) to transformers, run alongside ours."""
import json, math, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "llm-jp/optimal-sparsity-math-d512-E8-k2-320M-A170M"
TASK  = "/tmp/os/taskloss-eval/datasets/lm-evaluation-harness/gsm8k_4shot.json"
N     = int(sys.argv[1]) if len(sys.argv) > 1 else 200

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             device_map="cuda").eval()
pairs = [(d["prompt"], d["answer"]) for d in json.load(open(TASK))][:N]
maxlen = model.config.max_position_embeddings
print(f"{len(pairs)} examples, model max_len={maxlen}")

bits_tok, bits_byte, tot_nll, tot_tok, skipped = [], [], 0.0, 0, 0
for prompt, answer in pairs:
    q = tok.encode(prompt, add_special_tokens=True)
    full = tok.encode(prompt + answer, add_special_tokens=True)
    q_len, t_len = len(q), len(full) - len(q)
    if t_len <= 0 or len(full) > maxlen:
        skipped += 1; continue
    ids = torch.tensor([full], device="cuda")
    with torch.no_grad():
        lp = F.log_softmax(model(ids).logits[:, :-1].float(), -1)
    tl = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    nll = -tl[0, q_len - 1:].sum().item()          # answer tokens only
    bits_tok.append(nll / math.log(2) / t_len)     # their normalization
    bits_byte.append(nll / math.log(2) / len(answer.encode()))
    tot_nll += nll; tot_tok += t_len

print(f"skipped (too long): {skipped}\n")
print("THEIR METHOD  (per-example, then mean)")
print(f"  bits/token : {np.mean(bits_tok):.4f}")
print(f"  bits/byte  : {np.mean(bits_byte):.4f}")
print(f"  nats/token : {np.mean(bits_tok)*math.log(2):.4f}")
print("\nOUR METHOD  (all tokens pooled)")
print(f"  nats/token : {tot_nll/tot_tok:.4f}")
print(f"  bits/token : {tot_nll/tot_tok/math.log(2):.4f}")
print(f"\nmacro vs micro ratio: {np.mean(bits_tok)/(tot_nll/tot_tok/math.log(2)):.4f}")
