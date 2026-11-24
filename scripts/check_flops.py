"""
Check that the analytical FLOP estimates of the model (used for ALL compute accounting)
match the FLOPs actually executed, measured with torch.utils.flop_counter.FlopCounterMode.

Run this whenever you modify the architecture in nanochat/gpt.py:
python -m scripts.check_flops --depth 4 --max-seq-len 256

It checks three things on a small model on CPU:
  1. training:  forward+backward over T tokens   vs  estimate_flops() * T
  2. prefill:   forward over T tokens (no grad)  vs  estimate_prefill_flops(T)
  3. decode:    one token with a KV cache        vs  estimate_decode_flops(context)
Attention is computed with the plain (math) SDPA kernel so that its matmuls are visible to the
counter. With sliding windows ('S' in --window-pattern) the estimate assumes that masked blocks are
skipped (as Flash Attention does), so the measured number will be larger: check with 'L'.
Graders run this check on your final architecture; estimates must be within 5%.
"""

import argparse

import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.nn.attention import sdpa_kernel, SDPBackend

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache
from nanochat.common import COMPUTE_DTYPE

parser = argparse.ArgumentParser(description="Verify the analytical FLOP estimates of the model")
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=256)
parser.add_argument("--window-pattern", type=str, default="L")
parser.add_argument("--vocab-size", type=int, default=32768)
parser.add_argument("--tolerance", type=float, default=0.05)
args = parser.parse_args()

base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = GPTConfig(sequence_len=args.max_seq_len, vocab_size=args.vocab_size, n_layer=args.depth,
                   n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim, window_pattern=args.window_pattern)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device="cpu")
model.init_weights()
T = args.max_seq_len
idx = torch.randint(0, args.vocab_size, (1, T))

def measure(fn):
    with sdpa_kernel(SDPBackend.MATH):
        with FlopCounterMode(display=False) as counter:
            fn()
    return counter.get_total_flops()

# 1) training
model.train()
def train_step():
    loss = model(idx, idx)
    loss.backward()
measured_train = measure(train_step)
model.zero_grad(set_to_none=True)
estimated_train = model.estimate_flops() * T

# 2) prefill
model.eval()
with torch.no_grad():
    measured_prefill = measure(lambda: model(idx))
estimated_prefill = model.estimate_prefill_flops(T)

# 3) decode one token at context length T//2 + 1
context = T // 2
kv = KVCache(batch_size=1, num_heads=config.n_kv_head, seq_len=T, head_dim=model_dim // num_heads,
             num_layers=config.n_layer, device="cpu", dtype=COMPUTE_DTYPE)
with torch.no_grad():
    model(idx[:, :context], kv_cache=kv)
    measured_decode = measure(lambda: model(idx[:, context:context + 1], kv_cache=kv))
estimated_decode = model.estimate_decode_flops(context + 1)

ok = True
print(f"{'':10s} {'measured':>12s} {'estimated':>12s} {'rel. error':>10s}")
for name, m, e in [("train", measured_train, estimated_train), ("prefill", measured_prefill, estimated_prefill), ("decode", measured_decode, estimated_decode)]:
    err = (e - m) / m
    ok &= abs(err) <= args.tolerance
    print(f"{name:10s} {m:12.4e} {e:12.4e} {100 * err:9.2f}%")
print("OK: estimates match the measured FLOPs" if ok else f"MISMATCH: fix the estimate_* methods in nanochat/gpt.py (tolerance {100 * args.tolerance:.0f}%)")
raise SystemExit(0 if ok else 1)
