"""
Plan the compute of an experiment design BEFORE running it:

    C_total = C_pretrain + C_SFT + C_RL    <=    B = 3e18 FLOPs

Inference is fixed (the official configuration) and not charged, and C_data (model-based data selection) is declared
but not part of B: the report shows it separately.

The script uses the same FLOP formulas as the training scripts (the estimate_* methods of the model in
nanochat/gpt.py, and nanochat/compute.py), on a model built on the "meta" device: no GPU, no memory, no
checkpoint needed. You supply the design: model size, token counts, steps.

Examples (run from starter_kit/):
    python -m scripts.plan_budget                                    # the baseline: d12, 12 tokens/param, SFT
    python -m scripts.plan_budget --depth 16 --tokens-per-param 8
    python -m scripts.plan_budget --depth 14 --fill                  # pretrain on everything that is left
    python -m scripts.plan_budget --sweep                            # model size vs. affordable pretraining tokens
    python -m scripts.plan_budget --plan configs/plans/example.json  # several stages or data sources
    python -m scripts.plan_budget --depth 16 --print-plan > my_plan.json   # start a plan file from the flags

Every report ends with the flags of base_train / chat_sft / chat_rl (and the data_flops values) that give each stage
its planned compute: --num-iterations and --batch-size (sequences). Copy them into your experiment runner.

Training FLOPs are estimate_flops() * tokens (~ 6*N per token plus attention, N = matmul params). SFT counts
every row of every batch (padding included): steps * batch_size * sequence length. RL (a fixed pipeline, nanochat/rl.py) is planned
by its math and code token budgets: its rollouts and policy-gradient passes cost about rl.COST_PER_TOKEN x
estimate_flops() per RL token (measured; the ledger records the exact value). C_data: 2 * params * tokens scored by
the model that selects your data (e.g. a quality classifier; creating data is not allowed).
"""

import math
import json
import copy
import argparse

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat import rl
from nanochat.compute import TOTAL_BUDGET_FLOPS, BUDGET_TOLERANCE, train_flops

A100_FLOPS_PER_SEC = 1.72e14  # training throughput of the baseline (d12, bf16, SDPA) on one A100-80GB; ~55% MFU
B_REF = 2**19 # tokens per step of the d12 reference (base_train's default batch: 512 sequences of 1,024 tokens)
SFT_BATCH_SIZE = 512 # sequences per SFT step (the baseline's)
MODEL_DEFAULTS = {"depth": 12, "aspect_ratio": 64, "head_dim": 128, "max_seq_len": 1024, "window_pattern": "L", "vocab_size": 32768}

# -----------------------------------------------------------------------------
# Models and token counts

def build_model(spec):
    """A GPT on the meta device, from base_train-style arguments (depth, aspect_ratio, ...)."""
    s = {**MODEL_DEFAULTS, **spec}
    assert s["vocab_size"] == MODEL_DEFAULTS["vocab_size"], "The tokenizer is fixed: 32,768 tokens (README §5.2)"
    model_dim = -(-s["depth"] * s["aspect_ratio"] // s["head_dim"]) * s["head_dim"] # rounded up as in base_train
    num_heads = model_dim // s["head_dim"]
    config = GPTConfig(sequence_len=s["max_seq_len"], vocab_size=s["vocab_size"], n_layer=s["depth"],
                       n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim, window_pattern=s["window_pattern"])
    with torch.device("meta"):
        return GPT(config)


def scaling_params(model):
    """Transformer matrices + lm_head: the parameter count used for tokens/param ratios (as in base_train)."""
    counts = model.num_scaling_params()
    return counts["transformer_matrices"] + counts["lm_head"]


def default_batch_size(model, spec):
    """base_train's default --batch-size (sequences per step) for this model: ~B_REF x (N / N_d12)^0.383 tokens."""
    ratio = scaling_params(model) / scaling_params(build_model({**spec, "depth": 12}))
    return max(1, 2 ** round(math.log2(B_REF * ratio ** 0.383)) // model.config.sequence_len)


def describe(model):
    c = model.config
    return (f"d{c.n_layer}: {c.n_layer} layers, width {c.n_embd}, {c.n_head} heads, context {c.sequence_len} | "
            f"{scaling_params(model) / 1e6:.1f}M scaling params, {model.num_scaling_params()['total'] / 1e6:.1f}M total | "
            f"{model.estimate_flops():.3g} FLOPs per training token")


# -----------------------------------------------------------------------------
# Costing a plan

def stage_cost(stage, model):
    """(description, FLOPs) of one training stage."""
    kind = stage.get("stage", "pretraining")
    if kind == "rl":
        math_tokens, code_tokens = stage.get("math_tokens", 0), stage.get("code_tokens", 0)
        what = f"{math_tokens:.3g} math + {code_tokens:.3g} code RL tokens"
        return what, (math_tokens + code_tokens) * rl.COST_PER_TOKEN * model.estimate_flops()
    if "steps" in stage:
        batch, seq_len = stage.get("batch_size", SFT_BATCH_SIZE), model.config.sequence_len
        tokens = stage["steps"] * batch * seq_len
        what = f"{stage['steps']} steps x {batch} x {seq_len} tokens"
    elif "tokens_per_param" in stage:
        tokens = stage["tokens_per_param"] * scaling_params(model)
        what = f"{tokens:.3g} tokens ({stage['tokens_per_param']:g} tokens/param)"
    else:
        tokens = stage.get("tokens", 0)
        what = f"{tokens:.3g} tokens ({tokens / scaling_params(model):.1f} tokens/param)"
    return what, train_flops(model, tokens)


def stage_term(stage):
    """The budget term of a training stage: C_pretrain (base_train), C_SFT (chat_sft) or C_RL (chat_rl)."""
    return {"sft": "C_SFT", "rl": "C_RL"}.get(stage.get("stage", "pretraining"), "C_pretrain")


def cost(plan, budget=TOTAL_BUDGET_FLOPS):
    models = {name: build_model(spec) for name, spec in plan["models"].items()}
    rows = [] # (part, label, description, flops)
    stage_flops = {} # index in plan["train"] -> FLOPs
    fill = None
    for i, stage in enumerate(plan.get("train", [])):
        if stage.get("fill"):
            fill = i
            continue
        what, flops = stage_cost(stage, models[stage["model"]])
        stage_flops[i] = flops
        rows.append((stage_term(stage), f"{stage['model']} {stage.get('stage', 'pretraining')}", what, flops))
    data_rows = [] # C_data: declared, not part of B
    for source in plan.get("data", []):
        flops = source["flops"] if "flops" in source else 2 * source["params"] * source["tokens"]
        what = "" if "flops" in source else f"2 x {source['params']:.3g} params x {source['tokens']:.3g} tokens"
        data_rows.append(("C_data", source.get("source", "data"), what, flops))
    if fill is not None: # pretraining that uses whatever is left
        stage = plan["train"][fill]
        model = models[stage["model"]]
        left = max(0.0, budget - sum(r[3] for r in rows))
        stage_flops[fill] = left
        tokens = left / model.estimate_flops()
        what = f"{tokens:.3g} tokens ({tokens / scaling_params(model):.1f} tokens/param)"
        rows.insert(0, (stage_term(stage), f"{stage['model']} {stage.get('stage', 'pretraining')} (fill)", what, left))
    total = sum(r[3] for r in rows)
    return {"models": models, "rows": rows, "data_rows": data_rows, "total": total, "budget": budget,
            "stage_flops": [(stage, stage_flops[i]) for i, stage in enumerate(plan.get("train", []))],
            "within": total <= budget * (1 + BUDGET_TOLERANCE)}


def training_flags(plan, report):
    """(label, command, note) for every stage: the flags that give each stage exactly its planned compute."""
    lines, started = [], set()
    for stage, flops in report["stage_flops"]:
        name, kind = stage["model"], stage.get("stage", "pretraining")
        model = report["models"][name]
        per_token = model.estimate_flops()
        if kind == "sft":
            batch = stage.get("batch_size", SFT_BATCH_SIZE)
            steps = round(flops / (per_token * batch * model.config.sequence_len))
            lines.append((f"{name} sft", f"python -m scripts.chat_sft --num-iterations {steps} --batch-size {batch}",
                          f"{batch} sequences of {model.config.sequence_len} tokens per step, padding included; "
                          f"SFT stops earlier if the mixture runs out"))
        elif kind == "rl":
            lines.append((f"{name} rl", f"python -m scripts.chat_rl --model-tag <tag of the SFT model> --math-tokens "
                          f"{stage.get('math_tokens', 0):.3g} --code-tokens {stage.get('code_tokens', 0):.3g}",
                          f"the fixed RL pipeline; its exact cost is in the ledger (~{rl.COST_PER_TOKEN:g} x training FLOPs per token)"))
        else:
            spec = {**MODEL_DEFAULTS, **plan["models"][name]}
            batch = default_batch_size(model, spec)
            steps = max(1, int(flops / per_token // (batch * model.config.sequence_len)))
            note = (f"{steps} steps x {batch} x {model.config.sequence_len} tokens = {steps * batch * model.config.sequence_len:.3g} "
                    f"tokens ({steps * batch * model.config.sequence_len / scaling_params(model):.1f} tokens/param)")
            if name not in started: # the first pretraining stage creates the model
                arch = [f"--depth {spec['depth']}"] + [f"--{k.replace('_', '-')} {spec[k]}" for k in
                                                       ("aspect_ratio", "head_dim", "max_seq_len", "window_pattern") if spec[k] != MODEL_DEFAULTS[k]]
                lines.append((f"{name} {kind}", f"python -m scripts.base_train {' '.join(arch)} --num-iterations {steps} --batch-size {batch}", note))
            else: # later pretraining stages continue from the previous one
                lines.append((f"{name} {kind}", f"python -m scripts.base_train --init-from <tag of the previous stage> "
                              f"--num-iterations {steps} --batch-size {batch}", note))
        if kind not in ("sft", "rl"):
            started.add(name)
    for source in plan.get("data", []):
        flops = source["flops"] if "flops" in source else 2 * source["params"] * source["tokens"]
        lines.append((f"data", f'"data_flops": {flops:.3g}', f"declared in the data config entry of: {source.get('source', 'data')} (not in B)"))
    return lines


def print_report(plan, report):
    b = report["budget"]
    print("Models")
    for name, model in report["models"].items():
        print(f"  {name:10s} {describe(model)}")
    print(f"\nBudget B = {b:.2e} FLOPs")
    for part, label, what, flops in report["rows"]:
        label = label if len(label) <= 32 else label[:29] + "..."
        print(f"  {part:10s} {label:32s} {what:52s} {flops:10.3e} {100 * flops / b:7.2f}%")
    status = "OK, within budget" if report["within"] else f"OVER BUDGET (limit {b * (1 + BUDGET_TOLERANCE):.3e} incl. {100 * BUDGET_TOLERANCE:.0f}% tolerance)"
    print(f"  {'C_total':10s} {'':32s} {'':52s} {report['total']:10.3e} {100 * report['total'] / b:7.2f}%   {status}")
    for part, label, what, flops in report["data_rows"]:
        label = label if len(label) <= 32 else label[:29] + "..."
        print(f"  {part:10s} {label:32s} {what:52s} {flops:10.3e}   declared, not in B")
    pre = [s for s in plan.get("train", []) if s.get("stage", "pretraining") == "pretraining" and not s.get("fill")]
    if report["total"] < b and pre:
        model = report["models"][pre[0]["model"]]
        print(f"  Left: {b - report['total']:.3e} FLOPs = {(b - report['total']) / model.estimate_flops():.3g} more "
              f"pretraining tokens for {pre[0]['model']} (or use --fill)")
    train = report["total"]
    print(f"  Training time: ~{train / A100_FLOPS_PER_SEC / 3600:.1f} h on one A100-80GB "
          f"(at the baseline's {A100_FLOPS_PER_SEC:.2e} FLOP/s; smaller models reach a lower utilization)")

    print("\nFlags for your experiment runner (they fix the compute of each stage; add --model-tag, --mixture, ... yourself):")
    for label, command, note in training_flags(plan, report):
        print(f"  {label:20s} {command}\n  {'':20s}   # {note}")

# -----------------------------------------------------------------------------

def plan_from_args(args):
    model = {k: getattr(args, k, MODEL_DEFAULTS[k]) for k in MODEL_DEFAULTS} # vocab_size: fixed (no flag)
    pretraining = {"model": "policy", "stage": "pretraining"}
    if args.fill:
        pretraining["fill"] = True
    elif args.pretrain_tokens is not None:
        pretraining["tokens"] = args.pretrain_tokens
    else:
        pretraining["tokens_per_param"] = args.tokens_per_param
    train = [pretraining]
    if args.sft_steps > 0:
        train.append({"model": "policy", "stage": "sft", "steps": args.sft_steps, "batch_size": args.sft_batch_size})
    if args.rl_math_tokens > 0 or args.rl_code_tokens > 0:
        train.append({"model": "policy", "stage": "rl", "math_tokens": args.rl_math_tokens, "code_tokens": args.rl_code_tokens})
    data = []
    if args.data_flops > 0:
        data.append({"source": "declared", "flops": args.data_flops})
    if args.data_params > 0:
        data.append({"source": "model-based selection", "params": args.data_params, "tokens": args.data_tokens})
    return {"models": {"policy": model}, "train": train, "data": data}


def main():
    parser = argparse.ArgumentParser(description="Plan C_total = C_pretrain + C_SFT + C_RL of an experiment design")
    parser.add_argument("--plan", type=str, default=None, help="a plan file (json): several models, stages, data sources (see configs/plans/)")
    parser.add_argument("--print-plan", action="store_true", help="print the plan built from the flags (json) and exit")
    parser.add_argument("--sweep", type=str, nargs="?", const="4,6,8,10,12,14,16,18,20", default=None,
                        help="depths to compare: each one pretrains on everything left after SFT/RL/data")
    # model (same arguments as base_train)
    parser.add_argument("--depth", type=int, default=MODEL_DEFAULTS["depth"])
    parser.add_argument("--aspect-ratio", type=int, default=MODEL_DEFAULTS["aspect_ratio"])
    parser.add_argument("--head-dim", type=int, default=MODEL_DEFAULTS["head_dim"])
    parser.add_argument("--max-seq-len", type=int, default=MODEL_DEFAULTS["max_seq_len"])
    parser.add_argument("--window-pattern", type=str, default=MODEL_DEFAULTS["window_pattern"])
    # training
    parser.add_argument("--tokens-per-param", type=float, default=12, help="pretraining tokens per scaling param (turned into base_train's --num-iterations)")
    parser.add_argument("--pretrain-tokens", type=float, default=None, help="pretraining tokens (overrides --tokens-per-param)")
    parser.add_argument("--fill", action="store_true", help="pretrain on all the budget that is left")
    parser.add_argument("--sft-steps", type=int, default=85, help="SFT optimizer steps (the baseline does 85)")
    parser.add_argument("--sft-batch-size", type=int, default=SFT_BATCH_SIZE, help="SFT sequences per step (chat_sft's --batch-size)")
    parser.add_argument("--rl-math-tokens", type=float, default=0, help="RL tokens on math problems (chat_rl --math-tokens)")
    parser.add_argument("--rl-code-tokens", type=float, default=0, help="RL tokens on code problems (chat_rl --code-tokens)")
    # data
    parser.add_argument("--data-flops", type=float, default=0.0, help="C_data you already know (declared, not in B)")
    parser.add_argument("--data-params", type=float, default=0.0, help="active params of the model that scores/selects your data")
    parser.add_argument("--data-tokens", type=float, default=0.0, help="tokens that model scores")
    args = parser.parse_args()

    if args.plan:
        with open(args.plan, "r", encoding="utf-8") as f:
            plan = json.load(f)
    else:
        plan = plan_from_args(args)
    if args.print_plan:
        print(json.dumps(plan, indent=2))
        return

    if args.sweep is None:
        print_report(plan, cost(plan))
        return
    # sweep: the policy's pretraining takes whatever is left; everything else stays as planned
    print(f"{'depth':>5s} {'width':>5s} {'params':>8s} {'FLOPs/tok':>9s} {'pretrain tokens':>15s} {'tokens/param':>12s}")
    for depth in [int(d) for d in args.sweep.split(",")]:
        p = copy.deepcopy(plan)
        p["models"]["policy"]["depth"] = depth
        for stage in p["train"]:
            if stage["model"] == "policy" and stage.get("stage", "pretraining") == "pretraining":
                stage.clear()
                stage.update({"model": "policy", "stage": "pretraining", "fill": True})
        r = cost(p)
        model = r["models"]["policy"]
        tokens = r["rows"][0][3] / model.estimate_flops()
        print(f"{depth:5d} {model.config.n_embd:5d} {scaling_params(model) / 1e6:7.1f}M {model.estimate_flops():9.2e} "
              f"{tokens:15.3g} {tokens / scaling_params(model):12.1f}")
    print(f"Each row spends the whole budget: ~{r['total'] / A100_FLOPS_PER_SEC / 3600:.1f} h of training on one A100 at the "
          f"baseline's utilization (small models run at a lower one). nanochat's tuned ratio is ~12 tokens/param.")
    print("Pick a depth, then run with --depth <D> --fill to get the training flags of that design.")


if __name__ == "__main__":
    main()
