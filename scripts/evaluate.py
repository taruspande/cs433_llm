"""
Official evaluation: run the official solver (solvers/default.py) with your model and the official inference
configuration (the same for every team: greedy decoding, one sample of at most 512 tokens) on an evaluation suite.

!!! DO NOT MODIFY THIS FILE. The official grading runs our own copy of it on the hidden test suite. !!!

It reports the score (per subset, per family, final), checks the inference allowance of every
problem (INFERENCE_TOKENS_PER_PROBLEM tokens: a problem over it is scored as wrong), and reports the
budget  C_total = C_pretrain + C_SFT + C_RL  from the ledger of your checkpoint (and, outside the budget,
the C_data declared in your data card).
Inference FLOPs are measured too, for information (they are not charged).

Examples:

# evaluate an SFT checkpoint with greedy decoding on the released validation suite
python -m scripts.evaluate --model sft:d12

# only math, only 200 problems (quick check)
python -m scripts.evaluate --model sft:d12 --families math --max-problems 200

# evaluate a packaged submission exactly like the graders will (but on the validation suite)
python -m scripts.evaluate --submission submission/

# multi-GPU
torchrun --standalone --nproc_per_node=8 -m scripts.evaluate -- --model sft:d12
"""

import os
import glob
import json
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_dist_info, autodetect_device_type
from nanochat.checkpoint_manager import load_model_spec
from nanochat.compute import train_flops_by_term, budget_report, format_budget_report, INFERENCE_TOKENS_PER_PROBLEM
from nanochat.inference import MeteredLM, load_solver_class
from tasks.eval_suite import load_suite, public_view, score_item, aggregate_scores, format_summary

SOLVER = "solvers/default.py" # the official solver
OFFICIAL_CONFIG = {"num_samples": 1, "temperature": 0.0, "top_k": 50, "max_new_tokens": 512, # the same for every team:
                   "math_aggregation": "first", "code_selection": "first"}                   # greedy, one sample

# -----------------------------------------------------------------------------

def run_suite(lms, solver, items, log_every=50, batch_size=256):
    """
    Solve and score `items` (sharded across DDP ranks). Returns the list of per-item records of all ranks.
    lms: dict name -> MeteredLM (the same objects that were given to the solver).
    If the solver has a `solve_batch(items) -> outputs` method, problems are solved batch_size at a time
    (batched generation, much faster); otherwise one at a time with `solve(item)`. The tokens and FLOPs metered for
    each problem are the same in both modes, as long as the solver passes keys (generate_many). On a GPU, results can
    differ slightly between batch sizes (bf16 numerics depend on the batch shape, which occasionally flips a
    near-tie), so compare runs at the same batch size.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    my_items = items[ddp_rank::ddp_world_size]
    batched = batch_size > 1 and hasattr(solver, "solve_batch")
    chunk = batch_size if batched else 1
    records = []
    t0 = time.time()
    for b in range(0, len(my_items), chunk):
        batch = my_items[b:b + chunk]
        for lm in lms.values():
            lm.reset()
        if batched:
            outputs = solver.solve_batch([public_view(item) for item in batch])
            assert len(outputs) == len(batch), "solve_batch must return one output per item"
        else:
            outputs = [solver.solve(public_view(batch[0]))]
        outputs = ["" if o is None else str(o) for o in outputs]
        # keyed (generate_many) tokens and FLOPs go to their problem. Unkeyed tokens count fully for every problem
        # of the batch (we cannot tell which problem they were for); unkeyed FLOPs are shared equally
        unkeyed = sum(lm.flops for lm in lms.values()) / len(batch)
        unkeyed_tokens = sum(lm.tokens for lm in lms.values())
        with ThreadPoolExecutor(max_workers=8) as pool: # code tests run in subprocesses: score in parallel
            scores = list(pool.map(score_item, batch, outputs))
        for item, output, (correct, extracted) in zip(batch, outputs, scores):
            flops = unkeyed + sum(lm.flops_by_key.get(item["id"], 0.0) for lm in lms.values())
            tokens = unkeyed_tokens + sum(lm.tokens_by_key.get(item["id"], 0) for lm in lms.values())
            over = tokens > INFERENCE_TOKENS_PER_PROBLEM
            records.append({
                "id": item["id"], "family": item["family"], "subset": item["subset"],
                "correct": bool(correct) and not over, "tokens": tokens, "over_allowance": over,
                "flops": flops, "extracted": extracted, "output": output,
            })
        if ddp_rank == 0 and (len(records) // log_every) > ((len(records) - len(batch)) // log_every):
            acc = sum(r["correct"] for r in records) / len(records)
            print0(f"  [{len(records) * ddp_world_size}/{len(items)}] running accuracy {100 * acc:.2f}% | {time.time() - t0:.0f}s")
    if ddp:
        gathered = [None] * ddp_world_size
        dist.all_gather_object(gathered, records)
        records = [r for rank_records in gathered for r in rank_records]
    return records


def summarize(records):
    summary = aggregate_scores(records)
    summary["num_problems"] = len(records)
    n = max(len(records), 1)
    summary["inference_tokens_per_problem"] = {"mean": sum(r["tokens"] for r in records) / n,
                                               "max": max((r["tokens"] for r in records), default=0),
                                               "allowance": INFERENCE_TOKENS_PER_PROBLEM,
                                               "over_allowance": sum(r["over_allowance"] for r in records)}
    summary["inference_flops_per_problem"] = sum(r["flops"] for r in records) / n # for information, not charged
    return summary


def format_allowance(summary):
    t = summary["inference_tokens_per_problem"]
    line = (f"{'inference tokens':>22s}: mean {t['mean']:,.0f} / max {t['max']:,} per problem "
            f"(allowance {t['allowance']:,}; inference FLOPs, not charged: {summary['inference_flops_per_problem']:.2e} per problem)")
    if t["over_allowance"]:
        line += f"\n{'':>22s}  {t['over_allowance']} problem(s) over the allowance were scored as wrong"
    return line


def train_flops_of(metas):
    """{"pretrain", "sft", "rl"} -> training FLOPs of the models' ledgers (shared stages count once)."""
    return train_flops_by_term(metas)


def load_submission(submission_dir):
    """A submission is ONE model ("policy") + data card + runner. It is evaluated with OFFICIAL_CONFIG."""
    with open(os.path.join(submission_dir, "submission.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    resolve = lambda p: os.path.join(submission_dir, p)
    assert list(manifest["models"]) == ["policy"], (
        f"A submission contains exactly one model, 'policy' (no verifier or other models): got {list(manifest['models'])}")
    policy = resolve(manifest["models"]["policy"])
    checkpoints = sorted(os.path.basename(p) for p in glob.glob(os.path.join(policy, "model_*.pt")))
    assert len(checkpoints) <= 1, (
        f"A submission contains exactly one model: {policy} holds {len(checkpoints)} checkpoints {checkpoints}. "
        f"Package it again with scripts/make_submission.py")
    data_flops = 0.0
    data_card = resolve(manifest.get("data_card", "data_card.json"))
    if os.path.exists(data_card):
        with open(data_card, "r", encoding="utf-8") as f:
            data_flops = float(json.load(f).get("data_flops", 0.0))
    return policy, data_flops


def main():
    parser = argparse.ArgumentParser(description="Official evaluation with compute accounting")
    parser.add_argument("--submission", type=str, default=None, help="path to a packaged submission directory (overrides --model)")
    parser.add_argument("--model", type=str, default=None, help="model spec: sft:d12, sft:d12:500, base:d12, rl:d12, or a checkpoint directory")
    parser.add_argument("--config", type=str, default=None, help=argparse.SUPPRESS) # staff diagnostics: another solver config (json); never for --submission
    parser.add_argument("--data-flops", type=float, default=0.0, help="declared C_data (FLOPs spent by models on your data)")
    parser.add_argument("--suite", type=str, default="data/val", help="directory of the evaluation suite")
    parser.add_argument("--families", type=str, default="math,code", help="comma-separated families to evaluate")
    parser.add_argument("--max-problems", type=int, default=-1, help="evaluate a random subset of this many problems (-1 = all)")
    parser.add_argument("--batch-size", type=int, default=256, help="problems solved together (1 = one at a time)")
    parser.add_argument("--output", type=str, default=None, help="write a results json here")
    parser.add_argument("--save-outputs", type=str, default=None, help="write per-problem outputs (jsonl) here, for error analysis")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # Resolve what to evaluate
    config = OFFICIAL_CONFIG
    if args.submission is not None:
        model_spec, data_flops = load_submission(args.submission)
    else:
        assert args.model, "Provide --model (or --submission)"
        model_spec, data_flops = args.model, args.data_flops
        if args.config is not None: # staff diagnostics only
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)

    # Load the model and the official solver
    model, tokenizer, meta = load_model_spec(model_spec, device, phase="eval")
    lms = {"policy": MeteredLM(model, tokenizer)}
    solver = load_solver_class(SOLVER)(lms, config)
    print0(f"Model: {model_spec} | official solver | config: {config}")

    # Load the suite
    items = load_suite(args.suite, families=tuple(args.families.split(",")))
    if 0 < args.max_problems < len(items):
        items = random.Random(0).sample(items, args.max_problems)
    print0(f"Evaluating on {len(items)} problems from {args.suite}")

    # Go
    t0 = time.time()
    with torch.inference_mode():
        records = run_suite(lms, solver, items, batch_size=args.batch_size)
    summary = summarize(records)
    report = budget_report(train_flops_of([meta]), data_flops)
    print0("=" * 80)
    print0(format_summary(summary))
    print0("-" * 80)
    print0(format_allowance(summary))
    print0(format_budget_report(report))
    print0(f"Evaluation time: {time.time() - t0:.0f}s")

    if ddp_rank == 0 and args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"model": model_spec, "config": config, "suite": args.suite,
                       "summary": summary, "budget": report}, f, indent=2)
        print0(f"Wrote {args.output}")
    if ddp_rank == 0 and args.save_outputs:
        with open(args.save_outputs, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print0(f"Wrote {args.save_outputs}")
    compute_cleanup()


if __name__ == "__main__":
    main()
