"""
Compute accounting for the CS-433 compute-limited LLM project.

!!! DO NOT MODIFY THIS FILE. The official grading uses our own copy of it. !!!

Training compute is measured in FLOPs (floating point operations), which makes the
budget independent of the hardware you happen to have. The cost of a submission is

    C_total = C_pretrain + C_SFT + C_RL    <=    TOTAL_BUDGET_FLOPS

- C_pretrain, C_SFT, C_RL: every forward/backward pass that produced the weights of the
  submitted model, by training stage: pretraining (incl. continued pretraining / annealing,
  scripts/base_train.py), supervised fine-tuning (scripts/chat_sft.py) and reinforcement
  learning (scripts/chat_rl.py, incl. its rollouts). The training scripts log it automatically
  into the checkpoint metadata ("compute_ledger"), and every stage inherits the ledger of its parent.
- C_data: FLOPs spent by neural networks to select your training data (model-based scoring
  or filtering). It is NOT part of B, but you declare it (data config -> data card), and it is
  reported next to the budget.
Inference is NOT part of the budget either. Instead, every evaluation problem gets a fixed
allowance of INFERENCE_TOKENS_PER_PROBLEM tokens for the official solver. A generate() call uses  prompt length + num_samples * (length of the longest sample)
(the samples share one prompt prefill, and they decode in lockstep until the longest one is
done); a logprobs() call uses the length of every scored sequence. MeteredLM stops
generation when the allowance is used up, and a problem that still exceeds it (e.g. through
logprobs) is scored as wrong. Inference FLOPs are still measured, for information only.

FLOPs are computed analytically from the model's own estimate_* methods in gpt.py
(6N per trained token, 2N per inference token, plus attention). If you change the
architecture, keep those methods exact; scripts/check_flops.py verifies them.
"""

import math
import uuid
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# The budget (official values, announced in the project description)

TOTAL_BUDGET_FLOPS = 3e18       # B: total training budget (pretrain + SFT + RL), ~5 h of training on one A100-80GB
BUDGET_TOLERANCE = 0.05         # submissions within +5% of B are not penalized (measurement noise)
INFERENCE_TOKENS_PER_PROBLEM = 16384  # inference allowance per evaluation problem (all calls)

# -----------------------------------------------------------------------------
# FLOP formulas shared by all scripts (training, RL rollouts, evaluation)

def train_flops(model, num_tokens):
    """FLOPs of forward+backward over num_tokens tokens (all rows of all micro-batches)."""
    return model.estimate_flops() * num_tokens


def generation_flops(model, prompt_len, num_samples, num_generated):
    """
    FLOPs of one batched sampling call: a single prefill of the prompt (the KV cache is
    shared across samples), then num_generated decode steps for each of the num_samples
    rows. Rows that finish early keep occupying the batch until the longest row is done,
    so num_generated should be the length of the longest generated row.
    """
    flops = model.estimate_prefill_flops(prompt_len)
    for t in range(num_generated):
        flops += num_samples * model.estimate_decode_flops(prompt_len + t + 1)
    return flops


def scoring_flops(model, seq_lens):
    """FLOPs of forward passes (no generation) over a list of sequences, e.g. for reranking."""
    return sum(model.estimate_prefill_flops(n) for n in seq_lens)


def generation_tokens(prompt_len, num_samples, num_generated):
    """Inference allowance used by one sampling call (counted like generation_flops: one prefill, lockstep rows)."""
    return prompt_len + num_samples * num_generated

# -----------------------------------------------------------------------------
# The compute ledger lives inside the checkpoint metadata (meta_*.json)

def new_ledger():
    return {"stages": [], "train_flops": 0.0}


def ledger_from_meta(meta):
    """Return a copy of the ledger stored in a checkpoint's metadata (empty if absent)."""
    ledger = meta.get("compute_ledger") if meta else None
    if ledger is None:
        return new_ledger()
    return {"stages": [dict(s) for s in ledger["stages"]], "train_flops": float(ledger["train_flops"])}


def ledger_add_stage(ledger, stage, flops, **info):
    """Append a training stage to the ledger. `info` should describe what was done."""
    record = {
        "uid": uuid.uuid4().hex, # lets us count stages shared by several submitted models only once
        "stage": stage,
        "flops": float(flops),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **info,
    }
    ledger["stages"].append(record)
    ledger["train_flops"] = float(ledger["train_flops"]) + float(flops)
    return ledger

# -----------------------------------------------------------------------------
# Budget report

STAGE_TERMS = {"base_train": "pretrain", "sft": "sft", "rl": "rl"} # ledger stage -> budget term


def train_flops_by_term(metas):
    """{"pretrain", "sft", "rl"} -> FLOPs, from the ledgers of checkpoint metadata. Shared stages count once."""
    seen, terms = set(), {"pretrain": 0.0, "sft": 0.0, "rl": 0.0}
    for meta in metas:
        for stage in ledger_from_meta(meta)["stages"]:
            uid = stage.get("uid") or repr(sorted(stage.items()))
            if uid not in seen:
                seen.add(uid)
                terms[STAGE_TERMS.get(stage["stage"], "pretrain")] += stage["flops"]
    return terms


def budget_report(train_flops, data_flops=0.0, budget=TOTAL_BUDGET_FLOPS):
    """train_flops: {"pretrain", "sft", "rl"} -> FLOPs (or one number, all pretraining). data_flops is not counted in B."""
    terms = train_flops if isinstance(train_flops, dict) else {"pretrain": float(train_flops), "sft": 0.0, "rl": 0.0}
    total = terms["pretrain"] + terms["sft"] + terms["rl"]
    report = {
        "budget_flops": budget,
        "pretrain_flops": terms["pretrain"],
        "sft_flops": terms["sft"],
        "rl_flops": terms["rl"],
        "total_flops": total,
        "fraction_of_budget": total / budget if budget > 0 else math.inf,
        "within_budget": total <= budget * (1 + BUDGET_TOLERANCE),
        "data_flops": data_flops, # declared, not part of B
    }
    return report


def format_budget_report(report):
    b = report["budget_flops"]
    share = lambda x: f"({100 * x / b:6.2f}% of budget)"
    lines = [
        f"{'C_pretrain':>22s}: {report['pretrain_flops']:.3e} FLOPs {share(report['pretrain_flops'])}",
        f"{'C_SFT':>22s}: {report['sft_flops']:.3e} FLOPs {share(report['sft_flops'])}",
        f"{'C_RL':>22s}: {report['rl_flops']:.3e} FLOPs {share(report['rl_flops'])}",
        f"{'C_total':>22s}: {report['total_flops']:.3e} FLOPs ({100 * report['fraction_of_budget']:6.2f}% of budget B={b:.1e})",
        f"{'status':>22s}: {'OK, within budget' if report['within_budget'] else 'OVER BUDGET'}",
        f"{'C_data (not in B)':>22s}: {report['data_flops']:.3e} FLOPs declared for data selection",
    ]
    return "\n".join(lines)
