"""
Reinforcement learning (RL): a FIXED pipeline. You choose only how many tokens of math and of code to train on.

!!! DO NOT MODIFY THIS FILE (nor nanochat/rl.py). Graders use their own copies. !!!

1 GPU:
python -m scripts.chat_rl --model-tag d12 --math-tokens 2e7 --code-tokens 1e7

4 GPUs:
torchrun --standalone --nproc_per_node=4 -m scripts.chat_rl -- --model-tag d12 --math-tokens 2e7 --code-tokens 1e7

The problems, rewards, algorithm, optimizer and hyperparameters are in nanochat/rl.py. RL starts from an SFT checkpoint
(--source sft, the default). Its problems are fixed (GSM8K for math, TinyGSM + MBPP for code) and do not depend on
your data config: build them once with python -m scripts.prepare_data --rl. RL tokens are the tokens of all sampled sequences, prompts included. Compute: the
rollouts AND the policy-gradient passes go into the ledger as the stage "rl" (C_RL). Plan it with
python -m scripts.plan_budget --rl-math-tokens ... --rl-code-tokens ...
"""

import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import wandb
import torch
import torch.distributed as dist

from nanochat import rl
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from nanochat.compute import generation_flops, train_flops, ledger_from_meta, ledger_add_stage, TOTAL_BUDGET_FLOPS
from nanochat.datapipe import fingerprints, read_manifest, exclude_signature
from tasks.registry import build_task

parser = argparse.ArgumentParser(description="Reinforcement learning (fixed pipeline): choose the math and code token budgets")
parser.add_argument("--math-tokens", type=float, default=0, help="RL tokens on math problems (GSM8K train)")
parser.add_argument("--code-tokens", type=float, default=0, help="RL tokens on code problems (TinyGSM python_math + MBPP)")
parser.add_argument("--source", type=str, default="sft", help="where to load the model from: sft|rl")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--output-tag", type=str, default=None, help="tag of the RL checkpoint (default: the model tag)")
parser.add_argument("--save-every", type=int, default=100, help="also save a checkpoint every N steps (for preemption)")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
args = parser.parse_args()
budget = {"math": float(args.math_tokens), "code": float(args.code_tokens)}
assert budget["math"] >= 0 and budget["code"] >= 0 and budget["math"] + budget["code"] > 0, "Set --math-tokens and/or --code-tokens"

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
assert rl.PROBLEMS_PER_STEP % ddp_world_size == 0, f"Use a number of GPUs that divides {rl.PROBLEMS_PER_STEP}"
wandb_run = DummyWandb() if args.run == "dummy" or not master_process else wandb.init(project="cs433-llm-rl", name=args.run, config=vars(args))

model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)
parent_ledger = ledger_from_meta(meta)
output_tag = args.output_tag or args.model_tag or f"d{model.config.n_layer}"
special = {name: tokenizer.encode_special(f"<|{name}|>") for name in ["assistant_end", "python_end", "output_start", "output_end"]}

# the problems (the same stream on every rank; each rank takes its share of every step)
domains = [d for d in ("math", "code") if budget[d] > 0]
sources = [s for d in domains for s in rl.SOURCES[d]]
data_fingerprints = fingerprints(sources, "rl") # asserts that the problems are built (prepare_data --rl)
assert all(read_manifest(s, "rl")["spec"] == rl.PROBLEM_SOURCES[s]
           and read_manifest(s, "rl")["decontamination"].get("exclude") == exclude_signature(rl.excluded_problems(s))
           for s in sources), "Rebuild the RL problems: python -m scripts.prepare_data --rl"
tasks = {s: build_task(s, kind="rl")[0] for s in sources}
streams = {d: rl.ProblemStream(d, tasks, seed=1234 + i) for i, d in enumerate(domains)}
print0(f"RL budget: {budget['math']:.3g} math tokens + {budget['code']:.3g} code tokens | sources: {sources}")

optimizer = rl.setup_optimizer(model) # fixed, not your setup_optimizer: RL trains the same way for every team

done = {"math": 0.0, "code": 0.0}
flops = {"rollouts": 0.0, "policy_gradient": 0.0} # this rank
pool = ThreadPoolExecutor(max_workers=8) # rewards of code problems run programs in subprocesses


def save(step):
    totals = torch.tensor([flops["rollouts"], flops["policy_gradient"]], dtype=torch.float64, device=device)
    if ddp:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if not master_process:
        return
    checkpoint_dir = os.path.join(get_base_dir(), "chatrl_checkpoints", output_tag)
    ledger = ledger_add_stage(ledger_from_meta({"compute_ledger": parent_ledger}), "rl", float(totals.sum()),
                              run=args.run, model_tag=output_tag, step=step, rollout_flops=float(totals[0]),
                              math_tokens=done["math"], code_tokens=done["code"], data={"rl": data_fingerprints},
                              parent=f"{args.source}:{args.model_tag}:{args.model_step}")
    save_checkpoint(checkpoint_dir, step, model.state_dict(), None,
                    {"model_config": model.config.__dict__, "compute_ledger": ledger})
    print0(f"Saved {checkpoint_dir} step {step}: C_RL = {ledger['stages'][-1]['flops']:.3e} FLOPs | cumulative "
           f"{ledger['train_flops']:.3e} ({100 * ledger['train_flops'] / TOTAL_BUDGET_FLOPS:.1f}% of budget)")


step, t0 = 0, time.time()
while (domain := rl.next_domain(done, budget)) is not None:
    problems = streams[domain].take(rl.PROBLEMS_PER_STEP)[ddp_rank::ddp_world_size]

    # 1) rollouts: all answers to all problems of this rank in one batch
    prompts = [tokenizer.render_for_completion(p["conversation"]) for p in problems]
    model.eval()
    with torch.no_grad():
        completions, decode_steps = engine.generate_multi(prompts, num_samples=rl.NUM_SAMPLES, max_tokens=rl.MAX_NEW_TOKENS,
                                                          temperature=rl.TEMPERATURE, top_k=rl.TOP_K,
                                                          seeds=[rl.rollout_seed(p, step) for p in problems])
    for prompt, n in zip(prompts, decode_steps):
        flops["rollouts"] += generation_flops(model, len(prompt), rl.NUM_SAMPLES, n)

    # 2) rewards and advantages
    jobs = [(p, tokenizer.decode(c)) for p, cs in zip(problems, completions) for c in cs]
    rewards = list(pool.map(lambda job: rl.reward(*job), jobs))
    rows = []
    for i, (prompt, cs) in enumerate(zip(prompts, completions)):
        rows.extend(rl.training_rows(prompt, cs, rewards[i * rl.NUM_SAMPLES:(i + 1) * rl.NUM_SAMPLES], special))
    num_tokens = sum(len(tokens) for tokens, _, _ in rows)

    # 3) policy gradient (rows sorted by length to keep the padding small)
    model.train()
    rows.sort(key=lambda row: len(row[0]))
    num_valid = max(1, sum(sum(mask[1:]) for _, mask, _ in rows))
    for b in range(0, len(rows), rl.TRAIN_ROWS):
        batch = rows[b:b + rl.TRAIN_ROWS]
        width = max(len(tokens) for tokens, _, _ in batch)
        ids = torch.tensor([t + [special["assistant_end"]] * (width - len(t)) for t, _, _ in batch], dtype=torch.long, device=device)
        mask = torch.tensor([m + [0] * (width - len(m)) for _, m, _ in batch], dtype=torch.long, device=device)
        advantages = torch.tensor([a for _, _, a in batch], dtype=torch.float, device=device)
        inputs, targets = ids[:, :-1], ids[:, 1:].clone()
        targets[mask[:, 1:] == 0] = -1 # prompt, calculator outputs and padding are not trained on
        logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)
        loss = -(logp * advantages.unsqueeze(-1)).sum() / num_valid
        loss.backward()
        flops["policy_gradient"] += train_flops(model, inputs.numel())

    # 4) bookkeeping (token counts of all ranks), learning rate, update
    stats = torch.tensor([num_tokens, sum(rewards), len(rewards)], dtype=torch.float64, device=device)
    if ddp:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    done[domain] += float(stats[0])
    progress = min(1.0, (done["math"] + done["code"]) / (budget["math"] + budget["code"]))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * (1.0 - progress)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    step += 1
    mean_reward = float(stats[1] / stats[2])
    print0(f"Step {step} | {domain} | mean reward {mean_reward:.3f} | math {done['math']:.3g}/{budget['math']:.3g} "
           f"code {done['code']:.3g}/{budget['code']:.3g} tokens | {time.time() - t0:.0f}s")
    wandb_run.log({"step": step, f"reward/{domain}": mean_reward, "tokens/math": done["math"], "tokens/code": done["code"]})
    if step % args.save_every == 0:
        save(step)

save(step)
pool.shutdown()
wandb_run.finish()
compute_cleanup()
