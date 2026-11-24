"""
The reinforcement-learning (RL) stage: a FIXED pipeline, used by scripts/chat_rl.py.

!!! DO NOT MODIFY THIS FILE. RL is fixed: you only choose how many tokens of math and of code to train on. !!!

Problems: a FIXED selection of three post-training datasets (PROBLEM_SOURCES), built by the standard pipeline
(deduplicated, decontaminated) with `python -m scripts.prepare_data --rl`, independently of your data config. The
pipeline also drops the problems of a fixed exclusion list (rl_exclude.json), so the RL problems are the same for
everyone, graders included. Rewards are checked by a program:
    math  GSM8K train ("gsm8k"): 1 if the final number of the answer equals the reference answer, else 0
    code  half TinyGSM ("tinygsm", the python_math format): 1 if the program's solution() returns what the reference
          program returns; half MBPP ("mbpp"): 1 if the program passes the test shown in the prompt
Algorithm: for every problem, NUM_SAMPLES answers are sampled at TEMPERATURE. The advantage of an answer is its reward
minus the mean reward of the answers to the same problem, and the policy gradient is taken on the sampled tokens
(prompt and calculator outputs are masked), normalized by the number of tokens. On-policy, no KL term: REINFORCE with
a group baseline, a simplified GRPO.
RL tokens: every token of every sampled sequence (prompt + answer). Each step takes PROBLEMS_PER_STEP problems of one
domain, choosing the domain whose budget is used least, so both budgets are used up at the same rate. The optimizer
is fixed too (setup_optimizer below: the baseline's Muon + AdamW, whatever you change in your own optimizer), and its
learning rates decay linearly to 0 over the total budget. Compute: rollouts (generation FLOPs) + policy-gradient passes
(training FLOPs) = C_RL, about COST_PER_TOKEN x estimate_flops() per RL token.
"""

import os
import json
import random

from torch import nn

from nanochat.optim import MuonAdamW
from nanochat.inference import item_seed
from nanochat.execution import execute_code
from tasks.eval_suite import extract_math_answer, math_answers_match, extract_program, run_tests

NUM_SAMPLES = 16          # answers per problem
PROBLEMS_PER_STEP = 16    # problems per optimization step (all GPUs together)
MAX_NEW_TOKENS = 256      # tokens per answer
TEMPERATURE = 1.0
TOP_K = 50
TRAIN_ROWS = 32           # sequences per forward/backward pass (memory only, not the result)
INIT_LR_FRAC = 0.01       # the optimizer's learning rates (below) are scaled by this, then decay linearly to 0 (0.05 made scores worse)
CODE_TIMEOUT = 5.0        # seconds per program
SOURCES = {"math": ["gsm8k"], "code": ["tinygsm", "mbpp"]}
PROBLEM_SOURCES = {       # how the problems are built (scripts/prepare_data.py --rl): post-training datasets of the catalog
    "gsm8k": {"registry": "gsm8k"},                          # 7.5K problems
    "tinygsm": {"registry": "tinygsm", "max_docs": 50000},   # 50K of the first shard
    "mbpp": {"registry": "mbpp"},                            # 474 problems (never the test split)
}
EXCLUDE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_exclude.json")
COST_PER_TOKEN = 1.35     # C_RL / (RL tokens x estimate_flops()), measured on the d12 reference: 1.33-1.37
MBPP_TEST_MARKER = "Your code should pass this test:\n"

# -----------------------------------------------------------------------------
# Problems and rewards

def excluded_problems(name):
    """
    Keys (nanochat.datapipe.doc_key) of the problems of RL source `name` that overlap the evaluation POOLS: all problems
    of the public datasets that the validation and hidden suites are drawn from. The pools contain the whole hidden
    suite, but not which of their problems it holds, so the list reveals nothing about it.
    """
    with open(EXCLUDE_FILE, "r", encoding="utf-8") as f:
        return frozenset(json.load(f)["sources"].get(name, []))


def message_text(content):
    return content if isinstance(content, str) else "".join(p["text"] for p in content if p.get("type") == "text")


def python_math_reference(conversation):
    """The value returned by the reference program of a TinyGSM conversation, or None if it does not run."""
    code = extract_program(message_text(conversation["messages"][-1]["content"]))
    result = execute_code(code + "\nprint(repr(float(solution())))", timeout=CODE_TIMEOUT)
    try:
        return float(result.stdout.strip().splitlines()[-1]) if result.success else None
    except (ValueError, IndexError):
        return None


def make_problem(source, index, conversation):
    """A problem: the conversation (prompt = its user turn) and what the reward needs. None if it cannot be checked."""
    problem = {"source": source, "id": f"{source}/{index}", "conversation": conversation}
    if source == "gsm8k":
        problem["answer"] = extract_math_answer(message_text(conversation["messages"][-1]["content"]))
        return problem if problem["answer"] is not None else None
    if source == "tinygsm":
        problem["answer"] = python_math_reference(conversation)
        return problem if problem["answer"] is not None else None
    if source == "mbpp":
        user = message_text(conversation["messages"][0]["content"])
        problem["test"] = user.split(MBPP_TEST_MARKER, 1)[1] if MBPP_TEST_MARKER in user else None
        return problem if problem["test"] else None
    raise ValueError(f"Unknown RL source: {source}")


def reward(problem, text):
    """1.0 if the sampled answer `text` solves the problem, else 0.0."""
    if problem["source"] == "gsm8k":
        return float(math_answers_match(extract_math_answer(text), problem["answer"]))
    code = extract_program(text)
    if not code.strip():
        return 0.0
    if problem["source"] == "tinygsm":
        ref = problem["answer"]
        test = f"_answer = solution()\nassert abs(float(_answer) - ({ref!r})) <= 1e-6 * max(1.0, abs({ref!r}))\n"
    else:
        test = problem["test"]
    return float(run_tests({"imports": ""}, code, test, timeout=CODE_TIMEOUT))


class ProblemStream:
    """An endless, seeded stream of checkable problems of one domain (sources alternate, each one is shuffled)."""

    def __init__(self, domain, tasks, seed=0):
        self.sources = SOURCES[domain]
        self.tasks = tasks # source name -> task (indexable conversations)
        self.orders = {s: [] for s in self.sources}
        self.positions = {s: 0 for s in self.sources}
        self.epochs = {s: 0 for s in self.sources}
        self.rng = random.Random(seed)
        self.turn = 0

    def _next_index(self, source):
        if self.positions[source] >= len(self.orders[source]):
            self.orders[source] = list(range(len(self.tasks[source])))
            self.rng.shuffle(self.orders[source])
            self.positions[source] = 0
            self.epochs[source] += 1
        index = self.orders[source][self.positions[source]]
        self.positions[source] += 1
        return index

    def take(self, n):
        problems = []
        while len(problems) < n:
            source = self.sources[self.turn % len(self.sources)]
            self.turn += 1
            index = self._next_index(source)
            problem = make_problem(source, index, self.tasks[source][index])
            if problem is not None:
                problems.append(problem)
        return problems

# -----------------------------------------------------------------------------
# Rollouts and the schedule

def next_domain(done, budget):
    """The domain whose token budget is used least, among those with tokens left; None when all are used up."""
    open_domains = [d for d in ("math", "code") if budget.get(d, 0) > 0 and done.get(d, 0) < budget[d]]
    return min(open_domains, key=lambda d: done.get(d, 0) / budget[d]) if open_domains else None


def rollout_seed(problem, step):
    return item_seed(problem["id"], salt=step)


def loss_mask(prompt_len, completion, special):
    """1 for the tokens the model sampled, 0 for the prompt and for calculator outputs (<|output_start|> ... <|output_end|>
    right after <|python_end|>, which the engine forces)."""
    mask, forced = [0] * prompt_len, False
    for i, token in enumerate(completion):
        if token == special["output_start"] and i > 0 and completion[i - 1] == special["python_end"]:
            forced = True
        mask.append(0 if forced else 1)
        if forced and token == special["output_end"]:
            forced = False
    return mask


def training_rows(prompt, completions, rewards, special, max_new_tokens=MAX_NEW_TOKENS):
    """(tokens, mask, advantage) for every answer to one problem. Answers that ended get their <|assistant_end|> (trained on)."""
    mean = sum(rewards) / len(rewards)
    rows = []
    for completion, r in zip(completions, rewards):
        tokens = completion + ([special["assistant_end"]] if len(completion) < max_new_tokens else [])
        rows.append((prompt + tokens, loss_mask(len(prompt), tokens, special), r - mean))
    return rows

# -----------------------------------------------------------------------------
# The optimizer (fixed): MuonAdamW with the parameter groups and hyperparameters that the baseline's
# GPT.setup_optimizer gave RL, so RL trains the same way whatever you change in your own optimizer.
# Muon for the matrices of the Transformer blocks (momentum 0.95, no weight decay), AdamW for the rest.

# AdamW groups: learning rate (before INIT_LR_FRAC), whether it is scaled by (model_dim / 768) ** -0.5, betas, weight decay
ADAMW_GROUPS = {
    "lm_head":      (0.004, True, (0.8, 0.96), 0.01),
    "embedding":    (0.2, True, (0.8, 0.995), 0.001),
    "value_embeds": (0.1, True, (0.8, 0.995), 0.01),
    "resid":        (0.005, False, (0.8, 0.95), 0.05),
    "x0":           (0.5, False, (0.96, 0.95), 0.0),
    "scalars":      (0.2, False, (0.8, 0.95), 0.0),
}
MUON_GROUP = dict(lr=0.02, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0)
BASELINE_PARAMS = {"lm_head.weight": "lm_head", "transformer.wte.weight": "embedding", "resid_lambdas": "resid",
                   "x0_lambdas": "x0", "smear_gate.weight": "scalars", "smear_lambda": "scalars", "backout_lambda": "scalars"}


def optimizer_group(name, param, module, vocab_size):
    """The group of a parameter: by name for the baseline architecture, by kind for parameters of a changed architecture."""
    if name in BASELINE_PARAMS:
        return BASELINE_PARAMS[name]
    if name.startswith("value_embeds."):
        return "value_embeds"
    if isinstance(module, nn.Embedding):
        return "embedding"
    if name.startswith("transformer.h.") and param.ndim == 2:
        return "muon"
    if param.ndim == 2 and max(param.shape) >= vocab_size:
        return "lm_head" # a projection to the vocabulary
    if param.ndim == 2 and min(param.shape) > 1:
        return "muon"
    return "scalars"


def setup_optimizer(model):
    """The RL optimizer, with every learning rate x INIT_LR_FRAC (also recorded as initial_lr, for the decay)."""
    width_scale = (model.config.n_embd / 768) ** -0.5
    groups, seen = {}, set()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if id(param) not in seen: # a shared (tied) parameter is trained once
                seen.add(id(param))
                name = f"{module_name}.{param_name}" if module_name else param_name
                groups.setdefault(optimizer_group(name, param, module, model.config.vocab_size), []).append(param)
    param_groups = [dict(kind="adamw", params=groups[g], lr=lr * (width_scale if scaled else 1.0), betas=betas, eps=1e-10,
                         weight_decay=wd) for g, (lr, scaled, betas, wd) in ADAMW_GROUPS.items() if g in groups]
    matrices = groups.get("muon", [])
    param_groups += [dict(kind="muon", params=[p for p in matrices if p.shape == shape], **MUON_GROUP)
                     for shape in sorted({p.shape for p in matrices})]
    optimizer = MuonAdamW(param_groups)
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * INIT_LR_FRAC
        group["initial_lr"] = group["lr"]
    return optimizer
