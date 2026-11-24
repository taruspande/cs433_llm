# CS-433 Project 2, Option A: Compute-Limited LLM Training for Math & Code

You get a fixed compute budget, counted in FLOPs, and an empty model. With that budget you train a small
language model **from scratch**, to score as high as possible on a **hidden** suite of grade-school math and basic
Python programming problems. Every model is evaluated the same way: one greedy answer per problem.

You choose how to spend the budget:

| Lever | Examples |
|---|---|
| **Model scale** | depth/width, parameters vs. training tokens (Chinchilla-style trade-offs) |
| **Model architecture** | attention variants, MLP, normalization, embeddings, context length |
| **Data scale** | how many tokens, how many epochs over high-quality data |
| **Data recipe** | which catalog datasets, filtering/selection, deduplication, repetition, mixture weights |
| **Training pipeline** | pretraining → SFT → RL (fixed; you choose how many math and code tokens), budget per stage |

The code is a stripped-down, adapted fork of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
(MIT license). It is small enough to read end to end. Please do.

> The full project description (rules, grading rubric, deadlines) is in the PDF handout. This README covers the code.
> If they disagree, the PDF wins.

---

## Contents

1. [The task in one formula](#1-the-task-in-one-formula)
2. [Setup](#2-setup)
3. [Quickstart: smoke test and baseline](#3-quickstart-smoke-test-and-baseline)
4. [Evaluation, step by step (and how we grade)](#4-evaluation-step-by-step-and-how-we-grade)
5. [The pipeline, stage by stage](#5-the-pipeline-stage-by-stage)
6. [Compute accounting](#6-compute-accounting) (and [planning your budget](#61-plan-your-budget-before-you-train))
7. [What you may not change](#7-what-you-may-not-change)
8. [Rules on data, models and evaluation](#8-rules-on-data-models-and-evaluation)
9. [Making a submission](#9-making-a-submission)
10. [Repository layout](#10-repository-layout)
11. [Hardware notes and troubleshooting](#11-hardware-notes-and-troubleshooting)
12. [Ideas to get you started](#12-ideas-to-get-you-started)

---

## 1. The task in one formula

Maximize the score on the hidden test suite, subject to

```
C_total = C_pretrain + C_SFT + C_RL  ≤  B = 3·10^18 FLOPs
```

B is about **5 hours of training on one NVIDIA A100-80GB** at the baseline's throughput (1.7·10¹⁴ FLOP/s). The
budget is in FLOPs, not GPU hours, because it is the same on every GPU and can be verified from your checkpoints.
Faster code buys you iteration speed, not budget. Data selection (C_data) and inference are not part of the budget
(see below).

**You do not need to optimize everything.** A good project studies one or two levers in depth and keeps the rest
at the baseline. Examples:
- data sources and quality only;
- model scale/architecture vs. data scale only;
- post-training only.

See the PDF handout ("Choose your focus").

* **C_pretrain**, **C_SFT** and **C_RL** count every forward and backward pass that produced your submitted weights,
  by stage: pretraining (`base_train`), supervised fine-tuning
  (`chat_sft`) and reinforcement learning (`chat_rl`, rollouts included). The training scripts log them
  automatically into each checkpoint's **compute ledger**, and every stage inherits its parent's ledger. RL is
  optional and fixed: you only choose how many tokens of math and of code it trains on (§5.5).
* **C_data** (not part of B) counts FLOPs spent by neural networks to *select* training data, e.g. scoring documents
  with a quality classifier or with the perplexity of a small model. It is not counted in the budget, but you must
  declare it per source (`data_flops`) in your data config; it is summed from the data manifests into your data card
  and reported next to the budget. Rule-based filters, hashing and deduplication cost nothing, and neither does
  downloading the datasets of the catalog. All training data comes from a fixed **dataset catalog** (§5.1), and you
  may not create synthetic data of your own (§8).
* **Inference is fixed and not charged.** Every model is evaluated with the **official solver** in one fixed
  configuration: one answer per problem, generated greedily, of at most 512 tokens (§5.6). You choose nothing at test
  time.

To compute C_total of an experiment design before you run it: `python -m scripts.plan_budget` (§6.1).

The score is the mean of a **math** score and a **code** score (50/50). Each is the unweighted mean accuracy over
its two sub-benchmarks, one easy and one moderately hard for a model of this size:

| Family | Subset (validation / hidden test size) | What it is | Correct if… |
|---|---|---|---|
| math | `arithmetic` (200 / 400) | GPT-3's arithmetic battery: "What is 98 plus 45?" (2–3-digit +/−, 2-digit ×, 1-digit compound) | the extracted final number equals the reference |
| math | `gsm8k_easy` (147 / 292) | GSM8K test problems that need at most 2 calculation steps | 〃 |
| code | `python_math` (200 / 373) | write `solution()` that returns the answer to a simple word problem (MAWPS) | the returned function returns the right number |
| code | `mbpp_easy` (91 / 178) | MBPP problems with a short reference solution (≤ 3 lines); one test is shown | the program passes all hidden unit tests ([MBPP+](https://github.com/evalplus/evalplus) where available) |

The **validation suite is in `data/val/`**, with answers and tests, so you can select checkpoints and tune things
on it. The **test suite is hidden**. It comes from the same sources, but no problem appears in both (near-duplicates
are removed too), and it may contain newly written problems of the same style.

---

## 2. Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependencies and needs Python ≥ 3.10.

```bash
git clone <your team repo> && cd <repo>
uv sync --extra gpu          # CUDA machines (A100 / H100 / ...)
# uv sync --extra cpu        # laptops (CPU / Apple Silicon), for development only
source .venv/bin/activate
```

All data, tokenizers and checkpoints go to `$NANOCHAT_BASE_DIR` (default `~/.cache/nanochat`). Point it at a large,
fast disk on the cluster:

```bash
export NANOCHAT_BASE_DIR=/scratch/$USER/nanochat
```

Optional: log runs to [Weights & Biases](https://wandb.ai) with `wandb login`, then pass `--run=<name>` to the
training scripts. With the default `--run=dummy`, nothing is logged.

---

## 3. Quickstart: smoke test and baseline

### 3.1 Smoke test (laptop or GPU, about 10 minutes)

```bash
bash runs/smoke_cpu.sh
```

This trains a tiny model on very little data and runs every stage: data → tokenizer → FLOP check → pretraining →
SFT → evaluation → submission packaging. **The scores will be about 0.** The point is to check that your setup
works. Rerun it after every code change you make. It installs the CPU build of PyTorch on a laptop and keeps the CUDA
build on a GPU machine.

### 3.2 The baseline (GPU)

```bash
bash runs/baseline.sh                  # 1 GPU
NPROC=8 bash runs/baseline.sh          # 8 GPUs on one node, same result, ~8x faster
```

[`runs/baseline.sh`](runs/baseline.sh) is the baseline recipe. It is nanochat's recipe, scaled down:

| Stage | What | Compute (FLOPs) | % of B |
|---|---|---|---|
| Data | 4 shards of [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (sample-10BT) general web text | free | 0 |
| Tokenizer | BPE, 32K vocab, trained on ~2B characters | free (CPU) | 0 |
| Pretraining | `d12` Transformer (12 layers, width 768, 110M scaling params / 286M total incl. embeddings), context 1024, 1.32B tokens (12 tokens/param), Muon + AdamW | 1.02e18 | 34% |
| SFT | GSM8K-train ×4 (with calculator tool calls) + MBPP-train ×8 + 3% of TinyGSM (~21K Python solutions) + 10% of SmolTalk, 85 steps | 3.45e16 | 1.1% |
| RL | not used by the baseline | 0 | 0 |
| Inference | fixed for every team: greedy decoding, 1 sample, ≤ 512 tokens (307 tokens per problem on average) | not charged | – |
| **Total** | | **1.057e18** | **35.2%** |

Measured wall-clock on 1×A100-80GB (bf16, ~55% MFU): pretraining 99 min, SFT 3 min, evaluation of the full
validation suite 1 min with the official (batched) solver. An H100 should be roughly 2–3×
faster for training.

Validation-suite scores of the baseline (accuracy in %, full validation suite):

| | math: arithmetic / gsm8k_easy | code: python_math / mbpp_easy | **final** | C_total |
|---|---|---|---|---|
| baseline | **4.74**: 2.0 / 7.5 | **13.0**: 26.0 / 0.0 | **8.87** | 35.2% of B |

Two runs of this same script can differ by a few points: GPU training is not bit-for-bit reproducible. Another run
of the baseline scored 5.95 (`python_math` 15.5). Compare your own runs at the same settings, and expect noise.

The baseline is deliberately naive. It spends about a third of the budget and pretrains on generic web text only,
so there is a lot of room to improve.

---

## 4. Evaluation, step by step (and how we grade)

Everything you are graded on is computed by **one script**, [`scripts/evaluate.py`](scripts/evaluate.py). You have
it, and the graders run the original copy of it. If it works on the validation suite on your side, it works on the
hidden suite on ours.

### 4.1 What is evaluated

* **Your submission** (§9): one model checkpoint, the data manifest and data card, and your experiment runner.
* **On the hidden test suite**: the same 4 subsets as the validation suite (§1), about 1,240 problems, rendered with
  the same prompts.
* **Score**: accuracy per subset, unweighted mean per family, then ½ math + ½ code (0–100).
* **Budget**: C_total = C_pretrain + C_SFT + C_RL (from the ledger of your checkpoint) must be ≤ B (+5% tolerance).
  C_data (from your data manifests) is reported next to it, but not counted.
* **Inference**: fixed for every team: the official solver, one greedy answer of at most 512 tokens per problem.

### 4.2 Step 1: evaluate a checkpoint on the validation suite

```bash
python -m scripts.evaluate --model sft:d12                                   # official solver, all 638 problems
python -m scripts.evaluate --model sft:d12 --families math --max-problems 300     # quick check during development
python -m scripts.evaluate --model sft:d12 --output results/d12.json --save-outputs results/d12_outputs.jsonl
```

`--model` takes `base:<tag>[:step]`, `sft:<tag>[:step]`, `rl:<tag>[:step]` or a checkpoint directory. The script
prints:

```
           code/mbpp_easy:   0.00%  (n=91)       <- accuracy per subset
         code/python_math:  26.00%  (n=200)
          math/arithmetic:   2.00%  (n=200)
          math/gsm8k_easy:   7.48%  (n=147)
              family code:  13.00%               <- unweighted mean of the subsets
              family math:   4.74%
              FINAL SCORE:   8.87                <- 1/2 math + 1/2 code: what is graded
---------------------------------------------------------------------
      inference tokens: mean 307 / max 648 per problem (allowance 16,384; inference FLOPs, not charged: 6.99e+10 per problem)
            C_pretrain: 1.022e+18 FLOPs ( 34.08% of budget)   <- from the ledger of the checkpoint
                 C_SFT: 3.449e+16 FLOPs (  1.15% of budget)
                  C_RL: 0.000e+00 FLOPs (  0.00% of budget)
               C_total: 1.057e+18 FLOPs ( 35.23% of budget B=3.0e+18)
                status: OK, within budget
     C_data (not in B): 0.000e+00 FLOPs declared for data selection   <- from --data-flops (submission: data card)
```

`--save-outputs` writes every problem's raw output, extracted answer, correctness, tokens and FLOPs. Use it for error analysis.

### 4.3 Step 2: select a checkpoint

Save intermediate checkpoints (`--save-every` in `chat_sft.py` / `base_train.py`) and evaluate each one on the
validation suite **with the same settings**:

```bash
for step in 25 50 75; do python -m scripts.evaluate --model sft:d12:$step --output results/d12_s$step.json; done
```

Report uncertainty: with 91–200 problems per subset, a subset accuracy has a standard error of up to 3.5 points,
and a family score about 1.5 points. Differences smaller than that are noise. Never train on the validation suite.

### 4.4 Step 3: package and dry-run exactly what the graders run

```bash
python -m scripts.make_submission --model sft:d12_final --runner runs/experiment.sh --out submission/
python -m scripts.evaluate --submission submission/          # must end with "status: OK, within budget"
```

`--submission` evaluates the packaged directory: its checkpoint and tokenizer with the official solver (fixed
configuration), and the C_data in `data_card.json`. That's exactly what we run, except on the hidden suite. Time it: the validation
suite is about half the size of the hidden suite.

### 4.5 What the graders run

1. **Use our evaluation code.** We take your repository at its tagged commit and put back our own copy of the
   evaluation code (the "do not change" files of §7), so every team is evaluated with exactly the same code.
2. `python -m scripts.check_flops` on your architecture: the FLOP estimates must be within 5%.
3. `python -m scripts.evaluate --submission <your submission> --suite <hidden suite>`, with the **default
   settings** (`--batch-size 256`, one NVIDIA A100-80GB GPU, no network access, time limit 2 hours).
4. Re-check your data: every source in `data_manifest.json` is rebuilt or checked against the hidden suite with
   the standard pipeline, and the fingerprints are compared.
5. For some teams (e.g. the top of the leaderboard, or when something looks off), re-run your experiment runner
   from scratch and compare the compute ledger, the data fingerprints and the validation score.
6. Grade from `final_score` and the budget report (rubric in the PDF handout).

**Reproducibility.** On GPUs, scores move by a few tenths of a point with the batch size and the number of GPUs.
bf16 numerics depend on the batch shape. Sampling is seeded per problem, so the same settings give the same
results. Compare your own runs at the same settings.

**Common pitfalls:** data was rebuilt after training, so the fingerprints don't match (`make_submission` warns);
the experiment runner depends on a manual step or on a file that is not in the repository.

---

## 5. The pipeline, stage by stage

All commands are run from the repository root. Every script has `--help`.

### 5.1 Data: the standard pipeline

**All training data comes from a fixed catalog of public datasets and goes through one pipeline**, so that every
team's data is processed the same way and every checkpoint can be traced back to the exact data it was trained on.

**The dataset catalog** ([`scripts/prepare_data.py`](scripts/prepare_data.py), `python -m scripts.prepare_data
--list`). These are the only datasets you may train on. It has two disjoint parts, and every dataset is in exactly
one of them:
- **pretraining datasets**: large text corpora, read as documents by `tok_train` and `base_train`;
- **post-training datasets**: problems with solutions, and chat, read as conversations by `chat_sft`. RL uses a
  fixed set of problems from three of them (§5.5).

"LLM-generated" datasets were written by models and published by others before the project; you may use them, but
not create such data yourself (§8).

| Pretraining dataset | Source on the HF Hub | What it is | LLM-generated |
|---|---|---|---|
| `fineweb` | HuggingFaceFW/fineweb (sample-10BT) | general web text (the baseline's data): lightly filtered Common Crawl, *not* deduplicated across dumps, so there is room for your own filtering and dedup; ~0.7B tokens per shard | – |
| `finemath` | HuggingFaceTB/finemath (4+) | educational math web pages | – |
| `codeparrot` | codeparrot/codeparrot-clean | deduplicated Python files from GitHub | – |

| Post-training dataset | What it is (conversations) | LLM-generated | RL problems (fixed) |
|---|---|---|---|
| `gsm8k` | GSM8K train: 7.5K problems, solutions with calculator tool calls, answer after `####` | – | math: all |
| `mbpp` | MBPP train/prompt/validation splits (474 problems, never the test split) | – | code: all |
| `tinygsm` | TinyGSM, first shard: ~700K problems with Python solutions, in the `python_math` format | GPT-3.5 | code: 50K |
| `metamathqa` | MetaMathQA, GSM8K-derived subset (~240K) | GPT-3.5 | – |
| `magicoder` | Magicoder OSS-Instruct, Python subset (~38K) | GPT-3.5 | – |
| `smoltalk` | smol-smoltalk general chat (460K conversations) | yes | – |

**What you train on is up to you** (see "What you can change" at the end of this section). You declare your data in a
**data config** and build it with one command:

```bash
python -m scripts.prepare_data --list                                 # the catalog, and what you have built
python -m scripts.prepare_data --config configs/data/baseline.json   # build everything the config declares
python -m scripts.prepare_data --rl                                   # the fixed RL problems (only if you use RL)
```

```json
{
  "pretrain": {
    "fineweb":  {"registry": "fineweb", "num_shards": 4},
    "finemath": {"registry": "finemath", "num_shards": 4, "max_docs": 500000},
    "web_en":   {"registry": "fineweb", "num_shards": 4, "filter": "my_filters:confident_english"}
  },
  "sft": {
    "gsm8k":   {"registry": "gsm8k"},
    "tinygsm": {"registry": "tinygsm", "max_docs": 21000}
  }
}
```

Every source names a catalog dataset (`registry`) under a name of your choice, so the same dataset can be built
several times, e.g. with different filters. A **filter** is a function of yours that only *selects*:

```python
# my_filters.py, in the repository root
def confident_english(document, record):   # document: the text (or SFT conversation); record: the raw row
    return record["language_score"] >= 0.9  # True = keep. A filter cannot change the document.
```

The fields of `record` are listed by `--list` (e.g. `language_score`, `url`, `token_count` for FineWeb). Filters run in
order in one process, so they may keep state, e.g. a MinHash index for near-duplicate removal. A filter that uses a
model (a quality classifier, perplexity under a small model) costs `data_flops` = 2 × params × tokens scored.

Every source goes through the same stages ([`nanochat/datapipe.py`](nanochat/datapipe.py)):

| Stage | What happens |
|---|---|
| 1. read | the catalog dataset (downloaded once into `$NANOCHAT_BASE_DIR/raw/`) |
| 2. format | record → document, with the catalog's formatter (fixed) |
| 3. select | drop empty documents; optional `"filter": "my_module:my_fn"`, called as `my_fn(document, record)` → keep? |
| 4. dedup | exact deduplication within the source |
| 5. decontaminate | drop documents that overlap an evaluation problem (13/8-word n-grams; very short problems verbatim). **Automatic and mandatory.** |
| 6. split | seeded shuffle, held-out validation split |
| 7. write | `data/<name>/{train,val}/*.parquet` (pretraining), `data/sft/<name>/{train,val}.jsonl` (SFT), `data/rl/<name>/` (RL problems) |
| 8. manifest | `manifest.json`: spec, inputs, counts after every stage, removed items, `data_flops`, fingerprint |

* **Only built sources can be used.** `tok_train`, `base_train`, `chat_sft` and `chat_rl` refuse sources without a
  manifest. They record each source's fingerprint in the checkpoint's compute ledger.
* **Only catalog datasets, each in its own part.** `prepare_data` refuses anything else (local files, other
  datasets, a pretraining dataset as an SFT source or the reverse).
  `make_submission` turns the manifests into your data card and sums the `data_flops` into C_data (declared, not in B).
* Options per source: `num_shards` (pretraining: how many parquet files to download), `max_docs`, `val_docs` /
  `val_fraction`, `seed`, `filter`, `data_flops`, `notes`. An unchanged source is not rebuilt.
* Building data is CPU work (download + decontamination). On a cluster, run it in a CPU-only job before training.

A **mixture** assigns token proportions to sources:

```bash
--mixture configs/mixtures/baseline.json                  # a json file {"sources": {"fineweb": 1.0}}
--mixture "fineweb:0.6,finemath:0.2,codeparrot:0.2"       # or inline
```

The dataloader ([`nanochat/dataloader.py`](nanochat/dataloader.py)) streams every source independently and always
pulls from the source furthest below its target share of tokens. The logs show a per-source **epoch counter**, so you
can see when you start repeating a small, high-quality source.

**What you can change**, all in your data config (start from a copy of
[`configs/data/example_math_code.json`](configs/data/example_math_code.json)):
- **Which datasets**: add or remove catalog datasets. The name of each entry is yours: the mixtures of §5.3–5.4 use it.
  Keep the entry `"fineweb": {"registry": "fineweb", "num_shards": 4}` unchanged, because the tokenizer is trained on
  it (§5.2); to train on other FineWeb documents, build them under another name.
- **How much of each**: `num_shards` (pretraining datasets: how many files to download) and `max_docs`, e.g.
  `"finemath": {"registry": "finemath", "num_shards": 8}` or `"tinygsm": {"registry": "tinygsm", "max_docs": 50000}`.
- **Which documents**: a selection filter, `"filter": "my_filters:confident_english"` (the example above). It returns
  `True` to keep a document and may not change the text. Build the same dataset several times under different names to
  compare filters, or to repeat a filtered subset.
- **Selection with a model**: a filter may score documents with an open-weight model. Declare the FLOPs it spent as
  `"data_flops"` (C_data, declared but not counted in B).
- **How much of each source you train on**: not here, but in the mixtures of pretraining (§5.3) and SFT (§5.4).

You cannot use other data or create new data (§8).

### 5.2 Tokenizer (fixed)

Every team uses the same tokenizer: the baseline's BPE tokenizer with 32,768 tokens, trained on 2B characters of the
source `fineweb` built as in [`configs/data/baseline.json`](configs/data/baseline.json) (4 shards). Train it once, with
no options (CPU work, free):

```bash
python -m scripts.prepare_data --config configs/data/baseline.json --only fineweb   # if not built yet
python -m scripts.tok_train
```

**Nothing can be changed here**: not the vocabulary, the training text, or how text is cut into tokens.
[`nanochat/tokenizer.py`](nanochat/tokenizer.py) and [`scripts/tok_train.py`](scripts/tok_train.py) are fixed (§7),
and `tok_train` refuses any option and any other build of `fineweb`.

### 5.3 Pretraining: `scripts/base_train.py`

```bash
python -m scripts.base_train --depth 12 --mixture <mixture> --num-iterations 2520 --model-tag d12
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth 12 ...   # multi-GPU
```

* **Checkpoints and resuming**: `--save-every N`, then `--resume-from-step N` (same arguments). Checkpoints go to
  `$NANOCHAT_BASE_DIR/base_checkpoints/<model-tag>/`.
* At startup the script prints the run's projected FLOPs as a fraction of the budget. Check it before launching a
  long job.

**What you can change** (`python -m scripts.base_train --help` lists every flag):
- **Model size**: `--depth 16`, the number of Transformer layers. The **width** d is the model's hidden dimension:
  the size of each token's vector between layers. It is depth × `--aspect-ratio` (default 64), rounded up to a
  multiple of `--head-dim` (default 128). A d12 has d = 768 with 6 attention heads of 128; a d16 has d = 1,024 with 8
  heads. Each layer's MLP has an inner size of 4d. §6 gives the cost per token. The learning rates and weight decay
  follow the size automatically (muP-style transfer from d12).
- **Training length**: `--num-iterations 3000`, the number of optimization steps (required). The tokens trained on
  are steps × batch size × sequence length: 2,520 × 512 × 1,024 = 1.32B for the baseline. `plan_budget` prints the
  number that matches your plan (§6.1).
- **Batch size and sequence length**: `--batch-size 256`, the number of sequences per step (default: chosen from the
  model size, 512 for a d12), and `--max-seq-len 2048`, the tokens per sequence (also the model's context; default
  1024). `--device-batch-size` only sets the memory use, not the result.
- **Data mixture**: `--mixture "fineweb:0.6,finemath:0.3,codeparrot:0.1"`, the token proportions of built pretraining
  sources, or a JSON file like [`configs/mixtures/example_math_code.json`](configs/mixtures/example_math_code.json).
- **Learning rates and weight decay**: `--matrix-lr 0.03` (Muon), `--embedding-lr`, `--unembedding-lr`,
  `--scalar-lr` (AdamW), `--weight-decay 0.2`, `--warmup-steps 100`, `--warmdown-ratio 0.4`, `--final-lr-frac 0`.
- **Optimizer**: Muon and AdamW only, both in [`nanochat/optim.py`](nanochat/optim.py). Do not modify that file, and
  do not use other optimizers (Lion, Shampoo, SOAP, SAM, `torch.optim`, ...). By default Muon trains the matrices of the
  Transformer blocks and AdamW everything else (embeddings, output layer, scalars). `GPT.setup_optimizer` in
  [`nanochat/gpt.py`](nanochat/gpt.py) decides: each group of parameters has `kind='muon'` (matrices only) or
  `kind='adamw'`. Example, AdamW for all parameters: in `setup_optimizer`, replace the loop after the comment
  `# Muon groups` by
  ```python
  param_groups.append(dict(kind='adamw', params=matrix_params, lr=matrix_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay))
  ```
  and pretrain with `--matrix-lr 6e-4`: Muon's 0.02 is far too large for AdamW (6e-4 is what GPT-3 Small, the same
  width as d12, used). SFT inherits these learning rates.
- **Architecture**: edit [`nanochat/gpt.py`](nanochat/gpt.py) (e.g. `class MLP`). Keep its FLOP estimates exact and
  check them with your architecture flags: `python -m scripts.check_flops --depth 12` (§6).
- **Data loading** (packing, order of the documents within a run): [`nanochat/dataloader.py`](nanochat/dataloader.py),
  [`nanochat/dataset.py`](nanochat/dataset.py).
- **Your own training script**: allowed, if it records its FLOPs in the ledger (§7).

### 5.4 Supervised fine-tuning: `scripts/chat_sft.py`

```bash
python -m scripts.chat_sft --model-tag d12 --mixture "gsm8k:4,mbpp:8,tinygsm:1,smoltalk:1" --save-every 25
```

* Conversations are rendered with nanochat's chat template (`<|user_start|>…<|assistant_start|>…`). The loss is
  only on assistant tokens. The math answer convention is `#### <number>` at the end, as in GSM8K. Assistant messages
  can be a list of parts, including calculator calls (see `tasks/gsm8k.py`).
* `--val-every N` scores the model every N steps on a random subset of the validation suite (`--val-max-problems`,
  greedy decoding). The default `0` scores only at the end and `-1` never.

**What you can change** (`python -m scripts.chat_sft --help` lists every flag):
- **Mixture**: `--mixture "gsm8k:4,mbpp:8,tinygsm:0.1,metamathqa:0.1"`, `source:epochs` over the SFT sources of your
  data config (§5.1): `4` repeats a source four times, `0.1` takes the first 10% of it (shuffled).
- **Length**: SFT makes one pass over the mixture. To train longer, repeat sources in the mixture (e.g. `gsm8k:8`
  instead of `gsm8k:4`). `--num-iterations 50` stops earlier: it can shorten SFT, not lengthen it.
- **Batch size and sequence length**: `--batch-size 256` (sequences per step) and `--max-seq-len 512` (tokens per
  sequence, at most the model's context); both default to pretraining's.
- **Learning rates**: by default, pretraining's learning rates × `--init-lr-frac 0.8`, constant, then decaying to 0
  over the second half (`--warmdown-ratio 0.5`, `--final-lr-frac 0`). Also `--matrix-lr`, `--embedding-lr`,
  `--unembedding-lr`, `--warmup-ratio`.
- **Optimizer**: the one of pretraining (the same `setup_optimizer`, §5.3). It continues from the pretraining
  optimizer's state; `--load-optimizer 0` starts it fresh.
- **Checkpoints to choose from**: `--save-every 25`; then select the best one with `scripts/evaluate.py` (§4.3).

### 5.5 Reinforcement learning (optional, fixed): `scripts/chat_rl.py`

RL is a **fixed pipeline** ([`scripts/chat_rl.py`](scripts/chat_rl.py), [`nanochat/rl.py`](nanochat/rl.py), do not
modify). You need no RL background: you choose only **how many tokens of math and of code** it trains on.

```bash
python -m scripts.prepare_data --rl                                              # once: the fixed RL problems
python -m scripts.chat_rl --model-tag d12 --math-tokens 1e7 --code-tokens 1e7    # starts from sft:d12, saves rl:d12
python -m scripts.evaluate --model rl:d12
```

What it does, for reference:
- **Fixed problems with a checkable answer**, from three post-training datasets of the catalog, built by the standard
  pipeline (`prepare_data --rl`, into `data/rl/`) independently of your data config, so your SFT filters and
  `max_docs` do not change them. The pipeline also drops the problems listed in `nanochat/rl_exclude.json`: they
  overlap the public datasets that the evaluation suites are drawn from. So every team, and the graders, train on the
  same problems. Math: GSM8K train (7.5K problems); the reward is 1 if the final number is right.
  Code: half TinyGSM (50K problems; reward 1 if the program's `solution()` returns what the reference program
  returns), half MBPP (474 problems; reward 1 if the program passes the test shown in the prompt).
- **The algorithm**: for each problem, 16 answers are sampled at temperature 1.0 (at most 256 tokens each). Answers
  that score above the problem's average are made more likely, the others less likely (policy gradient with a
  group-mean baseline, a simplified GRPO, no KL term). 16 problems per step, learning rate decaying linearly to 0.
  The optimizer is fixed too: the baseline's Muon + AdamW, whatever you change in yours (§5.3).
- **RL tokens** are all tokens of all sampled answers, prompts included. Each step takes problems from the domain
  whose budget is used least, so math and code budgets are used up together.
- **Compute**: the rollouts (generation) and the policy-gradient passes go into the ledger as C_RL. It costs about
  1.35 × the training FLOPs of the same number of tokens (measured: 2.1e16 FLOPs, 0.7% of B, for 2e7 tokens on a d12,
  about an hour on one A100): plan it with `plan_budget --rl-math-tokens --rl-code-tokens`.

**What you can change**: only how many tokens, and which checkpoint RL starts from.
- **Tokens**: `--math-tokens 1e7 --code-tokens 0` (0 = no code). Plan them with
  `plan_budget --rl-math-tokens 1e7 --rl-code-tokens 0`.
- **Starting checkpoint**: `--model-tag` and `--model-step`, e.g. your best SFT checkpoint. `--source rl` continues
  from an RL checkpoint.

Everything else is the same for every team: the problems, rewards, sampling, learning rate, and the optimizer.

### 5.6 Inference (fixed): the official solver

Every model is evaluated the same way, by the official solver [`solvers/default.py`](solvers/default.py): one chat
prompt per problem and one answer, generated greedily (temperature 0, at most 512 tokens, calculator tool use
included). The configuration is built into [`scripts/evaluate.py`](scripts/evaluate.py) (`OFFICIAL_CONFIG`). A
submission contains one model and no other: no verifiers or other auxiliary models. Inference is not charged.

**Nothing can be changed here**: not the solver, its configuration (one greedy sample), the prompts, or the generation
code ([`nanochat/engine.py`](nanochat/engine.py)). They are fixed for every team (§7). What you control is what the
model learned in training.

### 5.7 Evaluation

See [section 4](#4-evaluation-step-by-step-and-how-we-grade).

---

## 6. Compute accounting

All FLOPs are computed analytically from the model's `estimate_flops()` (training: about 6·N per token plus
attention), `estimate_prefill_flops()` and `estimate_decode_flops()` (inference: about 2·N per token plus
attention; used for RL rollouts, and reported for evaluation runs for information), defined in
[`nanochat/gpt.py`](nanochat/gpt.py). The formulas that use them are in
[`nanochat/compute.py`](nanochat/compute.py).

**Reference costs** (context 1024, vocab 32K, nanochat architecture):

| depth | scaling params | train FLOPs / token | tokens for the *whole* B | tokens/param for the whole B |
|---|---|---|---|---|
| d8 | 42M | 3.0e8 | 9.9B | 236 |
| d10 | 70M | 5.0e8 | 6.0B | 86 |
| d12 | 110M | 7.7e8 | 3.9B | 35 |
| d14 | 164M | 1.1e9 | 2.6B | 16 |
| d16 | 235M | 1.6e9 | 1.9B | 7.9 |
| d20 | 435M | 2.9e9 | 1.0B | 2.4 |

A d12 trained at 12 tokens/param uses 34% of B, and a d14 75%.

**What counts toward the budget:** every training step in the lineage of your submitted model: pretraining
(C_pretrain), SFT (C_SFT), RL rollouts and updates (C_RL). **What is declared but not counted:** model calls used to
select your data (C_data). Inference at test time is fixed (§5.6) and not charged either.

**What is free:** inference, downloading the catalog datasets, CPU work (tokenizer training, rule-based filtering, hashing/dedup, code execution), evaluation on the validation suite, and exploratory
runs that do not lead to the submitted weights. Exploratory runs still cost your team's GPU hours, which are limited.
Plan small-scale experiments first: a d6 or d8 run costs 1–10% of a d12 run.

**If you change the architecture**, keep the three `estimate_*` methods exact, then run
```bash
python -m scripts.check_flops --depth 4 --max-seq-len 256
```
This compares them with FLOPs measured by `torch.utils.flop_counter`. The graders run it on your code, and the
estimates must be within 5%.

### 6.1 Plan your budget before you train

[`scripts/plan_budget.py`](scripts/plan_budget.py) computes C_total for an experiment design *before* you spend
GPU time on it. It builds your model on PyTorch's
`meta` device (no GPU, no memory) and calls the same FLOP formulas as the training scripts. The default design is
the baseline, and the script reproduces its measured cost (1.057e18, 35.2% of B):

```bash
python -m scripts.plan_budget                                    # the baseline: d12, 12 tokens/param, 85 SFT steps
python -m scripts.plan_budget --depth 16 --tokens-per-param 8
python -m scripts.plan_budget --depth 14 --fill                  # pretrain on whatever budget is left
python -m scripts.plan_budget --sweep                            # model size vs. pretraining tokens that fit
python -m scripts.plan_budget --plan configs/plans/example.json  # a design with a data source
python -m scripts.plan_budget --depth 14 --print-plan > my_plan.json   # turn flags into a plan file you can edit
```

It prints every term with its share of B (C_pretrain, C_SFT, C_RL; C_data separately), how many more pretraining tokens the rest of
the budget buys, and an A100 time estimate. Flags: model (`--depth`, `--aspect-ratio`, `--max-seq-len`, … as in
`base_train`), training (`--tokens-per-param` or `--pretrain-tokens`, `--sft-steps`, `--sft-batch-size`,
`--rl-math-tokens`, `--rl-code-tokens`) and data (`--data-params` and `--data-tokens`, giving 2 × params × tokens, or
`--data-flops`).

**From plan to experiment runner.** Every report ends with the flags that give each stage its planned compute:
`--num-iterations` and `--batch-size` (sequences). Copy them into your experiment runner (§9) and add the rest
yourself (`--model-tag`, `--mixture`, …). For the default plan (the baseline):

```
Flags for your experiment runner (they fix the compute of each stage; add --model-tag, --mixture, ... yourself):
  policy pretraining   python -m scripts.base_train --depth 12 --num-iterations 2520 --batch-size 512
                         # 2520 steps x 512 x 1024 tokens = 1.32e+09 tokens (12.0 tokens/param)
  policy sft           python -m scripts.chat_sft --num-iterations 85 --batch-size 512
                         # 512 sequences of 1024 tokens per step, padding included; SFT stops earlier if the mixture runs out
```

RL gets `chat_rl --math-tokens … --code-tokens …`, and model-based data selection gets its `data_flops` value for the
data config. Pick the design with `--sweep` or `--fill` first, then copy the flags. After training,
the ledger that the scripts print should match the plan.

---

## 7. What you may not change

Everything else may be changed: §5 lists, stage by stage, what you can change and how. These files are fixed (we
grade with our own copy of them):

| Fixed file | Why |
|---|---|
| `nanochat/compute.py` | budget constants and FLOP formulas |
| `nanochat/inference.py` | `MeteredLM`, the metered model access of the official solver |
| `nanochat/execution.py` | the code sandbox |
| `nanochat/datapipe.py` | the standard data pipeline (dedup, decontamination, manifests) |
| `scripts/prepare_data.py`, `tasks/{gsm8k,mbpp,metamath,tinygsm,smoltalk,magicoder}.py` | the dataset catalog and its formatters |
| `tasks/eval_suite.py` | prompts, answer extraction, scoring |
| `solvers/default.py`, `nanochat/engine.py` | the official solver and its generation code (inference is fixed, §5.6) |
| `scripts/chat_rl.py`, `nanochat/rl.py`, `nanochat/rl_exclude.json` | the fixed RL stage (you choose its math and code tokens) |
| `nanochat/optim.py` | Muon and AdamW, the only optimizers you may use (§5.3) |
| `nanochat/tokenizer.py`, `scripts/tok_train.py` | the tokenizer, the same for every team (§5.2) |
| `scripts/evaluate.py` | the official evaluation, with the official inference configuration |
| `data/val/` | the validation suite |

**Every training script must record its compute in the checkpoint ledger** (`ledger_add_stage` in
`nanochat/compute.py`), as the provided ones do. If you write a new training loop, add its FLOPs with
`train_flops(...)` / `generation_flops(...)`. A checkpoint with a missing or wrong ledger is treated as over budget.
Likewise, **all training data must be built by the standard pipeline** (§5.1), and its fingerprints are recorded in
the ledger.

---

## 8. Rules on data, models and evaluation

Summary. The PDF handout has the binding version.

1. **From scratch.** All weights start from random initialization. No pretrained weights, and no distillation from
   pretrained models. Every team uses the same tokenizer (§5.2).
2. **Only the dataset catalog** (§5.1). Every token you train on (pretraining, SFT, RL problems, tokenizer
   training) comes from the catalog datasets, built with the standard pipeline, which records it in your data
   manifest. Pretraining datasets are only for pretraining (and the tokenizer), post-training datasets only for SFT;
   the RL problems are fixed. No other datasets and no other downloads. You may select, filter, deduplicate, repeat, reweight and mix
   catalog data in any way you like.
3. **No synthetic data of your own.** You may not create new training text, by a program or by a model: no generated
   problems (e.g. templated arithmetic), no text written, rewritten, paraphrased, translated or augmented by a model,
   and no outputs of your own models saved as training data (distillation, rejection-sampling fine-tuning,
   self-training). The RL stage (`chat_rl.py`) is allowed: its rollouts are training compute, not data. Several
   catalog datasets were written by LLMs (TinyGSM, MetaMathQA, …), and using them is fine.
4. **No test data, anywhere in training.** These are forbidden in pretraining, SFT and RL, and as few-shot examples
   in prompts: the EleutherAI/GPT-3 `arithmetic` datasets; MAWPS and everything in it (AddSub, SingleOp, SingleEq,
   MultiArith, SimulEq, e.g. `MU-NLPC/Calc-mawps`); ASDiv and SVAMP (they overlap with MAWPS); the GSM8K test split; the MBPP test split
   (task ids 11–510) and MBPP+; HumanEval(+); GSM-Symbolic; collections that contain them (e.g. Līla, Calc-X);
   and anything derived from them. None of them is in the catalog. The **validation suite may be used for evaluation, checkpoint selection and hyperparameter tuning, but not for
   training.** The pipeline removes everything that overlaps the validation suite automatically, and we repeat
   the check against the hidden suite.
5. **Selecting data with models** is allowed: your filters may score catalog documents with a model (your own, or an
   open-weight model with a known parameter count) to decide what to keep. Declare `2 × active_params ×
   tokens_scored` as `data_flops` of that source in your data config (declared, not counted in B). Closed models behind APIs (GPT-x, Claude,
   Gemini, …) may not be used.
6. **Inference** is fixed: the official solver in its official configuration (one greedy answer of at most 512
   tokens per problem, §5.6), and one model: no verifiers or other auxiliary models, no network access. The
   hidden-suite evaluation of your submission must finish within **2 hours on one A100** (the baseline takes 1 min on
   the validation suite, which is about half the size).
7. **Budget.** C_total = C_pretrain + C_SFT + C_RL ≤ B = 3e18 FLOPs (about 5 A100-hours of training). Up to 5% over
   is tolerated to absorb measurement noise. C_data must be declared, but is not counted.

---

## 9. Making a submission

```bash
python -m scripts.make_submission --model sft:d12_final --runner runs/experiment.sh --out submission/
python -m scripts.evaluate --submission submission/        # exactly what the graders run (on the hidden suite)
```

The submission directory holds:
- `policy/`: **one** model checkpoint, with its compute ledger and the tokenizer (no verifier or other models);
- `data_manifest.json`, generated from the manifests of every data source in the lineage of the weights;
- `data_card.json`, whose source list and C_data are filled in automatically: **add a `how_used` note to every
  source and a `notes` field**;
- `experiment.sh`: a copy of your **experiment runner** (below);
- `submission.json`.

**The experiment runner** is one bash script in your repository, `runs/experiment.sh`, that rebuilds your submitted
model from scratch: it runs your commands of §5.1–5.6 (data, tokenizer, pretraining, SFT, RL, evaluation), in order,
with their exact arguments, and then packages the model with `make_submission`. For the example recipe of the
handout (the example data config and mixture, a d12, SFT and RL), it is:

```bash
#!/bin/bash
# runs/experiment.sh: rebuilds our submitted model from scratch. Run: NPROC=4 bash runs/experiment.sh
set -e                                             # stop at the first error
uv sync --extra gpu && source .venv/bin/activate
NPROC="${NPROC:-1}"                                # number of GPUs
if [ "$NPROC" -gt 1 ]; then LAUNCH="torchrun --standalone --nproc_per_node=$NPROC -m"; SEP="--"
else LAUNCH="python -m"; SEP=""; fi

# Step 1: data
python -m scripts.prepare_data --config configs/data/example_math_code.json
python -m scripts.prepare_data --rl
# Step 2: tokenizer (fixed)
python -m scripts.tok_train
# Step 3: pretraining
$LAUNCH scripts.base_train $SEP --depth 12 --mixture configs/mixtures/example_math_code.json \
    --num-iterations 2520 --model-tag d12_mc
# Step 4: SFT
$LAUNCH scripts.chat_sft $SEP --model-tag d12_mc \
    --mixture "gsm8k:4,mbpp:8,tinygsm:0.1,metamathqa:0.1" --save-every 25
# Step 5: RL
$LAUNCH scripts.chat_rl $SEP --model-tag d12_mc --math-tokens 1e7 --code-tokens 1e7
# Step 6: evaluation on the validation suite
$LAUNCH scripts.evaluate $SEP --model rl:d12_mc
# Step 7: packaging
python -m scripts.make_submission --model rl:d12_mc --runner runs/experiment.sh --out submission/
```

If you chose a checkpoint on the validation suite, package that one (e.g. `--model sft:d12_mc:50` for step 50 of
SFT). [`runs/baseline.sh`](runs/baseline.sh) is the same kind of script, for the baseline. Before you submit, run your
runner once from the repository root on a fresh `$NANOCHAT_BASE_DIR` (`NPROC=4 bash runs/experiment.sh`).

It must run end to end without manual steps, take the number of GPUs from `NPROC`, and use only files in your
repository. We may re-run it and compare the compute ledger, the data fingerprints and the validation score with
your submission.

Submit the directory together with the git commit of your code. Every number in your report must be reproducible
from that commit.

Checklist:
- [ ] `python -m scripts.evaluate --submission submission/` runs and reports `OK, within budget`
- [ ] `python -m scripts.check_flops` passes on your architecture
- [ ] `make_submission` printed no data warnings (all sources found, fingerprints match), and the data card notes are filled in
- [ ] the experiment runner rebuilds the submitted model from scratch (try it at small scale, e.g. with a smaller `--depth`)

---

## 10. Repository layout

```
├── README.md
├── configs/
│   ├── data/*.json                      # data configs: which sources to build (baseline, example, smoke)
│   ├── plans/example.json               # an experiment design for scripts/plan_budget.py
│   └── mixtures/*.json                  # pretraining data mixtures (token proportions of built sources)
├── data/val/                            # the validation suite (math_val.jsonl, code_val.jsonl)  [do not modify]
├── nanochat/
│   ├── compute.py                       # budget + FLOP accounting + ledger                   [do not modify]
│   ├── inference.py                     # MeteredLM, metered model access of the solver       [do not modify]
│   ├── execution.py                     # sandboxed Python execution                          [do not modify]
│   ├── datapipe.py                      # the standard data pipeline + manifests              [do not modify]
│   ├── gpt.py                           # the Transformer (+ FLOP estimates)
│   ├── optim.py                         # Muon + AdamW, the optimizers you may use            [do not modify]
│   ├── dataset.py, dataloader.py        # data sources, mixtures, packing
│   ├── tokenizer.py                     # BPE tokenizer + chat template (fixed)               [do not modify]
│   ├── engine.py                        # KV-cache inference with calculator tool            [do not modify]
│   ├── rl.py, rl_exclude.json           # the fixed RL stage: problems, rewards, algorithm    [do not modify]
│   └── checkpoint_manager.py, common.py, loss_eval.py, flash_attention.py, fp8.py
├── runs/
│   ├── baseline.sh                      # the baseline (GPU), an example of an experiment runner
│   └── smoke_cpu.sh                     # end-to-end smoke test (laptop or GPU, ~10 min)
├── scripts/
│   ├── prepare_data.py                  # the dataset catalog; builds data with the pipeline     [do not modify]
│   ├── tok_train.py                     # trains the fixed tokenizer (no options)              [do not modify]
│   ├── base_train.py                    # (continued) pretraining
│   ├── chat_sft.py                      # supervised fine-tuning
│   ├── chat_rl.py                       # the fixed RL stage (--math-tokens, --code-tokens)    [do not modify]
│   ├── evaluate.py                      # official evaluation + budget report                [do not modify]
│   ├── make_submission.py               # package a submission
│   ├── check_flops.py                   # verify FLOP estimates of your architecture
│   ├── plan_budget.py                   # plan C_total of an experiment design before running it
│   ├── decontaminate.py                 # n-gram overlap check against the evaluation suite
│   └── chat_cli.py                      # chat with a model in the terminal
├── solvers/default.py                   # the official solver (one greedy answer)            [do not modify]
├── tasks/
│   ├── eval_suite.py                    # prompts, answer extraction, scoring                [do not modify]
│   ├── registry.py                      # SFT mixtures
│   ├── gsm8k.py, mbpp.py, tinygsm.py, metamath.py, magicoder.py, smoltalk.py   # catalog SFT data  [do not modify]
│   └── customjson.py, common.py
```

---

## 11. Hardware notes and troubleshooting

* **Precision**: bf16 on A100/H100 (auto-detected), fp32 on older GPUs and CPU. Force it with `NANOCHAT_DTYPE`.
  FLOPs do not depend on precision, but wall-clock time does.
* **Attention**: Flash Attention 3 is used automatically on Hopper (H100) and Ampere when the `kernels` package can
  load it. Otherwise PyTorch SDPA is used. Sliding-window patterns (`--window-pattern SSSL`) only pay off with FA3.
  Keep the default `L` elsewhere.
* **Out of memory**: lower `--device-batch-size` (32 → 16 → 8 …). Gradient accumulation keeps the effective batch
  (`--batch-size` sequences per step) and therefore the results unchanged.
* **Little RAM** (e.g. a notebook server): the data is streamed from disk, so RAM use does not grow with the data.
  `prepare_data` needs about 2 GB with `--num-workers 4`. Each worker is a separate process, so use fewer workers
  if RAM is tight.
* **Multi-GPU**: `torchrun --standalone --nproc_per_node=<n> -m scripts.<script> -- <args>`. FLOPs are identical,
  only faster. The optimizer state is saved per GPU, so `chat_sft` (which warm-starts from the pretraining
  optimizer) must run on the same number of GPUs as the pretraining run, or with `--load-optimizer 0`.
* **Evaluation speed**: the official solver generates for many problems at once (`--batch-size`, default 256),
  which is 6–20× faster than one problem at a time. You can also shard over GPUs with `torchrun`.
* **Reproducibility of scores**: on a GPU, results differ slightly with the batch size and the number of GPUs.
  bf16 numerics depend on the batch shape, which occasionally flips a near-tie; we measured ±0.3 points on the
  baseline. Compare your runs at the same settings. The graders use the defaults.
* **Code execution** runs in a separate Python process with time and memory limits. It is not a security sandbox,
  so run it on your own machine or allocation only.

---

## 12. Ideas to get you started

* **Where is the compute-optimal model for *this* objective?** Chinchilla-style iso-FLOP curves on validation
  *accuracy* rather than loss. Downstream accuracy may favor over-training a smaller model.
* **Data beats parameters at this scale.** Domain mixtures (math web text, math problems, Python code) and the
  share of high-quality data in the mixture.
* **Data selection**: heuristic (`language_score`, length, URL), perplexity or classifier filtering of FineWeb, near-duplicate
  removal, and mixture optimization from small proxy runs. How often can a small high-quality source be repeated
  before it stops helping?
* **Teach the format early.** Mixing instruction-formatted data into pretraining, tool use (the calculator), and
  program-of-thought solutions executed in the sandbox.
* **Architecture**: context length, depth vs. width, attention and MLP variants.
* **Transfer**: which catalog data helps `arithmetic` without any arithmetic problems in the catalog? Which skills
  transfer between the math and the code subsets?

Relevant reading: Chinchilla (Hoffmann et al. 2022), Beyond Chinchilla-Optimal (Sardana et al. 2024), TinyGSM (Liu
et al. 2023), Textbooks Are All You Need (Gunasekar et al. 2023), DoReMi (Xie et al. 2023), and the nanochat
discussions on GitHub.

---

*Acknowledgements: the code is adapted from [nanochat](https://github.com/karpathy/nanochat) (MIT license, © Andrej
Karpathy). Evaluation data comes from the GPT-3 arithmetic tasks, GSM8K, MAWPS and MBPP/MBPP+. Please
cite them if you use them in your report.*
