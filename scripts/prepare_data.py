"""
Build training data with the standard pipeline (nanochat/datapipe.py), from the DATASET CATALOG below. Every source,
pretraining text or SFT conversations, goes through the same stages (format, clean, select, dedup, decontamination,
split, manifest).

!!! DO NOT MODIFY THIS FILE: it defines the catalog of allowed datasets. Graders use our own copy. !!!

    python -m scripts.prepare_data --list                                # the catalog (and what you have built)
    python -m scripts.prepare_data --config configs/data/baseline.json  # build everything a config declares
    python -m scripts.prepare_data --config configs/data/baseline.json --only fineweb,gsm8k
    python -m scripts.prepare_data --rl                                  # the RL problems (fixed, see nanochat/rl.py)

The catalog has two disjoint parts: PRETRAINING datasets (large text corpora, read as documents) and POST-TRAINING
datasets (problems with solutions and chat, read as conversations by SFT). A dataset is in exactly one of them.
The RL problems are not yours to choose: they are a fixed selection of three post-training datasets.

A data config declares the sources to build. Every source is a dataset of the catalog, under a name of your choice
(the same dataset can be built several times, e.g. with different filters):
{
  "pretrain": {
    "fineweb":      {"registry": "fineweb", "num_shards": 4},
    "finemath":     {"registry": "finemath", "num_shards": 4, "max_docs": 500000},
    "web_en":       {"registry": "fineweb", "num_shards": 4, "filter": "my_filters:confident_english"}
  },
  "sft": {
    "gsm8k":        {"registry": "gsm8k"},
    "tinygsm":      {"registry": "tinygsm", "max_docs": 21000}
  }
}
Optional keys for every source: num_shards (pretraining: how many parquet files to download), max_docs,
val_docs (pretraining) / val_fraction (SFT), seed, filter, data_flops, notes.
  filter      "module:function", called as function(document, record) -> bool: keep the document or not. `record`
              is the raw row (e.g. FineWeb's "language_score"). A filter only SELECTS: it cannot change documents.
              Filters run in order in one process, so they may keep state (e.g. an index for near-dedup).
  data_flops  FLOPs spent by models (e.g. a quality classifier) to score or select this data: counted in C_data.
The names are what you use in mixtures: --mixture "fineweb:0.8,finemath:0.2" (pretraining),
--mixture "gsm8k:4,tinygsm:1" (SFT). Build outputs and manifests go to $NANOCHAT_BASE_DIR/data/.

RULES (see the project description): only the datasets of this catalog may be used, and you may not create
synthetic data of your own (no generated, rewritten or augmented text, by a program or by a model). Selecting,
filtering, deduplicating, repeating and mixing catalog data is up to you. Decontamination against the validation
suite is automatic; graders repeat it against the hidden suite.
"""

import os
import json
import time
import argparse
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq

from nanochat.common import get_base_dir, hf_request
from nanochat.datapipe import build_source, read_manifest

# -----------------------------------------------------------------------------
# Formatting functions: one row of the original dataset -> one document (string), or None to drop it

def fmt_text(col):
    return lambda r: r[col]

# -----------------------------------------------------------------------------
# The catalog of PRETRAINING datasets: large text corpora, read as documents. kind="hf": rows of a HuggingFace dataset
# (via its parquet export), formatted with `fmt`; `columns` are the fields a filter sees in `record`. "LLM-generated"
# datasets were written by models and published by others before the project: you may use them, but not create such
# data yourself.

SOURCES = {
    # --- general web text ----------------------------------------------------
    "fineweb": dict(kind="hf", repo="HuggingFaceFW/fineweb", config="sample-10BT", split="train",
                    columns=["text", "url", "language_score", "token_count"], fmt=fmt_text("text"),
                    desc="FineWeb, 10BT sample: general web text (lightly filtered Common Crawl, not deduplicated across dumps; 15 files of ~0.7B tokens)."),
    # --- math ------------------------------------------------------------------
    "finemath": dict(kind="hf", repo="HuggingFaceTB/finemath", config="finemath-4plus", split="train", columns=["text"], fmt=fmt_text("text"),
                     desc="FineMath-4+: educational math web pages (~10B tokens, 64 files)."),
    # --- code ------------------------------------------------------------------
    "codeparrot": dict(kind="hf", repo="codeparrot/codeparrot-clean", config="default", split="train", columns=["content"], fmt=fmt_text("content"),
                       desc="CodeParrot-clean: deduplicated Python files from GitHub (partial parquet export, 10 files)."),
}

# -----------------------------------------------------------------------------
# The catalog of POST-TRAINING datasets: name -> (description, function returning the conversations ({"messages": [...]}))

def _task_conversations(make_task):
    def read():
        task = make_task()
        for i in range(len(task)):
            yield task[i]
    return read

def _sft_registry():
    from tasks.gsm8k import GSM8K
    from tasks.mbpp import MBPP
    from tasks.smoltalk import SmolTalk
    from tasks.metamath import MetaMathQA
    from tasks.tinygsm import TinyGSM
    from tasks.magicoder import Magicoder
    return {
        "gsm8k": ("GSM8K train (7.5K problems, calculator tool calls, answer after ####)", lambda: GSM8K(subset="main", split="train")),
        "mbpp": ("MBPP train/prompt/validation splits (474 problems, never the test split)", lambda: MBPP(splits=("train", "prompt", "validation"))),
        "tinygsm": ("TinyGSM, first shard (~700K problems with Python solutions, python_math format; LLM-generated)", lambda: TinyGSM()),
        "metamathqa": ("MetaMathQA, GSM8K-derived subset (~240K; LLM-generated)", lambda: MetaMathQA(gsm_only=True)),
        "magicoder": ("Magicoder OSS-Instruct, Python subset (~38K problems with solutions; LLM-generated)", lambda: Magicoder()),
        "smoltalk": ("smol-smoltalk general chat (460K conversations; LLM-generated)", lambda: SmolTalk(split="train")),
    }

# -----------------------------------------------------------------------------
# Downloading

def hf_headers():
    # authenticated requests (export HF_TOKEN=...) get much higher rate limits than anonymous ones
    return {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}


def download(url, path, max_attempts=8):
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=60, headers=hf_headers()) as response:
                response.raise_for_status()
                with open(path + ".tmp", "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            os.rename(path + ".tmp", path)
            print(f"Downloaded {os.path.basename(path)}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {url}: {e}")
            if os.path.exists(path + ".tmp"):
                os.remove(path + ".tmp")
            time.sleep(min(2 ** attempt, 120))
    return False


def download_all(urls, raw_dir, num_workers):
    paths = [os.path.join(raw_dir, f"{i:05d}.parquet") for i in range(len(urls))]
    with Pool(processes=num_workers) as pool:
        ok = pool.starmap(download, list(zip(urls, paths)))
    assert all(ok), "Some downloads failed, rerun to retry"
    return paths

# -----------------------------------------------------------------------------
# Turning a source spec into (parts, to_doc) for the pipeline

def _parquet_reader(path, columns):
    """The rows of a raw parquet file, streamed a batch at a time (never the whole file in memory)."""
    def read():
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
            yield from batch.to_pylist()
    return read


def pretrain_parts(name, spec, num_workers):
    reg = SOURCES[spec["registry"]]
    num_shards = int(spec.get("num_shards", -1))
    urls = json.loads(hf_request(f"https://huggingface.co/api/datasets/{reg['repo']}/parquet/{reg['config']}/{reg['split']}"))
    urls = urls[:num_shards] if num_shards > 0 else urls
    raw_dir = os.path.join(get_base_dir(), "raw", reg["repo"].replace("/", "--"), reg["config"], reg["split"])
    paths = download_all(urls, raw_dir, num_workers)
    parts = [({"url": url, "sha256": None}, _parquet_reader(path, reg["columns"])) for url, path in zip(urls, paths)]
    return parts, reg["fmt"]


def sft_parts(name, spec):
    desc, make_task = _sft_registry()[spec["registry"]]
    return [({"registry": spec["registry"], "desc": desc}, _task_conversations(make_task))], (lambda r: {"messages": r["messages"]})


def rl_problem_sources():
    from nanochat.rl import PROBLEM_SOURCES
    return PROBLEM_SOURCES


def rl_excluded_problems(name):
    from nanochat.rl import excluded_problems
    return excluded_problems(name)


def check_catalog(kind, name, spec):
    """Every source must be a dataset of the catalog: no local files, no data of your own. RL problems are fixed."""
    if kind == "rl":
        assert spec == rl_problem_sources().get(name), f"The RL problems are fixed (nanochat/rl.py), got rl source '{name}': {spec}"
        return
    catalog = SOURCES if kind == "pretrain" else _sft_registry()
    assert spec.get("registry") in catalog, (
        f"{kind} source '{name}': only {kind} datasets of the catalog may be used (python -m scripts.prepare_data --list); "
        f"got {spec}. Creating synthetic data of your own is not allowed.")
    assert not set(spec) & {"files", "text_field"}, f"{kind} source '{name}': local files are not allowed: {spec}"


def build_config(config, only=None, suites=("data/val",), num_workers=8, force=False):
    assert not set(config) - {"pretrain", "sft", "rl", "comment"}, f"Unknown sections in the data config: {sorted(config)}"
    manifests = {}
    for kind in ["pretrain", "sft", "rl"]:
        for name, spec in config.get(kind, {}).items():
            if only and name not in only:
                continue
            check_catalog(kind, name, spec)
            print(f"=== {kind} source '{name}': {spec}")
            if kind == "pretrain":
                parts, to_doc = pretrain_parts(name, spec, num_workers)
            else:
                parts, to_doc = sft_parts(name, spec)
            exclude = rl_excluded_problems(name) if kind == "rl" else frozenset() # the RL problems are the same for everyone
            manifests[(kind, name)] = build_source(name, kind, spec, parts, to_doc, list(suites),
                                                   num_workers=num_workers, force=force, exclude=exclude)
    return manifests


def describe():
    print("The dataset catalog (the only allowed training data). [built] = a source of that name exists in your base dir.")
    for name, spec in SOURCES.items():
        built = "built" if read_manifest(name, "pretrain") else "-"
        print(f"  pretrain {name:>18s} [{built:>5s}] {spec['repo']}: {spec['desc']} Filter fields: {', '.join(spec['columns'])}")
    for name, (desc, _) in _sft_registry().items():
        built = "built" if read_manifest(name, "sft") else "-"
        print(f"  sft      {name:>18s} [{built:>5s}]: {desc}")
    for name, spec in rl_problem_sources().items():
        built = "built" if read_manifest(name, "rl") else "-"
        print(f"  rl       {name:>18s} [{built:>5s}]: fixed RL problems from the post-training dataset '{spec['registry']}' {spec} (--rl)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training data with the standard pipeline")
    parser.add_argument("--config", type=str, default=None, help="data config (json), see configs/data/")
    parser.add_argument("--only", type=str, default=None, help="comma-separated source names to build (default: all in the config)")
    parser.add_argument("--list", action="store_true", help="list the dataset catalog")
    parser.add_argument("--rl", action="store_true", help="build the fixed RL problems (needed by scripts/chat_rl.py)")
    parser.add_argument("--num-workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)), help="parallel downloads / decontamination workers")
    parser.add_argument("--force", action="store_true", help="rebuild even if the manifest is up to date")
    parser.add_argument("--extra-suite", type=str, action="append", default=[], help=argparse.SUPPRESS) # staff: also decontaminate against these
    args = parser.parse_args()
    if args.list or (args.config is None and not args.rl):
        describe()
        raise SystemExit(0)
    config = {}
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        assert "rl" not in config, "The RL problems are fixed: build them with --rl, not in a data config"
    if args.rl:
        config["rl"] = rl_problem_sources()
    manifests = build_config(config, only=set(args.only.split(",")) if args.only else None,
                             suites=["data/val"] + args.extra_suite, num_workers=args.num_workers, force=args.force)
    print("\nSummary:")
    for (kind, name), m in manifests.items():
        c = m["counts"]
        print(f"  {kind:>8s} {name:>18s}: train {c['train']:,} | val {c['val']:,} | duplicates {c['duplicates']:,} | "
              f"contaminated {c['contaminated']:,} | fingerprint {m['fingerprint'][:12]}")
