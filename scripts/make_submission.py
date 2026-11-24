"""
Package a submission: ONE model (checkpoint + tokenizer) + data card + experiment runner. Every submission is
evaluated with the same, official inference configuration (scripts/evaluate.py: greedy, one sample).

python -m scripts.make_submission --model sft:d12 --runner runs/experiment.sh --out submission/

Then check it on the validation suite exactly like the graders will (score, budget):
python -m scripts.evaluate --submission submission/

Layout of the submission directory:
    submission.json        manifest (model directory, data card, data manifest, runner)
    policy/model_*.pt, meta_*.json, tokenizer/   your model (no verifier or other models: one model per submission)
    data_manifest.json     generated: the manifests of every data source in the lineage of the submitted weights
    data_card.json         your data card. C_data is computed from the manifests (the `data_flops` of each source);
                           you add the narrative (how and why each source was used)
    experiment.sh          copy of your experiment runner: the bash script that rebuilds this model from scratch
                           (data, tokenizer, every training stage, packaging), from the root of your repository
"""

import os
import glob
import json
import shutil
import argparse

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import checkpoint_dir_of, find_last_step
from nanochat.compute import ledger_from_meta, budget_report, format_budget_report
from nanochat.datapipe import read_manifest
from scripts.evaluate import train_flops_of

DATA_CARD_TEMPLATE = {
    "team": "<team name>",
    "sources": "<filled in automatically from the manifests: add a 'how_used' note to each>",
    "notes": "<anything the graders should know about your data: selection, filtering, dedup, repetition, mixtures, ...>",
}


def data_lineage(metas):
    """Collect the data fingerprints recorded in the ledgers of the submitted models -> {(kind, name): fingerprint}."""
    lineage = {}
    for meta in metas:
        for stage in ledger_from_meta(meta)["stages"]:
            for kind, prints in (stage.get("data") or {}).items():
                for name, fp in prints.items():
                    lineage[(kind, name)] = fp
    return lineage


def resolve_checkpoint(spec):
    """'sft:d12[:step]' or '<dir>[:step]' -> (checkpoint_dir, step)."""
    parts = spec.split(":")
    if parts[0] in ("base", "sft", "rl") and len(parts) >= 2:
        checkpoint_dir = checkpoint_dir_of(parts[0], parts[1])
        step = int(parts[2]) if len(parts) > 2 else find_last_step(checkpoint_dir)
    elif len(parts) > 1 and parts[-1].isdigit():
        checkpoint_dir, step = ":".join(parts[:-1]), int(parts[-1])
    else:
        checkpoint_dir, step = spec, find_last_step(spec)
    return checkpoint_dir, step


def check_runner(path):
    """The experiment runner: a bash script that rebuilds the submitted model from scratch."""
    assert os.path.isfile(path), f"Experiment runner not found: {path} (start from a copy of runs/baseline.sh)"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    assert text.startswith("#!") and "bash" in text.splitlines()[0], f"{path}: the experiment runner must be a bash script (#!/bin/bash)"
    for step in ["scripts.prepare_data", "scripts.base_train", "scripts.make_submission"]:
        if step not in text:
            print(f"WARNING runner: {path} does not call {step}. It must rebuild the model from scratch: data, tokenizer, "
                  f"every training stage, and packaging.")


def main():
    parser = argparse.ArgumentParser(description="Package a submission")
    parser.add_argument("--model", type=str, required=True, help="model spec: sft:d12[:step], rl:d12, or a checkpoint directory")
    parser.add_argument("--runner", type=str, required=True, help="your experiment runner (bash), e.g. runs/experiment.sh")
    parser.add_argument("--data-card", type=str, default=None, help="your data card json (a template is written if omitted)")
    parser.add_argument("--out", type=str, default="submission")
    args = parser.parse_args()

    check_runner(args.runner)
    os.makedirs(args.out, exist_ok=True)
    manifest = {"models": {"policy": "policy"}, "data_card": "data_card.json", "data_manifest": "data_manifest.json",
                "runner": "experiment.sh"}
    checkpoint_dir, step = resolve_checkpoint(args.model)
    target = os.path.join(args.out, "policy")
    os.makedirs(target, exist_ok=True)
    model_files = [f"model_{step:06d}.pt", f"meta_{step:06d}.json"]
    for old in glob.glob(os.path.join(target, "model_*.pt")) + glob.glob(os.path.join(target, "meta_*.json")):
        if os.path.basename(old) not in model_files: # packaged before into the same --out: one model per submission
            print(f"Removing {old} (packaged before)")
            os.remove(old)
    for fname in model_files:
        shutil.copy2(os.path.join(checkpoint_dir, fname), os.path.join(target, fname))
    tokenizer_src = os.path.join(get_base_dir(), "tokenizer")
    tokenizer_dst = os.path.join(target, "tokenizer")
    if os.path.exists(tokenizer_dst):
        shutil.rmtree(tokenizer_dst)
    shutil.copytree(tokenizer_src, tokenizer_dst)
    with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    metas = [meta]
    ledger = ledger_from_meta(meta)
    print(f"[policy] {checkpoint_dir} step {step}: {len(ledger['stages'])} ledger stages, {ledger['train_flops']:.3e} training FLOPs")
    for stage in ledger["stages"]:
        print(f"    - {stage['stage']:>10s}: {stage['flops']:.3e} FLOPs ({stage.get('model_tag', '')}, step {stage.get('step', '?')})")
    if not ledger["stages"]:
        print("    WARNING: empty compute ledger! Checkpoints must be produced by the provided training scripts.")

    shutil.copy2(args.runner, os.path.join(args.out, "experiment.sh"))
    # data: the manifests of every source in the lineage (recorded in the compute ledgers by the training scripts)
    manifests, problems = [], []
    for (kind, name), fp in sorted(data_lineage(metas).items()):
        m = read_manifest(name, kind)
        if m is None:
            problems.append(f"{kind} source '{name}': manifest not found (rebuild it with scripts/prepare_data.py)")
        elif m["fingerprint"] != fp:
            problems.append(f"{kind} source '{name}': rebuilt since training (fingerprint {m['fingerprint'][:12]} != {fp[:12]})")
        if m is not None:
            manifests.append(m)
    with open(os.path.join(args.out, "data_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"sources": manifests, "problems": problems}, f, indent=2)
    for p in problems:
        print(f"WARNING data: {p}")
    data_flops = sum(m["data_flops"] for m in manifests)
    data_card_path = os.path.join(args.out, "data_card.json")
    if args.data_card:
        with open(args.data_card, "r", encoding="utf-8") as f:
            card = json.load(f)
    elif os.path.exists(data_card_path):
        with open(data_card_path, "r", encoding="utf-8") as f:
            card = json.load(f)
    else:
        card = dict(DATA_CARD_TEMPLATE)
        print(f"Wrote a data card TEMPLATE to {data_card_path}: fill in the notes!")
    old_sources = {s["name"]: s for s in card.get("sources", []) if isinstance(s, dict)} if isinstance(card.get("sources"), list) else {}
    card["sources"] = [{"name": m["name"], "kind": m["kind"], "spec": m["spec"], "train_docs": m["counts"]["train"],
                        "contaminated_removed": m["counts"]["contaminated"], "data_flops": m["data_flops"],
                        "how_used": old_sources.get(m["name"], {}).get("how_used", "<fill in>")} for m in manifests]
    card["data_flops"] = data_flops # C_data = sum of the declared data_flops of the sources
    with open(data_card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"Data: {len(manifests)} sources in the lineage, C_data = {data_flops:.3e} FLOPs (see data_manifest.json)")
    with open(os.path.join(args.out, "submission.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Budget (inference is fixed and not charged):")
    print(format_budget_report(budget_report(train_flops_of(metas), data_flops)))
    print(f"\nSubmission written to {args.out}. Next, dry-run it on the validation suite (score + budget):")
    print(f"    python -m scripts.evaluate --submission {args.out}")


if __name__ == "__main__":
    main()
