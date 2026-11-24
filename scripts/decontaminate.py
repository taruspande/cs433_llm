"""
Check training data for overlap with an evaluation suite (n-gram based).

Training on evaluation data (or near-copies of it) is forbidden. The graders run this same
check against the HIDDEN test suite on the data you declare in your data card, so check
your data against the released validation suite yourself, and remove what is flagged:

python -m scripts.decontaminate --sources fineweb,finemath --max-docs 200000
python -m scripts.decontaminate --jsonl my_synthetic_sft.jsonl
python -m scripts.decontaminate --sources tinygsm --write-clean   # writes data/tinygsm_clean/ without flagged docs

A document is flagged if it shares at least one word n-gram (default n=13, after lowercasing and
removing punctuation) with an evaluation problem. Benchmark questions are long enough for 13-grams
to be a very specific signal; short ones (e.g. MBPP descriptions) are matched with n=8, and very short
ones (e.g. "What is 98 plus 45?") must appear verbatim. The data pipeline (scripts/prepare_data.py)
applies this check to every source automatically; this script is for ad-hoc checks.
"""

import os
import re
import json
import argparse

import pyarrow.parquet as pq

from nanochat.dataset import list_parquet_files, write_parquet_docs, source_dir
from tasks.eval_suite import load_suite

WORD_RE = re.compile(r"[a-z0-9]+")

def ngrams(text, n):
    words = WORD_RE.findall(text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def build_index(items, n, n_short):
    """
    n-gram -> evaluation item id. Long problems are indexed by their n-grams (n_short if shorter than n words);
    problems shorter than n_short words (e.g. "What is 98 plus 45?") are indexed by their whole normalized text,
    so that a document containing them verbatim is flagged too.
    """
    index, short_lengths = {}, set()
    for item in items:
        text = item.get("prompt") or item["question"]
        grams = ngrams(text, n) or ngrams(text, n_short)
        if not grams:
            words = WORD_RE.findall(text.lower())
            if len(words) >= 3:
                grams = {" ".join(words)}
                short_lengths.add(len(words))
        for g in grams:
            index.setdefault(g, item["id"])
    index["__short_lengths__"] = sorted(short_lengths)
    return index


def doc_hits(text, index, n, n_short):
    hits = set()
    words = WORD_RE.findall(text.lower())
    for k in {n, n_short, *index.get("__short_lengths__", [])}:
        for i in range(len(words) - k + 1):
            g = " ".join(words[i:i + k])
            if g in index:
                hits.add(index[g])
    return hits


def iter_docs(args):
    for name in [s for s in (args.sources or "").split(",") if s]:
        count = 0
        for path in list_parquet_files(name, "train"):
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                for text in pf.read_row_group(rg).column("text").to_pylist():
                    yield name, path, text
                    count += 1
                    if 0 < args.max_docs <= count:
                        break
                if 0 < args.max_docs <= count:
                    break
            if 0 < args.max_docs <= count:
                break
    for path in [p for p in (args.jsonl or "").split(",") if p]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield path, path, json.dumps(json.loads(line), ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="n-gram decontamination check")
    parser.add_argument("--suite", type=str, default="data/val", help="evaluation suite directory")
    parser.add_argument("--sources", type=str, default=None, help="comma-separated data sources (data/<source>/train)")
    parser.add_argument("--jsonl", type=str, default=None, help="comma-separated jsonl files (e.g. SFT data)")
    parser.add_argument("--n", type=int, default=13)
    parser.add_argument("--n-short", type=int, default=8)
    parser.add_argument("--max-docs", type=int, default=-1, help="max documents per source (-1 = all)")
    parser.add_argument("--write-clean", action="store_true", help="write <source><suffix>/ copies of the sources without flagged documents")
    parser.add_argument("--clean-suffix", type=str, default="_clean", help="name suffix of the cleaned copies")
    parser.add_argument("--report", type=str, default=None, help="write flagged (document, item) pairs to this jsonl")
    args = parser.parse_args()

    items = load_suite(args.suite)
    index = build_index(items, args.n, args.n_short)
    print(f"Indexed {len(index):,} n-grams from {len(items)} evaluation items in {args.suite}")

    stats, flagged, clean_docs = {}, [], {}
    for name, path, text in iter_docs(args):
        n_docs, n_flagged = stats.get(name, (0, 0))
        hits = doc_hits(text, index, args.n, args.n_short)
        stats[name] = (n_docs + 1, n_flagged + int(bool(hits)))
        if hits:
            flagged.append({"source": name, "file": path, "items": sorted(hits), "text": text[:500]})
        elif args.write_clean and args.sources and name in args.sources.split(","):
            clean_docs.setdefault(name, []).append(text)

    for name, (n_docs, n_flagged) in stats.items():
        print(f"{name}: {n_flagged:,} / {n_docs:,} documents flagged ({100 * n_flagged / max(n_docs, 1):.3f}%)")
    for f in flagged[:10]:
        print(f"  e.g. {f['source']} overlaps with {f['items'][:3]}: {f['text'][:150]!r}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fout:
            for f in flagged:
                fout.write(json.dumps(f) + "\n")
        print(f"Wrote {len(flagged)} flagged documents to {args.report}")
    for name, docs in clean_docs.items():
        out = os.path.join(source_dir(f"{name}{args.clean_suffix}", "train"), "part_00000.parquet")
        write_parquet_docs(docs, out)
        print(f"Wrote {len(docs):,} clean documents to {out} (copy or symlink a val split if you need one)")


if __name__ == "__main__":
    main()
