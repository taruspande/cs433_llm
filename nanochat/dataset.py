"""
Pretraining data = a weighted mixture of *data sources*.

A data source is a directory built by the standard data pipeline (nanochat/datapipe.py):

    $NANOCHAT_BASE_DIR/data/<source_name>/train/*.parquet   (a `text` column)
    $NANOCHAT_BASE_DIR/data/<source_name>/val/*.parquet
    $NANOCHAT_BASE_DIR/data/<source_name>/manifest.json

Build sources with `python -m scripts.prepare_data --config <data config>`. Your own data (filtered,
synthetic, ...) is declared in the data config as local files and goes through the same pipeline.
Only built sources (with a manifest) can be used in a mixture.

A mixture assigns a weight to each source, e.g. {"fineweb": 0.8, "finemath": 0.2}.
Weights are proportions of *tokens* (not documents). A mixture can be given as:
- a path to a JSON file, e.g. configs/mixtures/baseline.json
- an inline JSON string, e.g. '{"fineweb": 0.8, "finemath": 0.2}'
- a compact string, e.g. "fineweb:0.8,finemath:0.2"
"""

import os
import json

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Source locations

def data_dir():
    return os.path.join(get_base_dir(), "data")


def source_dir(name, split):
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    return os.path.join(data_dir(), name, split)


def list_sources():
    """Names of all built pretraining sources."""
    if not os.path.exists(data_dir()):
        return []
    return sorted(n for n in os.listdir(data_dir()) if os.path.exists(os.path.join(data_dir(), n, "manifest.json")))


def list_parquet_files(name, split):
    """Full paths of all parquet files of a source split (sorted, may be empty)."""
    d = source_dir(name, split)
    if not os.path.exists(d):
        return []
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(d, f) for f in files]


def write_parquet_docs(texts, path, row_group_size=1024):
    """Write a list of documents (strings) as a parquet file in the format expected by the dataloader."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.Table.from_pydict({"text": texts})
    tmp_path = path + ".tmp"
    pq.write_table(table, tmp_path, row_group_size=row_group_size)
    os.rename(tmp_path, path)

# -----------------------------------------------------------------------------
# Mixtures

def load_mixture(spec):
    """Parse a mixture spec (see module docstring) into a list of (source, weight), weights summing to 1."""
    if isinstance(spec, dict):
        raw = spec
    elif os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw = raw.get("sources", raw) # allow {"sources": {...}, "comment": ...}
    elif spec.strip().startswith("{"):
        raw = json.loads(spec)
    else:
        raw = {}
        for part in spec.split(","):
            name, _, weight = part.strip().partition(":")
            raw[name] = float(weight) if weight else 1.0
    mixture = [(name, float(w)) for name, w in raw.items() if float(w) > 0]
    assert mixture, f"Empty mixture: {spec}"
    total = sum(w for _, w in mixture)
    mixture = [(name, w / total) for name, w in mixture]
    for name, _ in mixture:
        assert os.path.exists(os.path.join(data_dir(), name, "manifest.json")), (
            f"Data source '{name}' was not built by the standard pipeline (no {os.path.join(data_dir(), name, 'manifest.json')}). "
            f"Declare it in a data config and run `python -m scripts.prepare_data --config <config>`.")
    return mixture


def format_mixture(mixture):
    return ", ".join(f"{name}={w:.3f}" for name, w in mixture)

# -----------------------------------------------------------------------------
# Simple iteration (used e.g. by the tokenizer training)

def parquets_iter_batched(split, sources, start=0, step=1):
    """
    Iterate through the documents of the given sources, one source after the other,
    in batches of underlying row groups. start/step are useful for skipping row groups in DDP.
    """
    for name in sources:
        for filepath in list_parquet_files(name, split):
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(start, pf.num_row_groups, step):
                rg = pf.read_row_group(rg_idx)
                yield rg.column('text').to_pylist()
