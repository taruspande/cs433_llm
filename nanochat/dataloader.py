"""
Distributed dataloaders for pretraining on a *mixture* of data sources.

Mixture sampling:
   - Each source (see nanochat/dataset.py) is streamed independently, sharded across DDP ranks.
   - Whenever the document buffer needs refilling, we pull the next batch of documents from
     the source that is most behind its target share of tokens. This deterministically keeps
     the token proportions of the stream at the mixture weights (measured before packing).
   - When a source is exhausted it restarts from the beginning (epoch += 1 for that source).
     Watch the per-source epoch counters in the training logs to see how often you repeat data!

BOS-aligned bestfit packing (unchanged from nanochat):
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding)
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _source_batches(name, split, resume_state, tokenizer_batch_size):
    """
    Infinite iterator over document batches of a single source.
    Each yield is (text_batch, state) where state = {pq_idx, rg_idx, epoch} allows approximate resumption.
    Row groups are strided across DDP ranks. If a file has fewer row groups than ranks,
    every rank reads every row group and keeps a rank-strided subset of its rows instead.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    parquet_paths = list_parquet_files(name, split)
    assert len(parquet_paths) != 0, f"No {split} parquet files found for data source '{name}'"

    resume_pq_idx = resume_state["pq_idx"] if resume_state is not None else 0
    resume_rg_idx = resume_state["rg_idx"] if resume_state is not None else None
    epoch = resume_state.get("epoch", 1) if resume_state is not None else 1
    first_pass = True

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            pf = pq.ParquetFile(parquet_paths[pq_idx])
            row_sharding = pf.num_row_groups < ddp_world_size # small file: shard rows instead of row groups
            rg_start, rg_step = (0, 1) if row_sharding else (ddp_rank, ddp_world_size)
            rg_idx = rg_start
            # Start from the resume point if resuming on the same file (the saved state is the one of rank 0)
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // rg_step
                rg_idx = (base_idx + 1) * rg_step + rg_start # advance by one so we don't repeat data after resuming
                resume_rg_idx = None # only do this once
            while rg_idx < pf.num_row_groups:
                batch = pf.read_row_group(rg_idx).column('text').to_pylist()
                if row_sharding:
                    batch = batch[ddp_rank::ddp_world_size]
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
                rg_idx += rg_step
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split, mixture,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping over a mixture of sources.

    mixture: list of (source_name, weight) with weights summing to 1 (see dataset.load_mixture).
    For the "val" split, sources without validation files are skipped (and weights renormalized).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Yields (inputs, targets, state_dict); state_dict can be passed back as resume_state_dict.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    if split == "val":
        mixture = [(name, w) for name, w in mixture if list_parquet_files(name, "val")]
        assert mixture, "None of the sources in the mixture has validation files"
        total = sum(w for _, w in mixture)
        mixture = [(name, w / total) for name, w in mixture]

    row_capacity = T + 1
    resume_sources = resume_state_dict["sources"] if resume_state_dict is not None else {}
    iterators = {name: _source_batches(name, split, resume_sources.get(name), tokenizer_batch_size) for name, _ in mixture}
    source_states = {name: dict(resume_sources.get(name, {"pq_idx": 0, "rg_idx": 0, "epoch": 1})) for name, _ in mixture}
    source_tokens = {name: 0 for name, _ in mixture}
    if resume_state_dict is not None:
        source_tokens.update({k: v for k, v in resume_state_dict["tokens"].items() if k in source_tokens})
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []

    def refill_buffer():
        # pick the source that is most behind its target share of tokens
        total_tokens = sum(source_tokens.values())
        name = max(mixture, key=lambda nw: nw[1] * (total_tokens + 1) - source_tokens[nw[0]])[0]
        doc_batch, source_states[name] = next(iterators[name])
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)
            source_tokens[name] += len(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {
            "sources": {name: dict(s) for name, s in source_states.items()},
            "tokens": dict(source_tokens),
        }

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
