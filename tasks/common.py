"""
Base class for all Tasks.
A Task is basically a dataset of conversations, together with some
metadata and often also evaluation criteria.
Example tasks: GSM8K, MBPP, SmolTalk, MetaMathQA, TinyGSM (see tasks/registry.py).
"""

import os
import json
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from nanochat.common import get_base_dir, hf_request


class HubDataset:
    """
    Minimal stand-in for a HuggingFace datasets Dataset: wraps a pyarrow
    Table and offers lazy row access and a seeded shuffle.
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self, seed):
        # matches datasets.Dataset.shuffle(seed=seed) exactly, row order comes out identical
        permutation = np.random.default_rng(seed).permutation(len(self))
        return HubDataset(self.table, permutation)

    def __getitem__(self, index):
        physical_index = index if self.permutation is None else int(self.permutation[index])
        row = {column: self.table[column][physical_index].as_py() for column in self.table.column_names}
        return row


def load_hub_dataset(repo_id, subset="default", split="train"):
    """
    Minimal stand-in for HuggingFace datasets.load_dataset(repo_id, subset, split=split).
    Every dataset on the hub has an auto-generated parquet export. We list the parquet
    shards via the hub API, download them (once) into the local cache directory, and
    read them with pyarrow. Under torchrun, only one rank downloads, the others wait.
    """
    base_dir = get_base_dir()
    slug = repo_id.replace("/", "--")
    shards_dir = os.path.join(base_dir, "task_data", slug, subset, split)
    # the manifest is written last, so its existence means the download completed
    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        os.makedirs(shards_dir, exist_ok=True)
        with FileLock(manifest_path + ".lock"):
            # only a single rank acquires the lock and downloads, the others block
            # here and then skip the download because they recheck the manifest
            if not os.path.exists(manifest_path):
                listing_url = f"https://huggingface.co/api/datasets/{repo_id}/parquet/{subset}/{split}"
                shard_urls = json.loads(hf_request(listing_url))
                filenames = []
                for shard_index, shard_url in enumerate(shard_urls):
                    filename = f"{shard_index:05d}.parquet"
                    print(f"Downloading {shard_url} ...")
                    content = hf_request(shard_url)
                    with open(os.path.join(shards_dir, filename), "wb") as f:
                        f.write(content)
                    filenames.append(filename)
                with open(manifest_path, "w") as f:
                    json.dump(filenames, f)
    with open(manifest_path, "r") as f:
        filenames = json.load(f)
    shard_paths = [os.path.join(shards_dir, filename) for filename in filenames]
    tables = [pq.read_table(path) for path in shard_paths]
    table = pa.concat_tables(tables)
    return HubDataset(table)


class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    @property
    def eval_type(self):
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a mixture of datasets.
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        self.starts = np.cumsum([0] + self.lengths) # conversation i of the concatenated tasks is in task searchsorted(starts, i)
        # Deterministically shuffle to mix tasks throughout training (kept as a compact array)
        order = list(range(self.num_conversations))
        random.Random(42).shuffle(order)
        self.order = np.array(order, dtype=np.int64)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.
        This ensures tasks are mixed throughout training, regardless of dataset size.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        position = int(self.order[index])
        task_idx = int(np.searchsorted(self.starts, position, side="right")) - 1
        return self.tasks[task_idx][position - int(self.starts[task_idx])]


class TaskSequence(Task):
    """
    For SFT Training sometimes we want to sequentially train on a list of tasks.
    This is useful for cases that require a training curriculum.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length
