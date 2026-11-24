"""
SFT mixtures over SFT sources built by the standard data pipeline (python -m scripts.prepare_data --config ...).

An SFT mixture is a comma-separated list of "<source>:<epochs>", e.g.
    "gsm8k:4,mbpp:8,tinygsm:1,smoltalk:1,my_sft:2"
where every <source> is declared in the "sft" section of your data config (a dataset of the catalog, see
scripts/prepare_data.py) and built into $NANOCHAT_BASE_DIR/data/sft/<source>/{train,val}.jsonl (+ manifest.json).
Epochs can be fractional: 0.1 means the first 10% of the (shuffled) source. Integer parts repeat the source.
"""

import os

from tasks.common import Task, TaskMixture
from tasks.customjson import CustomJSON
from nanochat.datapipe import source_root, fingerprints


class TaskView(Task):
    """A contiguous slice [start, stop) of another task."""

    def __init__(self, task, start=0, stop=None):
        super().__init__(start=start, stop=stop)
        self.task = task

    @property
    def eval_type(self):
        return self.task.eval_type

    def num_examples(self):
        return len(self.task)

    def get_example(self, index):
        return self.task[index]


def build_task(name, kind="sft"):
    """Returns (train_task, val_task or None) for a built SFT source (or, with kind="rl", the RL problems)."""
    fingerprints([name], kind) # asserts that the source was built by the pipeline
    root = source_root(name, kind)
    val_path = os.path.join(root, "val.jsonl")
    return CustomJSON(os.path.join(root, "train.jsonl")), (CustomJSON(val_path) if os.path.exists(val_path) else None)


def mixture_sources(spec):
    """Source names of an SFT mixture spec."""
    names = []
    for part in spec.split(","):
        name, _, epochs = part.strip().rpartition(":")
        try:
            float(epochs)
        except ValueError:
            name = part.strip()
        names.append(name or part.strip())
    return names


def build_sft_mixture(spec):
    """Parse an SFT mixture spec. Returns (train TaskMixture, val TaskMixture, description)."""
    train_tasks, val_tasks, desc = [], [], []
    for part in spec.split(","):
        part = part.strip()
        name, _, epochs = part.rpartition(":")
        try:
            epochs = float(epochs)
        except ValueError: # no epochs given
            name, epochs = part, 1.0
        name = name or part
        train, val = build_task(name)
        full, frac = int(epochs), epochs - int(epochs)
        train_tasks.extend([train] * full)
        if frac > 0:
            train_tasks.append(TaskView(train, 0, max(1, int(frac * len(train)))))
        if val is not None:
            val_tasks.append(val)
        desc.append(f"{name} x{epochs:g} ({len(train):,} rows)")
    train_mixture = TaskMixture(train_tasks)
    val_mixture = TaskMixture(val_tasks) if val_tasks else TaskMixture([TaskView(train_tasks[0], 0, min(500, len(train_tasks[0])))])
    return train_mixture, val_mixture, ", ".join(desc)
