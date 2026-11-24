"""
A built SFT source (data/sft/<name>/train.jsonl, written by the standard pipeline), read back as a task. Each line is
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
or the shorthand
    {"question": "...", "answer": "..."}
Used by tasks/registry.py for SFT mixtures (and the RL problems).
Conversations are read from disk when they are used: only the position of every line is kept in memory.
"""

import os
import json

import numpy as np

from tasks.common import Task

class CustomJSON(Task):

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        offsets, position = [], 0
        with open(filepath, "rb") as f:
            for line in f:
                if line.strip():
                    offsets.append(position)
                position += len(line)
        self.offsets = np.array(offsets, dtype=np.int64)
        self._file, self._pid = None, None

    def num_examples(self):
        return len(self.offsets)

    def get_example(self, index):
        if self._pid != os.getpid(): # one file handle per process
            self._file, self._pid = open(self.filepath, "rb"), os.getpid()
        self._file.seek(int(self.offsets[index]))
        row = json.loads(self._file.readline())
        if "messages" not in row:
            row = {"messages": [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": row["answer"]}]}
        return row
