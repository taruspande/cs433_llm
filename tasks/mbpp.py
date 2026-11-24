"""
MBPP (Mostly Basic Python Problems), as SFT training data.
https://huggingface.co/datasets/google-research-datasets/mbpp

Only the train / validation / prompt splits may be used for training (task ids 1-10, 511-974).
The test split (task ids 11-510) is part of the evaluation suites and is FORBIDDEN for training.

!!! DO NOT MODIFY THIS FILE: it is part of the dataset catalog (see scripts/prepare_data.py). !!!
"""

from tasks.common import Task, load_hub_dataset
from tasks.eval_suite import render_mbpp_question

ALLOWED_SPLITS = ("train", "validation", "prompt")

class MBPP(Task):

    def __init__(self, splits=("train", "prompt"), **kwargs):
        super().__init__(**kwargs)
        assert all(s in ALLOWED_SPLITS for s in splits), f"MBPP splits must be in {ALLOWED_SPLITS} (the test split is reserved for evaluation)"
        self.rows = []
        for split in splits:
            ds = load_hub_dataset("google-research-datasets/mbpp", subset="full", split=split)
            self.rows.extend(ds[i] for i in range(len(ds)))

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        row = self.rows[index]
        code = row["code"].replace("\r\n", "\n").strip("\n")
        messages = [
            {"role": "user", "content": render_mbpp_question(row["text"], row["test_list"][:1])},
            {"role": "assistant", "content": f"```python\n{code}\n```"},
        ]
        return {"messages": messages}
