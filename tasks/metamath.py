"""
MetaMathQA: 395K math question/answer pairs, augmented (by GPT-3.5) from the GSM8K and MATH *train* sets.
https://huggingface.co/datasets/meta-math/MetaMathQA
Responses end with "The answer is: <answer>", which tasks/eval_suite.py knows how to extract.

!!! DO NOT MODIFY THIS FILE: it is part of the dataset catalog (see scripts/prepare_data.py). !!!
"""

from tasks.common import Task, load_hub_dataset

class MetaMathQA(Task):

    def __init__(self, gsm_only=True, **kwargs):
        super().__init__(**kwargs)
        ds = load_hub_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=42)
        rows = (ds[i] for i in range(len(ds)))
        # the "type" column is e.g. GSM_Rephrased, GSM_AnsAug, MATH_FOBAR, ...
        self.rows = [r for r in rows if r["type"].startswith("GSM")] if gsm_only else list(rows)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        row = self.rows[index]
        messages = [
            {"role": "user", "content": row["query"]},
            {"role": "assistant", "content": row["response"]},
        ]
        return {"messages": messages}
