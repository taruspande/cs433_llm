"""
The official solver: one chat prompt, k samples of your model, then aggregate.

!!! DO NOT MODIFY THIS FILE. Graders evaluate every submission with our own copy of it. !!!

The configuration is FIXED: every submission is evaluated with the official one (OFFICIAL_CONFIG in
scripts/evaluate.py: greedy decoding, one sample of at most 512 new tokens). Teams choose nothing at test time.

Config keys (other values are for staff diagnostics only):
    num_samples       number of samples per problem (default 1)
    temperature       sampling temperature (default 0.0 = greedy)
    top_k             top-k sampling (default 50)
    max_new_tokens    maximum tokens per sample (default 512)
    math_aggregation  "majority" (self-consistency) | "first"
    code_selection    "visible_tests" (first candidate that passes the shown tests) | "first"
    max_batch_rows    rows (problems x samples) generated together (default 256; speed only, not the result)

Interface used by scripts/evaluate.py: Solver(models, config) with models = {"policy": MeteredLM}; solve(item) -> str, and
solve_batch(items) -> list of str (batched generation with MeteredLM.generate_many: same metering, much faster).
For math the output contains the final answer (e.g. "#### 42"); for code, the program.
"""

from collections import Counter

from nanochat.inference import item_seed
from tasks.eval_suite import extract_math_answer, extract_program, run_visible_tests, parse_number


class Solver:

    def __init__(self, models, config):
        self.lm = models["policy"]
        self.num_samples = int(config.get("num_samples", 1))
        self.temperature = float(config.get("temperature", 0.0))
        self.top_k = int(config.get("top_k", 50))
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.math_aggregation = config.get("math_aggregation", "majority")
        self.code_selection = config.get("code_selection", "visible_tests")
        self.max_batch_rows = int(config.get("max_batch_rows", 256)) # rows (problems x samples) generated together
        if self.num_samples > 1 and self.temperature == 0.0:
            print("WARNING: num_samples > 1 with temperature 0 produces identical samples")

    def sample(self, item):
        prompt_ids = self.lm.render_chat(item["question"])
        completions = self.lm.generate(
            prompt_ids,
            num_samples=self.num_samples,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            seed=item_seed(item["id"]),
        )
        return [self.lm.decode(c) for c in completions]

    def solve(self, item):
        texts = self.sample(item)
        if item["family"] == "math":
            return self.aggregate_math(texts)
        return self.select_code(item, texts)

    def solve_batch(self, items):
        """Same as [solve(item) for item in items], but with batched generation (much faster on a GPU)."""
        prompts = [self.lm.render_chat(item["question"]) for item in items]
        completions = []
        rows_per_call = max(1, self.max_batch_rows // self.num_samples) # bound the KV-cache memory
        for b in range(0, len(items), rows_per_call):
            completions.extend(self.lm.generate_many(
                prompts[b:b + rows_per_call], keys=[item["id"] for item in items[b:b + rows_per_call]],
                num_samples=self.num_samples, max_tokens=self.max_new_tokens,
                temperature=self.temperature, top_k=self.top_k,
                seeds=[item_seed(item["id"]) for item in items[b:b + rows_per_call]],
            ))
        outputs = []
        for item, comps in zip(items, completions):
            texts = [self.lm.decode(c) for c in comps]
            outputs.append(self.aggregate_math(texts) if item["family"] == "math" else self.select_code(item, texts))
        return outputs

    def aggregate_math(self, texts):
        answers = [extract_math_answer(t) for t in texts]
        answers = [a for a in answers if a is not None]
        if not answers:
            return texts[0]
        if self.math_aggregation == "first":
            return f"#### {answers[0]}"
        # majority vote over numerically-normalized answers (self-consistency)
        counts = Counter(parse_number(a) for a in answers)
        best_value, _ = counts.most_common(1)[0]
        best = next(a for a in answers if parse_number(a) == best_value)
        return f"#### {best}"

    def select_code(self, item, texts):
        programs = [extract_program(t) for t in texts]
        if self.code_selection == "visible_tests" and len(programs) > 1:
            for program in programs:
                if program.strip() and run_visible_tests(item, program):
                    return program
        return programs[0]
