"""
Official inference code: metered model access for the official solver (solvers/default.py).

!!! DO NOT MODIFY THIS FILE. The official grading uses our own copy of it. !!!

The solver never touches the model directly. It receives a MeteredLM that counts the tokens
of every call against the inference allowance of each problem (see nanochat/compute.py:
INFERENCE_TOKENS_PER_PROBLEM), and their FLOPs (for information only: inference is not part
of the budget).
"""

import os
import sys
import zlib
import importlib.util

import torch

from nanochat.engine import Engine
from nanochat.compute import generation_flops, scoring_flops, generation_tokens, INFERENCE_TOKENS_PER_PROBLEM


class MeteredLM:
    """A model + tokenizer, with token and FLOP metering. This is the only way solvers may use a model."""

    def __init__(self, model, tokenizer, name="policy"):
        self._model = model
        self._engine = Engine(model, tokenizer)
        self.tokenizer = tokenizer
        self.name = name
        self.max_seq_len = model.config.sequence_len
        self.flops = 0.0 # FLOPs since the last reset (i.e. for the current problem) not attributed to a key
        self.tokens = 0 # allowance used since the last reset, not attributed to a key
        self.flops_by_key = {} # batched mode (generate_many): FLOPs per problem id
        self.tokens_by_key = {} # batched mode: allowance used per problem id

    def reset(self):
        self.flops = 0.0
        self.tokens = 0
        self.flops_by_key = {}
        self.tokens_by_key = {}

    # --- prompt helpers ------------------------------------------------------

    def render_chat(self, user_message, assistant_prefix=None):
        """Token ids of a single-turn conversation, primed for the assistant to answer."""
        conversation = {"messages": [{"role": "user", "content": user_message}, {"role": "assistant", "content": ""}]}
        ids = self.tokenizer.render_for_completion(conversation)
        if assistant_prefix:
            ids = ids + self.tokenizer.encode(assistant_prefix)
        return ids

    def encode(self, text, prepend_bos=True):
        return self.tokenizer.encode(text, prepend=self.tokenizer.get_bos_token_id() if prepend_bos else None)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    # --- metered model calls -------------------------------------------------

    def generate(self, prompt_ids, num_samples=1, max_tokens=256, temperature=0.0, top_k=50, seed=0):
        """
        Sample num_samples completions of prompt_ids (in one batch, sharing the prompt prefill).
        Returns a list of num_samples lists of generated token ids (prompt excluded,
        terminal tokens excluded). Tool calls (calculator) are executed inside the Engine.
        Generation stops early when the problem's inference allowance is used up.
        """
        assert isinstance(prompt_ids, list) and len(prompt_ids) > 0
        assert num_samples >= 1
        # respect the context length of the model and the inference allowance of the problem
        max_tokens = min(max_tokens, self.max_seq_len - len(prompt_ids))
        max_tokens = min(max_tokens, (INFERENCE_TOKENS_PER_PROBLEM - self.tokens - len(prompt_ids)) // num_samples)
        if max_tokens <= 0:
            return [[] for _ in range(num_samples)]
        results, _ = self._engine.generate_batch(
            prompt_ids, num_samples=num_samples, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k, seed=seed,
        )
        completions = [r[len(prompt_ids):] for r in results]
        # rows run in lockstep: the batch is busy until the longest row is done (+1 for the terminal token)
        num_steps = min(max(len(c) for c in completions) + 1, max_tokens)
        self.flops += generation_flops(self._model, len(prompt_ids), num_samples, num_steps)
        self.tokens += generation_tokens(len(prompt_ids), num_samples, num_steps)
        return completions

    def generate_many(self, prompts, keys, num_samples=1, max_tokens=256, temperature=0.0, top_k=50, seeds=None):
        """
        Batched version of generate() for several problems at once (much faster on a GPU).
        prompts: list of prompt token lists; keys: the problem id of each prompt (used to attribute tokens and FLOPs
        to problems); seeds: one seed per prompt. Returns one list of num_samples completions per prompt.
        Every prompt is metered exactly as if generate() had been called on it alone.
        """
        assert len(prompts) == len(keys) and len(prompts) > 0 and num_samples >= 1
        seeds = [0] * len(prompts) if seeds is None else list(seeds)
        limits = []
        for prompt, key in zip(prompts, keys):
            assert isinstance(prompt, list) and len(prompt) > 0
            used = self.tokens_by_key.get(key, 0)
            limits.append(min(max_tokens, self.max_seq_len - len(prompt), (INFERENCE_TOKENS_PER_PROBLEM - used - len(prompt)) // num_samples))
        outputs = [[[] for _ in range(num_samples)] for _ in prompts]
        todo = [i for i, limit in enumerate(limits) if limit > 0]
        if not todo:
            return outputs
        if hasattr(self._engine, "generate_multi"):
            results, _ = self._engine.generate_multi(
                [prompts[i] for i in todo], num_samples=num_samples, max_tokens=[limits[i] for i in todo],
                temperature=temperature, top_k=top_k, seeds=[seeds[i] for i in todo],
            )
        else: # an engine without batched generation: one prompt at a time
            results = []
            for i in todo:
                rows, _ = self._engine.generate_batch(prompts[i], num_samples=num_samples, max_tokens=limits[i],
                                                      temperature=temperature, top_k=top_k, seed=seeds[i])
                results.append([r[len(prompts[i]):] for r in rows])
        for i, completions in zip(todo, results):
            outputs[i] = completions
            # identical accounting to generate(): one prefill + lockstep decode until the longest row is done
            num_steps = min(max(len(c) for c in completions) + 1, limits[i])
            key = keys[i]
            self.flops_by_key[key] = self.flops_by_key.get(key, 0.0) + generation_flops(self._model, len(prompts[i]), num_samples, num_steps)
            self.tokens_by_key[key] = self.tokens_by_key.get(key, 0) + generation_tokens(len(prompts[i]), num_samples, num_steps)
        return outputs

    @torch.no_grad()
    def logprobs(self, sequences):
        """
        Per-token log-probabilities of each sequence (list of lists of token ids) under the model.
        Returns a list of lists of floats; entry t is log p(seq[t+1] | seq[:t+1]).
        Useful for reranking / verification. Every scored token counts against the inference allowance.
        """
        device = self._model.get_device()
        out = []
        for seq in sequences:
            seq = seq[:self.max_seq_len]
            ids = torch.tensor([seq], dtype=torch.long, device=device)
            logits = self._model(ids[:, :-1]) # (1, T-1, V)
            logp = torch.log_softmax(logits.float(), dim=-1)
            token_logp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
            out.append(token_logp.tolist())
        lens = [min(len(s), self.max_seq_len) - 1 for s in sequences]
        self.flops += scoring_flops(self._model, lens)
        self.tokens += sum(lens)
        return out


def item_seed(item_id, salt=0):
    """A deterministic per-problem seed, so that evaluations are reproducible."""
    return (zlib.crc32(item_id.encode("utf-8")) + salt) & 0x7FFFFFFF


def load_solver_class(path):
    """Import the class `Solver` from a python file (e.g. solvers/default.py)."""
    path = os.path.abspath(path)
    module_name = "user_solver_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "Solver"), f"{path} must define a class named `Solver`"
    return module.Solver
