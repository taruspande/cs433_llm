"""
The evaluation suite: loading, prompt rendering, answer extraction and scoring.

!!! DO NOT MODIFY THIS FILE. The official grading uses our own copy of it. !!!

A suite is a directory with one jsonl file per family (math_*.jsonl, code_*.jsonl).
We release the *validation* suite in data/val/. The *test* suite used for grading is
hidden; it is drawn from the same sources and rendered in exactly the same way.

Subsets (easiest first):
    math: arithmetic (GPT-3 arithmetic battery), gsm8k_easy (GSM8K test problems with <= 2 calculation steps)
    code: python_math (write solution() returning the answer to a simple word problem),
          mbpp_easy (MBPP problems with short reference solutions, scored with MBPP+ tests)

Every item has the fields:
    id        unique id, e.g. "gsm8k_easy/123"
    family    "math" | "code"
    subset    the source/sub-benchmark, see above
    question  the user message shown to the model (already rendered, see render_* below)
Math items additionally have:
    answer    the reference final answer (a number, as a string)
Code items additionally have:
    prompt         the raw problem (natural language text)
    entry_point    the name of the function to implement
    visible_tests  a few assert statements shown in the prompt (may be empty)
    test           the full (hidden) test code that decides correctness
    imports        import lines needed by the tests / stub

Your solver only ever sees the public fields (see `public_view`): never `answer` or `test`.

Scoring:
    item score   math: extracted final answer == reference (numerically)
                 code: the returned program passes the full hidden test (sandboxed, with timeout)
    subset score mean item score
    family score unweighted mean over the subsets of the family
    final score  weighted mean over families (FAMILY_WEIGHTS)
"""

import os
import re
import json
import glob

from nanochat.execution import execute_code

FAMILY_WEIGHTS = {"math": 0.5, "code": 0.5}
PRIVATE_FIELDS = {"answer", "test"}
CODE_TIMEOUT_SECONDS = 20.0
CODE_MEMORY_BYTES = 2 * 1024 ** 3

# -----------------------------------------------------------------------------
# Prompt rendering (shared with the SFT tasks so that training and evaluation formats agree)

def render_math_question(question):
    return question.strip()


def render_mbpp_question(text, visible_tests):
    tests = "\n".join(visible_tests)
    return f"{text.strip()}\nYour code should pass this test:\n{tests}"


def render_python_math_question(question):
    return ("Write a Python function `solution()` that computes and returns the numerical answer to this problem.\n"
            f"{question.strip()}")

# -----------------------------------------------------------------------------
# Loading

def load_suite(suite_dir, families=("math", "code")):
    items = []
    for family in families:
        for path in sorted(glob.glob(os.path.join(suite_dir, f"{family}_*.jsonl"))):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        assert item["family"] == family, f"{path}: item {item['id']} has family {item['family']}"
                        items.append(item)
    assert items, f"No items found in {suite_dir} for families {families}"
    return items


def public_view(item):
    """The part of an item that a solver is allowed to see."""
    return {k: v for k, v in item.items() if k not in PRIVATE_FIELDS}

# -----------------------------------------------------------------------------
# Math: answer extraction and comparison

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")

def parse_number(s):
    """Parse a number from a string like '$1,234.50' or '12.0'. Returns float or None."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    match = NUMBER_RE.search(s)
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def extract_math_answer(text):
    """
    Extract the final answer from a model response, in order of preference:
    1) the number after the last '####' (GSM8K convention, used by our SFT data)
    2) the number after the last 'answer is' (e.g. MetaMathQA: 'The answer is: 42')
    3) the number inside the last \\boxed{...}
    4) the last number in the text
    Returns the number as a string, or None.
    """
    if text is None:
        return None
    for pattern in [r"####\s*([^\n]*)", r"answer is[:\s]*([^\n]*)", r"\\boxed\{([^}]*)\}"]:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for candidate in reversed(matches):
            match = NUMBER_RE.search(candidate.replace(",", "").replace("$", ""))
            if match:
                return match.group(0)
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def math_answers_match(pred, ref, rel_tol=1e-6):
    p, r = parse_number(pred), parse_number(ref)
    if p is None or r is None:
        return False
    return abs(p - r) <= rel_tol * max(1.0, abs(r))

# -----------------------------------------------------------------------------
# Code: program extraction and execution

def extract_program(completion):
    """Return the first ```python ...``` (or ``` ... ```) block of a completion, else the whole completion."""
    if completion is None:
        return ""
    matches = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", completion, re.DOTALL)
    if matches:
        return matches[0].strip("\n")
    # an unterminated block (e.g. generation was cut off)
    match = re.search(r"```(?:python|py)?[^\n]*\n(.*)", completion, re.DOTALL)
    if match:
        return match.group(1).strip("\n")
    return completion.strip("\n")


def build_program(item, code, tests):
    """Assemble the program that is executed: imports + solution + tests."""
    return f"{item.get('imports', '')}\n\n{code}\n\n{tests}\n"


def run_tests(item, code, tests, timeout=CODE_TIMEOUT_SECONDS):
    """Run `tests` (a string of Python code that raises on failure) against `code`. Returns bool."""
    program = build_program(item, code, tests)
    result = execute_code(program, timeout=timeout, maximum_memory_bytes=CODE_MEMORY_BYTES)
    return bool(result.success)


def run_visible_tests(item, code, timeout=CODE_TIMEOUT_SECONDS):
    """Solvers may use this to check a candidate program against the tests shown in the prompt."""
    if not item.get("visible_tests"):
        return True
    return run_tests(item, code, "\n".join(item["visible_tests"]), timeout=timeout)

# -----------------------------------------------------------------------------
# Scoring

def score_item(item, solver_output):
    """Score the output of a solver for one item: returns (correct: bool, extracted: str)."""
    if item["family"] == "math":
        pred = extract_math_answer(solver_output)
        return math_answers_match(pred, item["answer"]), pred
    elif item["family"] == "code":
        code = extract_program(solver_output)
        if not code.strip():
            return False, code
        return run_tests(item, code, item["test"]), code
    raise ValueError(f"Unknown family: {item['family']}")


def aggregate_scores(records):
    """records: list of dicts with keys id, family, subset, correct. Returns a nested summary dict."""
    subsets = {}
    for r in records:
        key = (r["family"], r["subset"])
        n, c = subsets.get(key, (0, 0))
        subsets[key] = (n + 1, c + int(r["correct"]))
    summary = {"subsets": {}, "families": {}}
    for (family, subset), (n, c) in sorted(subsets.items()):
        summary["subsets"][f"{family}/{subset}"] = {"n": n, "accuracy": c / n}
    for family in sorted({f for f, _ in subsets}):
        accs = [c / n for (f, _), (n, c) in subsets.items() if f == family]
        summary["families"][family] = sum(accs) / len(accs)
    weights = {f: FAMILY_WEIGHTS.get(f, 0.0) for f in summary["families"]}
    total_weight = sum(weights.values())
    summary["final_score"] = sum(weights[f] * summary["families"][f] for f in weights) / total_weight if total_weight > 0 else 0.0
    return summary


def format_summary(summary):
    lines = []
    for name, s in summary["subsets"].items():
        lines.append(f"{name:>28s}: {100 * s['accuracy']:6.2f}%  (n={s['n']})")
    for family, acc in summary["families"].items():
        lines.append(f"{'family ' + family:>28s}: {100 * acc:6.2f}%")
    lines.append(f"{'FINAL SCORE':>28s}: {100 * summary['final_score']:6.2f}")
    return "\n".join(lines)
