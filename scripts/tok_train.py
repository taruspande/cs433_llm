"""
Train the tokenizer using our own BPE Tokenizer library, in the style of the GPT-4 tokenizer.

!!! DO NOT MODIFY THIS FILE. The tokenizer is FIXED: every team trains the same one (README §5.2), with no options:
    python -m scripts.tok_train
It is the baseline's: 32,768 tokens, trained on 2B characters of the source "fineweb", which must be built as in
configs/data/baseline.json ({"registry": "fineweb", "num_shards": 4}). Tokenizer training (CPU) is free compute. !!!
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched, load_mixture, format_mixture
from nanochat.datapipe import read_manifest

# -----------------------------------------------------------------------------
# Parse command line arguments (the options are for internal tests only: run it without options)

parser = argparse.ArgumentParser(description='Train the (fixed) BPE tokenizer: run it without options')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
parser.add_argument('--mixture', type=str, default='fineweb', help='data mixture whose sources are used to train the tokenizer')
parser.add_argument('--smoke-test', action='store_true', help=argparse.SUPPRESS) # internal tests: a small, non-standard tokenizer
args = parser.parse_args()

STANDARD = {"max_chars": 2_000_000_000, "doc_cap": 10_000, "vocab_size": 32768, "mixture": "fineweb"}
FINEWEB_SPEC = {"registry": "fineweb", "num_shards": 4} # as in configs/data/baseline.json
if not args.smoke_test:
    changed = [name for name, value in STANDARD.items() if getattr(args, name) != value]
    assert not changed, f"The tokenizer is fixed: run python -m scripts.tok_train without options (changed: {changed})"
    spec = {k: v for k, v in ((read_manifest("fineweb", "pretrain") or {}).get("spec") or {}).items() if k != "notes"}
    assert spec == FINEWEB_SPEC, (
        f"The tokenizer trains on the source 'fineweb' built as in configs/data/baseline.json ({FINEWEB_SPEC}); got "
        f"{spec or 'not built'}. Build it with: python -m scripts.prepare_data --config configs/data/baseline.json --only fineweb")
mixture = load_mixture(args.mixture)
print(f"mixture: {format_mixture(mixture)}")
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    for name, weight in mixture:
        # each source contributes characters in proportion to its mixture weight
        max_chars = weight * args.max_chars
        nchars = 0
        for batch in parquets_iter_batched(split="train", sources=[name]):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[:args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > max_chars:
                    break
            if nchars > max_chars:
                break
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    if token_id in special_ids:
        token_bytes.append(0) # special tokens are not counted
    else:
        # use the raw bytes of the token: decoding to a string first corrupts
        # tokens that are not valid standalone UTF-8 (e.g. the raw bytes >= 0x80)
        num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
        token_bytes.append(num_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")
