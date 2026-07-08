"""
data.py — dataset helpers for conversational / reasoning training.

Expect input text files where each example is a conversation turn pair in the
format:

    USER: <user message>
    ASSISTANT: <assistant reply>

The whole corpus is concatenated with a separator so the model learns to
continue from "ASSISTANT:".

For a real project you would swap get_batch to stream from disk, but for
study-sized data this in-RAM version is fine.
"""
import torch


def load_corpus(paths):
    """Concatenate text files with a turn separator."""
    sep = "\n\n"
    parts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            parts.append(f.read())
    return sep.join(parts)


def build_dataset(text, tokenizer, block_size):
    ids = tokenizer.encode(text)
    # pad/truncate not needed: we slice windows in get_batch
    return torch.tensor(ids, dtype=torch.long)


def get_batch(data, batch_size, block_size, device):
    n = len(data)
    ix = torch.randint(0, max(1, n - block_size), (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
