"""
tokenizer.py — Tokenizer from scratch.

Two modes (--char-level in train.py = CharTokenizer, default; --bpe = BPE):

- CharTokenizer: each character = 1 token. Instant encode, works for small
  experiments, lets the GPU actually run instead of waiting for BPE training.
- Tokenizer (BPE): subword merges, smaller vocab, better for real models but
  slower to train on CPU. Use with --bpe on large datasets or Colab.

Usage:
  tok = CharTokenizer(text)         # learns vocab from text
  ids = tok.encode("ciao")
  txt = tok.decode(ids)
"""
import regex as re
from collections import Counter


class CharTokenizer:
    """Character-level tokenizer — instant, zero training."""
    def __init__(self):
        self.vocab = {}
        self._inv = {}
        self.vocab_size = 0

    def train(self, text, vocab_size=None, min_freq=None, verbose=False):
        chars = sorted(set(text))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self._inv = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        if verbose:
            print(f"char vocab: {self.vocab_size} unique chars")

    def encode(self, text):
        return [self.vocab.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self._inv.get(i, "") for i in ids)

    def save(self, path):
        import json
        json.dump({"vocab": self.vocab}, open(path, "w"), ensure_ascii=False)

    def load(self, path):
        import json
        d = json.load(open(path, encoding="utf-8"))
        self.vocab = d["vocab"]
        self._inv = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)


class Tokenizer:
    def __init__(self):
        self.vocab = {}      # token str -> id
        self.merges = {}     # (a, b) -> id
        self.pattern = re.compile(r"""[A-Za-z]+|[0-9]+|\S|\s""")

    def _get_pairs(self, tokens):
        return Counter(zip(tokens[:-1], tokens[1:]))

    def train(self, text, vocab_size=4096, min_freq=2, verbose=False):
        # 1. base vocab: every character in the text
        chars = sorted(set(text))
        for i, ch in enumerate(chars):
            self.vocab[ch] = i
        # reverse lookup available from the start (used during merges)
        self._inv = {v: k for k, v in self.vocab.items()}

        # 2. pre-tokenize into words (so merges don't cross whitespace)
        words = re.findall(self.pattern, text)
        word_ids = [list(map(self.vocab.get, list(w))) for w in words]
        # drop any word that produced a None (shouldn't happen, all chars known)
        word_ids = [w for w in word_ids if None not in w and len(w) >= 1]

        num_merges = max(0, vocab_size - len(self.vocab))
        next_id = len(self.vocab)

        # Cache: global pair -> count, plus per-word pair -> count.
        # After each merge we update only the affected pairs (incremental),
        # so training is ~O(num_merges * distinct_pairs) instead of O(N) per step.
        pair_counts = Counter()
        for ids in word_ids:
            if len(ids) >= 2:
                pair_counts.update(self._get_pairs(ids))

        for step in range(num_merges):
            # pick most frequent pair above threshold
            best = None
            best_count = min_freq
            for pair, c in pair_counts.items():
                if c > best_count:
                    best, best_count = pair, c
            if best is None:
                if verbose:
                    print(f"stop @ merge {step}: no pair >= {min_freq} freq")
                break
            a, b = best
            merged_str = self._inv[a] + self._inv[b]
            self.vocab[merged_str] = next_id
            self.merges[best] = next_id
            self._inv[next_id] = merged_str
            new_id = next_id
            next_id += 1

            # apply merge incrementally: rebuild pair_counts from scratch is too
            # slow, so we update only pairs touching `best` in each word.
            new_pair_counts = Counter()
            for ids in word_ids:
                if len(ids) < 2:
                    continue
                merged = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(ids[i])
                        i += 1
                # update global counts: remove old pairs, add new ones
                old_pairs = list(self._get_pairs(ids))
                new_pairs = list(self._get_pairs(merged))
                for p in old_pairs:
                    pair_counts[p] -= 1
                for p in new_pairs:
                    new_pair_counts[p] += 1
                ids[:] = merged
            pair_counts.update(new_pair_counts)
            if verbose and step % 200 == 0:
                print(f"merge {step}: {merged_str} (count={best_count})")

        # build reverse lookup for decoding
        self._inv = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def _id_to_str(self, idx):
        # idx may be int id; find its string
        return self._inv.get(idx, str(idx))

    def encode(self, text):
        words = re.findall(self.pattern, text)
        ids = []
        for w in words:
            # start from char ids
            if w not in self.vocab and len(w) == 1:
                # unseen char -> treat as unknown token if present, else skip
                if "<unk>" in self.vocab:
                    ids.append(self.vocab["<unk>"])
                continue
            sym = list(w)
            # greedily apply merges
            while True:
                pairs = self._get_pairs(list(map(self.vocab.get, sym))) if all(s in self.vocab for s in sym) else None
                if pairs is None:
                    break
                best_rank = None
                best_pair = None
                for pair in pairs:
                    if pair in self.merges and (best_rank is None or self.merges[pair] < best_rank):
                        best_rank, best_pair = self.merges[pair], pair
                if best_pair is None:
                    break
                a, b = best_pair
                new_sym = []
                i = 0
                while i < len(sym):
                    if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best_pair:
                        new_sym.append(a + b)
                        i += 2
                    else:
                        new_sym.append(sym[i])
                        i += 1
                sym = new_sym
            for s in sym:
                ids.append(self.vocab.get(s, self.vocab.get("<unk>", 0)))
        return ids

    def decode(self, ids):
        return "".join(self._inv.get(i, "") for i in ids)

    def save(self, path):
        import json
        json.dump({"vocab": self.vocab, "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()}},
                  open(path, "w"), ensure_ascii=False)

    def load(self, path):
        import json
        d = json.load(open(path, encoding="utf-8"))
        self.vocab = d["vocab"]
        self.merges = {tuple(k.split(",")): v for k, v in d["merges"].items()}
        self._inv = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
