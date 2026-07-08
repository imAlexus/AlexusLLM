"""
tokenizer.py — Minimal Byte-Pair Encoding (BPE) tokenizer, built from scratch.

Why BPE and not char-level?
  Char-level gives ~30-60 tokens/word and the model wastes capacity learning
  spelling instead of meaning. BPE gives subword units (like "ragion", "##are")
  so a small model can still form real words and actually converse.

This is a tiny reimplementation of the core BPE idea (similar in spirit to
Karpathy's minbpe). It learns merges on your own training text, so it adapts
to IT+EN mixed data.

Usage:
  tok = Tokenizer()
  tok.train(text, vocab_size=4096)
  ids = tok.encode("ciao come stai?")
  txt = tok.decode(ids)
"""
import regex as re
from collections import Counter


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

        # 2. pre-tokenize into words (so merges don't cross whitespace)
        words = re.findall(self.pattern, text)
        # map each word to a list of char-ids
        word_ids = [list(map(self.vocab.get, list(w))) for w in words]

        num_merges = max(0, vocab_size - len(self.vocab))
        next_id = len(self.vocab)
        for step in range(num_merges):
            # count all adjacent pairs
            pair_counts = Counter()
            for ids in word_ids:
                if len(ids) < 2:
                    continue
                pair_counts.update(self._get_pairs(ids))
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
            new_tok = self.vocab[a] if isinstance(a, str) else a  # keep ids stable
            # actually we store the merged token as a string of the two symbols
            merged_str = self._id_to_str(a) + self._id_to_str(b)
            self.vocab[merged_str] = next_id
            self.merges[best] = next_id
            next_id += 1
            # apply merge to all words
            new_word_ids = []
            for ids in word_ids:
                merged = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                        merged.append(self.merges[best])
                        i += 2
                    else:
                        merged.append(ids[i])
                        i += 1
                new_word_ids.append(merged)
            word_ids = new_word_ids
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
