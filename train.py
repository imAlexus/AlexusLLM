"""
train.py — minimal training loop for the from-scratch Transformer.

Defaults are tiny so it runs on CPU in minutes and on a free Colab GPU in
seconds. Bump dim / n_layers / data size for a stronger model.

Run:
    python train.py --data data/sample.txt --epochs 5
"""
import argparse
import math
import torch
from model import TransformerLM
from tokenizer import Tokenizer, CharTokenizer
from data import load_corpus, build_dataset, get_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--save", default="checkpoints/model.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bpe", action="store_true",
                    help="use BPE tokenizer (slower to train, better quality) instead of char-level")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    text = load_corpus(args.data)
    print(f"corpus chars: {len(text):,}", flush=True)

    if args.bpe:
        tok = Tokenizer()
        tok.train(text, vocab_size=args.vocab_size, min_freq=2, verbose=True)
    else:
        tok = CharTokenizer()
        tok.train(text, verbose=True)
    print(f"vocab size: {tok.vocab_size}", flush=True)
    import os
    os.makedirs("checkpoints", exist_ok=True)
    tok.save("checkpoints/tokenizer.json")
    print("tokenizer saved, encoding corpus...", flush=True)
    data = build_dataset(text, tok, args.block_size)
    print(f"tokens: {len(data):,}", flush=True)

    model = TransformerLM(
        vocab_size=tok.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        block_size=args.block_size,
    ).to(device)
    print(f"params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    import os
    os.makedirs("checkpoints", exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        for _ in range(args.max_steps // args.epochs):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            logits, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step} | epoch {epoch} | loss {loss.item():.4f}", flush=True)
            if step >= args.max_steps:
                break
        torch.save({"model": model.state_dict(), "vocab_size": tok.vocab_size,
                    "dim": args.dim, "n_layers": args.n_layers,
                    "n_heads": args.n_heads, "block_size": args.block_size},
                   args.save)
        print(f"checkpoint saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
