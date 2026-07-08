"""
chat.py — interactive multi-turn chat with a trained AlexusLLM checkpoint.

Keeps a rolling conversation history so the model has context of previous
turns (within block_size). Run:

    python chat.py --checkpoint checkpoints/model.pt
"""
import argparse
import torch
from model import TransformerLM
from tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/model.pt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--history_turns", type=int, default=6,
                    help="how many past turns to keep as context")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    tok = Tokenizer()
    tok.load("checkpoints/tokenizer.json")

    model = TransformerLM(
        vocab_size=ckpt["vocab_size"], dim=ckpt["dim"], n_layers=ckpt["n_layers"],
        n_heads=ckpt["n_heads"], block_size=ckpt["block_size"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("=== AlexusLLM chat (scrivi 'exit' per uscire) ===")
    history = []  # list of (role, text)
    while True:
        prompt = input("TU: ").strip()
        if prompt.lower() in ("exit", "quit", "esci"):
            break
        history.append(("USER", prompt))
        # build context from recent turns
        ctx_turns = history[-args.history_turns:]
        context = "\n".join(f"{r}: {t}" for r, t in ctx_turns) + "\nASSISTANT:"
        ids = tok.encode(context)
        # truncate to block_size
        ids = ids[-ckpt["block_size"] + args.max_new_tokens:]
        x = torch.tensor(ids, dtype=torch.long)[None, :].to(device)
        with torch.no_grad():
            out = model.generate(x, args.max_new_tokens,
                                 temperature=args.temperature, top_k=args.top_k)
        reply = tok.decode(out[0].tolist()[len(ids):])
        reply = reply.split("USER:")[0].strip()
        history.append(("ASSISTANT", reply))
        print("IA:", reply)


if __name__ == "__main__":
    main()
