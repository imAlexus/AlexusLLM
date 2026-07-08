"""
model.py — Decoder-only Transformer from scratch (PyTorch)

Educational, self-contained implementation. No HuggingFace, no fancy deps.
Components:
  - Token + positional embeddings
  - Multi-head causal self-attention
  - Pre-LayerNorm feed-forward blocks
  - Final RMSNorm + unembedding

Run:  python train.py   (see train.py for the loop)
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm — stable, cheap, used by modern LLMs."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, T, C)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, block_size):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.block_size = block_size

        # fused QKV projection
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # causal mask: lower-triangular, registered as buffer (not a param)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)  # each (B, T, C)

        # reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, n_heads, T, head_dim)

        # merge heads back: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult, bias=False),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LN transformer block."""
    def __init__(self, dim, n_heads, block_size):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, block_size)
        self.ln2 = RMSNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=6, n_heads=8, block_size=512, ffn_mult=4):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, dim))
        self.blocks = nn.ModuleList([Block(dim, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # weight tying between input embedding and output head
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"seq len {T} > block_size {self.block_size}"
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]

        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling with temperature + optional top-k."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx
