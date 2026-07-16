"""Model architectures for word-level BSL classification.

Contains a from-scratch pre-LN Transformer encoder classifier (no use of
``nn.Transformer`` / ``nn.TransformerEncoderLayer`` / ``nn.MultiheadAttention``)
and a unidirectional LSTM baseline, both mapping (B, 64, 315) pose-feature
sequences to 50-class logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with manual scaled dot-product.

    Uses a single fused input projection ``in_proj: Linear(d_model, 3*d_model)``
    (rows ordered q, k, v — matching ``nn.MultiheadAttention``'s layout) and an
    output projection ``out_proj: Linear(d_model, d_model)``. Dropout is applied
    to the attention weights after softmax.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, D) -> (B, S, D)."""
        b, s, _ = x.shape
        qkv = self.in_proj(x)  # (B, S, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, S, D)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # (B, H, S, hd)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, H, S, S)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v  # (B, H, S, hd)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Pre-LN encoder layer: attention and FFN sub-blocks with residuals.

    ``x = x + drop(attn(ln1(x)))``; ``x = x + drop(ffn(ln2(x)))`` where
    ``ffn = Linear(d, d_ff) -> GELU -> dropout -> Linear(d_ff, d)``.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, D) -> (B, S, D)."""
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class TransformerClassifier(nn.Module):
    """From-scratch pre-LN Transformer encoder for sequence classification.

    Input projection -> prepend learnable CLS token -> add learned positional
    embeddings -> embedding dropout -> encoder layers -> final LayerNorm ->
    linear head on the CLS position.
    """

    def __init__(
        self,
        in_dim: int = 315,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 6,
        d_ff: int = 384,
        n_classes: int = 50,
        seq_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(in_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_layers)
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, seq_len, in_dim) -> (B, n_classes)."""
        b = x.shape[0]
        h = self.input_proj(x)  # (B, S, D)
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, D)
        h = torch.cat([cls, h], dim=1)  # (B, S+1, D)
        h = h + self.pos_embedding[:, : h.shape[1]]
        h = self.emb_dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.final_ln(h)
        return self.head(h[:, 0])


class LSTMClassifier(nn.Module):
    """Unidirectional LSTM baseline: final hidden state of last layer -> head."""

    def __init__(
        self,
        in_dim: int = 315,
        hidden: int = 256,
        layers: int = 2,
        n_classes: int = 50,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, seq_len, in_dim) -> (B, n_classes)."""
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


def build_model(cfg: dict) -> nn.Module:
    """Construct a model from a config dict; ``cfg['arch']`` selects the arch.

    All hyperparameters fall back to the fixed project defaults, so a minimal
    ``{'arch': 'transformer'}`` config works.
    """
    arch = cfg["arch"]
    in_dim = cfg.get("in_dim", 315)
    n_classes = cfg.get("n_classes", 50)
    dropout = cfg.get("dropout", 0.1)
    if arch == "transformer":
        return TransformerClassifier(
            in_dim=in_dim,
            d_model=cfg.get("d_model", 192),
            n_layers=cfg.get("n_layers", 4),
            n_heads=cfg.get("n_heads", 6),
            d_ff=cfg.get("d_ff", 384),
            n_classes=n_classes,
            seq_len=cfg.get("seq_len", 64),
            dropout=dropout,
        )
    if arch == "lstm":
        return LSTMClassifier(
            in_dim=in_dim,
            hidden=cfg.get("lstm_hidden", 256),
            layers=cfg.get("lstm_layers", 2),
            n_classes=n_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown arch: {arch!r} (expected 'transformer' or 'lstm')")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    transformer = build_model({"arch": "transformer"})
    lstm = build_model({"arch": "lstm"})
    t_params = count_parameters(transformer)
    l_params = count_parameters(lstm)
    print("Model parameter counts (trainable):")
    print(f"  TransformerClassifier: {t_params:,} ({t_params / 1e6:.2f}M)")
    print(f"  LSTMClassifier:        {l_params:,} ({l_params / 1e6:.2f}M)")
    print(f"  Ratio (LSTM/Transformer): {l_params / t_params:.3f}")
    print(f"  Relative difference |T-L|/T: {abs(t_params - l_params) / t_params:.3f}")
