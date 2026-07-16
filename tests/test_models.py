"""Tests for src.models: attention equivalence, shapes, budgets, gradients."""

import torch
import torch.nn as nn

from src.models import (
    LSTMClassifier,
    MultiHeadSelfAttention,
    TransformerClassifier,
    build_model,
    count_parameters,
)


def test_attention_matches_torch_multihead_attention() -> None:
    """From-scratch MHSA must reproduce nn.MultiheadAttention exactly."""
    torch.manual_seed(42)
    d_model, n_heads = 192, 6
    mine = MultiHeadSelfAttention(d_model, n_heads, dropout=0.0)
    ref = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)

    with torch.no_grad():
        ref.in_proj_weight.copy_(mine.in_proj.weight)
        ref.in_proj_bias.copy_(mine.in_proj.bias)
        ref.out_proj.weight.copy_(mine.out_proj.weight)
        ref.out_proj.bias.copy_(mine.out_proj.bias)

    mine.eval()
    ref.eval()
    x = torch.randn(2, 10, d_model)
    with torch.no_grad():
        out_mine = mine(x)
        out_ref, _ = ref(x, x, x, need_weights=False)

    assert out_mine.shape == out_ref.shape == (2, 10, d_model)
    assert torch.allclose(out_mine, out_ref, atol=1e-5), (
        f"max abs diff {(out_mine - out_ref).abs().max().item():.2e}"
    )


def test_forward_shapes() -> None:
    torch.manual_seed(42)
    x = torch.randn(8, 64, 315)
    for model in (TransformerClassifier(), LSTMClassifier()):
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 50)


def test_parameter_budgets() -> None:
    transformer = build_model({"arch": "transformer"})
    lstm = build_model({"arch": "lstm"})
    t_params = count_parameters(transformer)
    l_params = count_parameters(lstm)
    print(f"\nTransformer params: {t_params:,}; LSTM params: {l_params:,}")
    assert 900_000 <= t_params <= 1_500_000, f"transformer params {t_params:,} out of budget"
    rel_diff = abs(t_params - l_params) / t_params
    assert rel_diff <= 0.15, f"param mismatch |T-L|/T = {rel_diff:.3f} > 0.15"


def test_gradients_flow_to_cls_and_pos_embedding() -> None:
    torch.manual_seed(42)
    model = TransformerClassifier()
    model.train()
    x = torch.randn(4, 64, 315)
    loss = model(x).sum()
    loss.backward()
    assert model.cls_token.grad is not None
    assert model.pos_embedding.grad is not None
    assert model.cls_token.grad.abs().sum().item() > 0.0
    assert model.pos_embedding.grad.abs().sum().item() > 0.0


def test_eval_forward_deterministic() -> None:
    torch.manual_seed(42)
    x = torch.randn(4, 64, 315)
    for model in (TransformerClassifier(), LSTMClassifier()):
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2)
