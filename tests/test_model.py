"""Tests for model shape, determinism, and LSTM equivalence.

Usage:
    pytest tests/test_model.py
    pytest tests/test_model.py -k forward
"""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grid_cells.cells.ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from grid_cells.cells.model import GridCellsRNN


def make_model(
    nh_lstm=16,
    nh_bottleneck=32,
    dropout_rate=0.0,
    bottleneck_has_bias=False,
    bottleneck_post_activation="none",
):
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    return GridCellsRNN(
        pc_ens,
        hdc_ens,
        nh_lstm=nh_lstm,
        nh_bottleneck=nh_bottleneck,
        dropout_rate=dropout_rate,
        bottleneck_has_bias=bottleneck_has_bias,
        bottleneck_post_activation=bottleneck_post_activation,
    ), pc_ens, hdc_ens


def _fixed_bottleneck_model(bottleneck_post_activation="none"):
    model, pc_ens, hdc_ens = make_model(
        nh_lstm=4,
        nh_bottleneck=2,
        dropout_rate=0.0,
        bottleneck_has_bias=True,
        bottleneck_post_activation=bottleneck_post_activation,
    )
    with torch.no_grad():
        model.bottleneck.weight.zero_()
        model.bottleneck.bias.copy_(torch.tensor([-1.5, 2.0]))
        model.pc_heads[0].weight.zero_()
        model.pc_heads[0].bias.zero_()
        model.pc_heads[0].weight[0, 0] = 1.0
        model.pc_heads[0].weight[1, 1] = 1.0
    return model, pc_ens, hdc_ens


def _run_fixed_bottleneck_forward(model, pc_ens, hdc_ens):
    batch, seq_len = 2, 3
    init_cond_size = sum(e.n_cells for e in pc_ens) + sum(e.n_cells for e in hdc_ens)
    init_cond = torch.zeros(batch, init_cond_size)
    velocity = torch.zeros(batch, seq_len, 3)
    return model(init_cond, velocity, training=False)


def test_forward_output_shapes():
    """Model returns tensors of expected shapes."""
    model, pc_ens, hdc_ens = make_model()
    model.eval()
    B, T = 3, 20
    init_cond_size = sum(e.n_cells for e in pc_ens) + sum(e.n_cells for e in hdc_ens)
    init_cond = torch.zeros(B, init_cond_size)
    velocity = torch.randn(B, T, 3)

    pc_logits, hdc_logits, bottleneck, lstm_acts = model(init_cond, velocity, training=False)

    assert pc_logits[0].shape == (B, T, 8)
    assert hdc_logits[0].shape == (B, T, 4)
    assert bottleneck.shape == (B, T, 32)
    assert lstm_acts.shape == (B, T, 16)


def test_forward_deterministic_with_dropout_off():
    """Same input produces same output when dropout=0."""
    model, pc_ens, hdc_ens = make_model()
    model.eval()
    B, T = 2, 10
    init_cond_size = sum(e.n_cells for e in pc_ens) + sum(e.n_cells for e in hdc_ens)
    init_cond = torch.randn(B, init_cond_size)
    velocity = torch.randn(B, T, 3)

    _, _, bn1, _ = model(init_cond, velocity, training=False)
    _, _, bn2, _ = model(init_cond, velocity, training=False)
    assert torch.allclose(bn1, bn2)


def test_default_and_none_bottleneck_post_activation_preserve_linear_output():
    """Default and explicit none should leave bottleneck linear outputs unchanged."""
    for activation in ("none",):
        model, pc_ens, hdc_ens = _fixed_bottleneck_model(activation)
        _, _, bottleneck, _ = _run_fixed_bottleneck_forward(model, pc_ens, hdc_ens)
        expected = torch.tensor([-1.5, 2.0]).reshape(1, 1, 2).expand_as(bottleneck)

        assert torch.allclose(bottleneck, expected)

    model_default, pc_ens, hdc_ens = _fixed_bottleneck_model()
    _, _, bottleneck_default, _ = _run_fixed_bottleneck_forward(
        model_default,
        pc_ens,
        hdc_ens,
    )
    expected_default = torch.tensor([-1.5, 2.0]).reshape(1, 1, 2).expand_as(
        bottleneck_default
    )
    assert torch.allclose(bottleneck_default, expected_default)


def test_relu_bottleneck_post_activation_feeds_output_heads():
    """ReLU post activation should clamp bottleneck values before output heads."""
    model, pc_ens, hdc_ens = _fixed_bottleneck_model("relu")
    pc_logits, _, bottleneck, _ = _run_fixed_bottleneck_forward(model, pc_ens, hdc_ens)
    expected = torch.tensor([0.0, 2.0]).reshape(1, 1, 2).expand_as(bottleneck)

    assert torch.allclose(bottleneck, expected)
    assert torch.allclose(pc_logits[0][..., :2], expected)


def test_tanh_bottleneck_post_activation_applies_tanh():
    """Tanh post activation should transform bottleneck linear outputs."""
    model, pc_ens, hdc_ens = _fixed_bottleneck_model("tanh")
    _, _, bottleneck, _ = _run_fixed_bottleneck_forward(model, pc_ens, hdc_ens)
    raw = torch.tensor([-1.5, 2.0]).reshape(1, 1, 2).expand_as(bottleneck)

    assert torch.allclose(bottleneck, torch.tanh(raw))


@pytest.mark.parametrize("activation", ["gelu", None])
def test_invalid_bottleneck_post_activation_rejected(activation):
    """Unsupported model-side bottleneck activations should fail fast."""
    with pytest.raises(ValueError, match="Unsupported model.bottleneck_post_activation"):
        make_model(bottleneck_post_activation=activation)


def test_lstm_numerical_equivalence():
    """nn.LSTM and LSTMCell produce identical outputs for the same weights."""
    import torch.nn as nn

    torch.manual_seed(42)
    nh_lstm, nh_bottleneck = 16, 32
    model_new, pc_ens, hdc_ens = make_model(nh_lstm=nh_lstm, nh_bottleneck=nh_bottleneck)
    model_new.eval()

    B, T = 2, 15
    init_cond_size = sum(e.n_cells for e in pc_ens) + sum(e.n_cells for e in hdc_ens)
    init_cond = torch.randn(B, init_cond_size)
    velocity = torch.randn(B, T, 3)

    # Run the new model
    _, _, bottleneck_new, lstm_new = model_new(init_cond, velocity, training=False)

    # Manually run LSTMCell with the same weights to get reference outputs
    lstm_cell = nn.LSTMCell(3, nh_lstm)
    with torch.no_grad():
        lstm_cell.weight_ih.copy_(model_new.lstm.weight_ih_l0)
        lstm_cell.weight_hh.copy_(model_new.lstm.weight_hh_l0)
        lstm_cell.bias_ih.copy_(model_new.lstm.bias_ih_l0)
        lstm_cell.bias_hh.copy_(model_new.lstm.bias_hh_l0)

    with torch.no_grad():
        h = model_new.state_init(init_cond)
        c = model_new.cell_init(init_cond)
        ref_outputs = []
        for t in range(T):
            h, c = lstm_cell(velocity[:, t, :], (h, c))
            ref_outputs.append(h)
        lstm_ref = torch.stack(ref_outputs, dim=1)

    assert torch.allclose(lstm_new, lstm_ref, atol=1e-5), \
        f"Max diff: {(lstm_new - lstm_ref).abs().max().item()}"
