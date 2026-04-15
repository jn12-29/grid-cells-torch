"""Tests for GridCellsRNN model."""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from model import GridCellsRNN


def make_model(nh_lstm=16, nh_bottleneck=32, dropout_rate=0.0):
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    return GridCellsRNN(pc_ens, hdc_ens, nh_lstm=nh_lstm, nh_bottleneck=nh_bottleneck,
                        dropout_rate=dropout_rate), pc_ens, hdc_ens


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
