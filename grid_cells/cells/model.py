"""Define the core recurrent model used by this repository.

`GridCellsRNN` mirrors the original DeepMind architecture: initial-condition
codes seed an LSTM, velocity drives recurrent updates, and bottleneck features
feed place-cell and head-direction-cell prediction heads.

Usage:
    from grid_cells.cells.model import GridCellsRNN
    model = GridCellsRNN(pc_ensembles, hdc_ensembles)
    pc_logits, hdc_logits, bottleneck, lstm_acts = model(init_cond, ego_vel)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GridCellsRNN(nn.Module):
    def __init__(self, pc_ensembles, hdc_ensembles, nh_lstm=128, nh_bottleneck=256,
                 dropout_rate=0.5, bottleneck_has_bias=False, init_weight_disp=0.0):
        """
        Args:
            pc_ensembles:        list of PlaceCellEnsemble
            hdc_ensembles:       list of HeadDirectionCellEnsemble
            nh_lstm:             LSTM hidden size (default 128)
            nh_bottleneck:       bottleneck linear layer output size (default 256)
            dropout_rate:        dropout probability applied after bottleneck
            bottleneck_has_bias: whether the bottleneck linear layer has bias
            init_weight_disp:    mean displacement for output head weight uniform init
        """
        super().__init__()

        # CellEnsemble is a pure-numpy helper (not nn.Module), so plain list is correct.
        # These hold no trainable parameters — only cell centers used for loss computation.
        self.pc_ensembles = pc_ensembles
        self.hdc_ensembles = hdc_ensembles
        self.nh_lstm = nh_lstm
        self.nh_bottleneck = nh_bottleneck
        self.dropout_rate = dropout_rate
        self.init_weight_disp = init_weight_disp

        # ------------------------------------------------------------------
        # Compute init_cond_size: sum of n_cells for all pc + hdc ensembles
        # ------------------------------------------------------------------
        self.init_cond_size = (
            sum(e.n_cells for e in pc_ensembles) +
            sum(e.n_cells for e in hdc_ensembles)
        )

        # ------------------------------------------------------------------
        # Initial state projections: init_cond -> (h_0, c_0)
        # Two independent linear layers, no activation
        # ------------------------------------------------------------------
        self.state_init = nn.Linear(self.init_cond_size, nh_lstm)
        self.cell_init  = nn.Linear(self.init_cond_size, nh_lstm)

        # ------------------------------------------------------------------
        # Core LSTM (cuDNN fused kernel, batch_first)
        # Input: velocity (3-dim ego velocity [vx, vy, omega])
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(input_size=3, hidden_size=nh_lstm, batch_first=True)

        # ------------------------------------------------------------------
        # Bottleneck: Linear(nh_lstm -> nh_bottleneck), no activation
        # ------------------------------------------------------------------
        self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck, bias=bottleneck_has_bias)

        # ------------------------------------------------------------------
        # Output heads: one per ensemble, no activation
        # ------------------------------------------------------------------
        self.pc_heads = nn.ModuleList([
            nn.Linear(nh_bottleneck, e.n_cells) for e in pc_ensembles
        ])
        self.hdc_heads = nn.ModuleList([
            nn.Linear(nh_bottleneck, e.n_cells) for e in hdc_ensembles
        ])

        # ------------------------------------------------------------------
        # Weight initialisation for output heads
        # ------------------------------------------------------------------
        self._init_output_heads()

    # ------------------------------------------------------------------
    # Weight initialisation helpers
    # ------------------------------------------------------------------

    def _init_output_heads(self):
        """Uniform init for output head weights: U[-bound+disp, bound+disp].

        bound = 1 / sqrt(nh_bottleneck), matching the original TF implementation.
        Biases (if any) are initialised to zero.
        """
        bound = 1.0 / math.sqrt(self.nh_bottleneck)
        disp  = self.init_weight_disp
        for head in list(self.pc_heads) + list(self.hdc_heads):
            nn.init.uniform_(head.weight, -bound + disp, bound + disp)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, init_cond, velocity, training=True):
        """
        Args:
            init_cond: torch.Tensor (batch, init_cond_size) — concatenated ensemble
                       init codes (output of get_init for all pc + hdc ensembles).
            velocity:  torch.Tensor (batch, seq_len, 3)    — ego velocity [vx, vy, omega].
            training:  bool — controls dropout (independent of self.training).

        Returns:
            pc_logits:       list of Tensor (batch, seq_len, n_cells_i), one per pc ensemble
            hdc_logits:      list of Tensor (batch, seq_len, n_cells_i), one per hdc ensemble
            bottleneck_acts: Tensor (batch, seq_len, nh_bottleneck)
            lstm_acts:       Tensor (batch, seq_len, nh_lstm)
        """
        assert init_cond.shape[1] == self.init_cond_size, (
            f"init_cond dim {init_cond.shape[1]} != expected {self.init_cond_size}")
        assert velocity.shape[2] == 3, f"velocity last dim should be 3, got {velocity.shape[2]}"
        batch_size, seq_len, _ = velocity.shape

        # ------------------------------------------------------------------
        # Initialise LSTM state from init_cond
        # ------------------------------------------------------------------
        h = self.state_init(init_cond)   # (batch, nh_lstm)
        c = self.cell_init(init_cond)    # (batch, nh_lstm)

        # ------------------------------------------------------------------
        # LSTM over full sequence (cuDNN fused kernel)
        # ------------------------------------------------------------------
        lstm_acts, _ = self.lstm(velocity, (h.unsqueeze(0), c.unsqueeze(0)))
        # lstm_acts: (B, T, nh_lstm)

        # Bottleneck + dropout over full sequence
        bottleneck_acts = self.bottleneck(lstm_acts)                         # (B, T, nh_bottleneck)
        bottleneck_acts = F.dropout(bottleneck_acts, p=self.dropout_rate, training=training)

        # ------------------------------------------------------------------
        # Per-ensemble output heads (no activation)
        # ------------------------------------------------------------------
        pc_logits  = [head(bottleneck_acts) for head in self.pc_heads]
        hdc_logits = [head(bottleneck_acts) for head in self.hdc_heads]

        return pc_logits, hdc_logits, bottleneck_acts, lstm_acts
