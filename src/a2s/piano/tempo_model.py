"""Downbeat-phase branch: dilated TCN and BiMamba stack with a circular readout."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_mamba2_time(mamba: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run Mamba2 with time dim padded to a multiple of 8.

    The Triton causal_conv1d path inside Mamba2 rejects some tail chunks whose
    time length is not 8-aligned. Pad only the sequence dimension, then trim.
    """
    T = x.shape[1]
    T_pad = ((T + 7) // 8) * 8
    if T_pad == T:
        return mamba(x)
    x_padded = F.pad(x, (0, 0, 0, T_pad - T))
    out = mamba(x_padded)
    return out[:, :T, :]


class BiMambaLayer(nn.Module):
    """Single Bi-Mamba layer: forward + backward Mamba2 with independent weights.

    BeatMamba-style topology: pre-normalize the shared input, scan it in both
    directions, merge the directional outputs with a content-dependent gate,
    then project the merged state onto the residual branch.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError(
                "mamba_ssm is required for BiMambaLayer. It must enter the "
                "environment via pyproject.toml + poetry lock (no bare pip)."
            )

        self.mamba_forward = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.mamba_backward = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv)
        self.norm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_forward = _safe_mamba2_time(self.mamba_forward, x)
        x_backward = _safe_mamba2_time(
            self.mamba_backward, x.flip(dims=[1])
        ).flip(dims=[1])
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([x_forward, x_backward], dim=-1))
        )
        x = gate * x_forward + (1.0 - gate) * x_backward
        return residual + self.dropout(self.out_proj(x))


class DilatedConvBlock(nn.Module):
    """Non-causal residual temporal convolution with spatial dropout."""

    def __init__(self, d_model: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        if dilation < 1:
            raise ValueError("dilation must be positive")
        self.dilated = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.pointwise(F.elu(self.dilated(x)))
        x = self.dropout(x).transpose(1, 2)
        return residual + x


class TempoModel(nn.Module):
    """Temporal inference stack with a circular phase readout.

    The readout is a categorical posterior over one bar cycle; no angle
    argmax or tempo-to-phase accumulator is involved.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 128,
        d_conv: int = 4,
        dropout: float = 0.1,
        n_phase_classes: int = 360,
        dilations: tuple[int, ...] = (),
        dilated_dropout: float = 0.1,
    ):
        super().__init__()
        if n_phase_classes < 2:
            raise ValueError("n_phase_classes must be at least 2")
        self.dilated_frontend = nn.ModuleList([
            DilatedConvBlock(
                d_model=d_model, dilation=d, dropout=dilated_dropout)
            for d in dilations
        ])
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.phase_head = nn.Linear(d_model, n_phase_classes)

    def forward(self, c: torch.Tensor):
        """Return the phase logits in fp32."""
        h = c
        for layer in self.dilated_frontend:
            h = layer(h)
        for layer in self.layers:
            h = layer(h)
        # Circular losses are sensitive to small readout differences; do not
        # quantize the phase heads to bf16.
        with torch.amp.autocast("cuda", enabled=False):
            h = h.float()
            return self.phase_head(h)


