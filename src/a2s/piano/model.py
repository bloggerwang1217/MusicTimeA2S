"""
PianoModel — frozen hFT foundation + Transformer decoder.

Architecture:
    hFT tap [B, T, 88, 256]
      → CrossAttentionConverter: 32 queries attend across 256 hidden dims per note
        → [B, T, 88, 32] → flatten → Linear(2816, d_model) → [B, T, d_model]
      → Transformer decoder (cross-attend to compressed frames, AR kern output)

Decoder is a plain Transformer; default hyperparameters
(d_model=256, ff_dim=256, 8 layers, 4 heads) are taken from Alfaro-Contreras et al. 2024.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .foundation import HFT_ARCH, HFT_MEL, HFT_PAD_VALUE, build_frozen_hft
from .tokenizer import VOCAB, VOCAB_SIZE


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConformerFFN(nn.Module):
    """Half-step FFN for Conformer macaron structure (output scaled by 0.5)."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.linear2 = nn.Linear(d_model * expansion, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.dropout(F.silu(self.linear1(x)))
        x = self.dropout(self.linear2(x))
        return residual + 0.5 * x


class ConformerConvModule(nn.Module):
    """Pointwise → GLU → Depthwise(kernel=65) → BN → Swish → Pointwise."""

    def __init__(self, d_model: int, kernel_size: int = 65, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw_expand = nn.Linear(d_model, 2 * d_model)
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw_project = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.pw_expand(x)
        a, b = x.chunk(2, dim=-1)
        x = a * torch.sigmoid(b)                        # GLU
        x = self.dw_conv(x.transpose(1, 2))              # [B, D, T]
        x = F.silu(self.bn(x)).transpose(1, 2)           # [B, T, D]
        x = self.pw_project(x)
        return residual + self.dropout(x)


class ConformerMHSA(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_drop_p = dropout
        # RoPE frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("_inv_freq", inv_freq)
        # Rope tables are plain attributes, not buffers: they are a pure
        # function of T and grow per process with the longest window seen, so
        # as buffers their shapes would differ across DDP ranks and the buffer
        # broadcast would scramble every other buffer in the same bucket.
        self._rope_len = 0
        self._rope_cos = None
        self._rope_sin = None

    def _ensure_rope(self, T: int, device: torch.device):
        if (self._rope_cos is not None and T <= self._rope_len
                and self._rope_cos.device == device):
            return
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self._inv_freq.to(device))
        self._rope_cos = freqs.cos()[None, None]
        self._rope_sin = freqs.sin()[None, None]
        self._rope_len = T

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, head_dim]
        T = x.shape[2]
        cos = self._rope_cos[:, :, :T]
        sin = self._rope_sin[:, :, :T]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        self._ensure_rope(T, x.device)

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each [B, H, T, hd]
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        return residual + self.dropout(out)


class ConformerBlock(nn.Module):
    """½FFN → MHSA(RoPE) → ConvModule(kernel=65) → ½FFN → LayerNorm."""

    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 ff_expansion: int = 4, conv_kernel: int = 65,
                 dropout: float = 0.1):
        super().__init__()
        self.ffn1 = ConformerFFN(d_model, ff_expansion, dropout)
        self.mhsa = ConformerMHSA(d_model, n_heads, dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = ConformerFFN(d_model, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.final_norm(x)


# =============================================================================
# Converter
# =============================================================================


class CrossAttentionConverter(nn.Module):
    """Convert hFT tap [B, T, 88, 256] → [B, T, d_model] via cross-attention.

    Transpose note features so 256 hidden dims become the sequence axis.
    Learned queries attend across hidden dims, producing n_queries features
    per note. Flatten and project to decoder dimension.
    Uses F.scaled_dot_product_attention (flash attention compatible).
    """

    def __init__(self, hid_dim: int = 256, n_notes: int = 88,
                 n_queries: int = 32, d_model: int = 256,
                 n_heads: int = 4, head_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        inner_dim = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.queries = nn.Parameter(torch.randn(1, n_queries, inner_dim) * 0.02)
        self.k_proj = nn.Linear(n_notes, inner_dim, bias=False)
        self.v_proj = nn.Linear(n_notes, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, n_notes)
        self.norm = nn.LayerNorm(n_notes)
        self.project = nn.Linear(n_notes * n_queries, d_model)
        self.attn_drop_p = dropout

    def forward(self, tap: torch.Tensor) -> torch.Tensor:
        # tap: [B, T, 88, 256]
        B, T, N, H = tap.shape
        BT = B * T
        n_q = self.queries.size(1)

        kv = tap.reshape(BT, N, H).permute(0, 2, 1)        # [BT, 256, 88]
        q = self.queries.expand(BT, -1, -1)                  # [BT, n_queries, inner]
        k = self.k_proj(kv)                                   # [BT, 256, inner]
        v = self.v_proj(kv)                                   # [BT, 256, inner]

        q = q.view(BT, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(BT, H,   self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(BT, H,   self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )                                                      # [BT, heads, n_q, head_dim]

        out = out.transpose(1, 2).reshape(BT, n_q, -1)        # [BT, n_queries, inner]
        out = self.out_proj(out)                                # [BT, n_queries, 88]
        out = self.norm(out).permute(0, 2, 1)                  # [BT, 88, n_queries]
        out = out.reshape(B, T, -1)                            # [B, T, 88 * n_queries]
        return self.project(out)                                # [B, T, d_model]


class PianoModel(nn.Module):
    """hFT features + learned compression + AR Transformer decoder."""

    def __init__(
        self,
        hft_state_dict_path: str,
        hft_parameter_json: str = None,
        # Converter
        n_queries: int = 32,
        # Conformer bridge (0 = no conformer, backward compatible with A0)
        n_conformer_layers: int = 0,
        conformer_kernel: int = 65,
        # Decoder (default hyperparameters from Alfaro-Contreras 2024)
        d_model: int = 256,
        n_heads: int = 4,
        ff_dim: int = 256,
        n_layers: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        # Vocab
        vocab_size: int = VOCAB_SIZE,
        pad_id: int = VOCAB["<pad>"],
        device: str = "cuda",
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.bar_id = VOCAB["<bar>"]
        # Single pre-allocated causal mask; sliced per forward call
        self.register_buffer(
            "_causal_mask",
            nn.Transformer.generate_square_subsequent_mask(max_seq_len),
            persistent=False,
        )

        # --- hFT foundation (frozen) ---
        self.hft = build_frozen_hft(
            hft_state_dict_path,
            device=device,
            parameter_json=hft_parameter_json,
        )
        # Compiling a module's forward does not compile its other methods.
        hft_mod = getattr(self.hft, "_orig_mod", self.hft)
        self._hft_tap_fn = torch.compile(
            hft_mod.tap_only, dynamic=True,
            options={"emulate_precision_casts": True},
        )

        # --- Learned converter ---
        self.converter = CrossAttentionConverter(
            hid_dim=HFT_ARCH.hid_dim,
            n_notes=HFT_ARCH.n_note,
            n_queries=n_queries,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # --- Positional encoding (shared class, separate instances) ---
        self.memory_pe = SinusoidalPE(d_model, max_len=4096, dropout=dropout)

        # --- Conformer bridge (between converter and decoder) ---
        if n_conformer_layers > 0:
            self.conformer = nn.ModuleList([
                ConformerBlock(
                    d_model=d_model, n_heads=n_heads,
                    conv_kernel=conformer_kernel, dropout=dropout,
                )
                for _ in range(n_conformer_layers)
            ])
        else:
            self.conformer = None

        # --- Decoder ---
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPE(d_model, max_len=max_seq_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.hft.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen hFT features stay deterministic.
        self.hft.eval()
        return self

    def encode_frames(self, input_spec: torch.Tensor) -> torch.Tensor:
        """hFT → sliding window tap → frame representations.

        Args:
            input_spec: [B, n_bin, T_total] where T_total = N*n_frame + 2*n_margin.
                        N windows of n_frame, with n_margin context on each side.

        Returns:
            [B, N*n_frame, d_model] converter frame representations
        """
        n_frame = HFT_ARCH.n_frame   # 128
        n_margin = HFT_ARCH.n_margin  # 32
        window = n_frame + 2 * n_margin  # 192

        B, n_bin, T_total = input_spec.shape
        N = (T_total - 2 * n_margin) // n_frame

        # Batch all N windows into one hFT call instead of looping
        windows = input_spec.unfold(2, window, n_frame)     # [B, n_bin, N, window]
        windows = windows.permute(0, 2, 1, 3).contiguous()  # [B, N, n_bin, window]
        windows = windows.reshape(B * N, n_bin, window)     # [B*N, n_bin, window]

        with torch.no_grad():
            tap = self._hft_tap_fn(windows)                  # [B*N, 128, 88, 256]

        tap = tap.reshape(B, N * n_frame, HFT_ARCH.n_note, HFT_ARCH.hid_dim)
        return self.converter(tap)

    def encode(self, input_spec: torch.Tensor) -> torch.Tensor:
        """Encode audio into decoder memory for the score-only model."""
        memory = self.memory_pe(self.encode_frames(input_spec))
        if self.conformer is not None:
            for layer in self.conformer:
                memory = layer(memory)
        return memory

    def decode(
        self,
        memory: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """One decoder forward pass (teacher forcing).

        Args:
            memory: [B, T_enc, d_model] from encode()
            tgt_ids: [B, T_dec] token ids (shifted right)
            tgt_mask: [T_dec, T_dec] causal mask
            tgt_key_padding_mask: [B, T_dec] True = ignore
            memory_key_padding_mask: [B, T_enc] True = ignore

        Returns:
            [B, T_dec, vocab_size] logits
        """
        return self.output_proj(self.decode_hidden(
            memory,
            tgt_ids,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        ))

    def decode_hidden(
        self,
        memory: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        memory_moments: torch.Tensor = None,
        fourierpe_feats: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decoder hidden states before the vocabulary projection, [B, T_dec, d_model].

        ``memory_moments`` [B, T_enc, 24] / ``fourierpe_feats`` [B, T_dec, 24] are the
        audio-side and score-side bar-phase Fourier PE features; they are consumed
        only when the coordinate is delivered at the cross-attention."""
        if tgt_mask is None:
            T = tgt_ids.size(1)
            tgt_mask = self._causal_mask[:T, :T]

        if getattr(self, "coordinate_delivery", "memory") == "cross_attention":
            return self._decode_hidden_fourierpe(
                memory, tgt_ids, tgt_mask, tgt_key_padding_mask,
                memory_key_padding_mask, memory_moments, fourierpe_feats,
            )

        tgt_emb = self.pos_encoding(self.token_embedding(tgt_ids))

        return self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def _decode_hidden_fourierpe(
        self,
        memory,
        tgt_ids,
        tgt_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        memory_moments,
        fourierpe_feats,
    ):
        """Post-norm decoder layers with the bar-phase Fourier PE at the cross-attention.

        Same arithmetic as ``nn.TransformerDecoderLayer`` (norm_first=False)
        except that the audio-side PE enters the cross-attention keys and
        values and the score-side PE enters the token input; content never
        carries the coordinate, so a wrong coordinate cannot rewrite it."""
        if fourierpe_feats is None or (memory_moments is None and (
            self.fourierpe_key_proj is not None or self.fourierpe_value_proj is not None
        )):
            raise ValueError(
                "cross-attention coordinate delivery needs the audio-side and "
                "score-side Fourier PE"
            )
        x = self.token_embedding(tgt_ids)
        if self.fourierpe_token_proj is not None:
            x = x + self.fourierpe_token_proj(fourierpe_feats.to(x.dtype))
        x = self.pos_encoding(x)
        B, S, D = x.shape
        H = self.decoder.layers[0].multihead_attn.num_heads
        hd = D // H
        keep = None
        if memory_key_padding_mask is not None:
            keep = ~memory_key_padding_mask[:, None, None, :]
        for i, layer in enumerate(self.decoder.layers):
            sa_out = layer.self_attn(
                x, x, x, attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask, need_weights=False,
            )[0]
            x = layer.norm1(x + layer.dropout1(sa_out))
            ca = layer.multihead_attn
            W, b = ca.in_proj_weight, ca.in_proj_bias
            q = F.linear(x, W[:D], b[:D])
            k = F.linear(memory, W[D:2 * D], b[D:2 * D])
            v = F.linear(memory, W[2 * D:], b[2 * D:])
            if self.fourierpe_key_proj is not None:
                k = k + self.fourierpe_key_proj[i](memory_moments.to(k.dtype))
            if self.fourierpe_value_proj is not None:
                v = v + self.fourierpe_value_proj[i](memory_moments.to(v.dtype))
            T = memory.shape[1]
            q = q.view(B, S, H, hd).transpose(1, 2)
            k = k.view(B, T, H, hd).transpose(1, 2)
            v = v.view(B, T, H, hd).transpose(1, 2)
            ca_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=keep,
                dropout_p=ca.dropout if self.training else 0.0,
            )
            ca_out = ca.out_proj(ca_out.transpose(1, 2).reshape(B, S, D))
            x = layer.norm2(x + layer.dropout2(ca_out))
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm3(x + layer.dropout3(ff))
        return x

    def forward(
        self,
        input_spec: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None,
        frame_valid: torch.Tensor = None,
        prefix_offsets: torch.Tensor = None,
        memory_start: torch.Tensor = None,
        score_phase: torch.Tensor = None,
    ) -> torch.Tensor:
        """Full forward: encode + decode (teacher forcing).

        The phase-branch keyword arguments are accepted so the training loop can call
        every model variant uniformly; this base model ignores them.

        Args:
            input_spec: [B, 256, N*128 + 64] log-mel with margins
            tgt_ids: [B, T_dec] target token ids (shifted right, starts with <sos>)
            tgt_key_padding_mask: [B, T_dec] True at <pad> positions

        Returns:
            [B, T_dec, vocab_size] logits
        """
        if input_spec.dim() == 4:
            input_spec = input_spec.squeeze(1)
        memory = self.encode(input_spec)

        return self.decode(
            memory, tgt_ids,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )


class JointPianoModel(PianoModel):
    """The downbeat-phase posterior conditions the score branch."""

    def __init__(
        self,
        hft_state_dict_path: str,
        hft_parameter_json: str = None,
        n_queries: int = 32,
        n_global_layers: int = 2,
        conformer_kernel: int = 65,
        ftheta_layers: int = 6,
        ftheta_d_state: int = 128,
        ftheta_d_conv: int = 4,
        ftheta_dilations: tuple[int, ...] = (1, 1, 2, 4, 8, 16, 32),
        ftheta_dilated_dropout: float = 0.1,
        downbeat_phase_classes: int = 360,
        d_model: int = 256,
        n_heads: int = 4,
        ff_dim: int = 256,
        n_layers: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        vocab_size: int = VOCAB_SIZE,
        pad_id: int = VOCAB["<pad>"],
        coordinate_delivery: str = "memory",
        fourierpe_terms: str = "token,key,value",
        device: str = "cuda",
    ):
        if coordinate_delivery not in {"memory", "cross_attention"}:
            raise ValueError("coordinate_delivery must be 'memory' or 'cross_attention'")
        terms = {t.strip() for t in str(fourierpe_terms).split(",") if t.strip()}
        if terms - {"token", "key", "value"}:
            raise ValueError("fourierpe_terms may only name token, key, value")
        if n_global_layers < 1:
            raise ValueError("JointPianoModel requires at least one global layer")

        super().__init__(
            hft_state_dict_path=hft_state_dict_path,
            hft_parameter_json=hft_parameter_json,
            n_queries=n_queries,
            n_conformer_layers=n_global_layers,
            conformer_kernel=conformer_kernel,
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            n_layers=n_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            pad_id=pad_id,
            device=device,
        )

        # Local imports avoid a module cycle: tempo_model reuses the Conformer
        # primitives defined above.
        from .tempo_model import TempoModel

        self.tempo = TempoModel(
            d_model=d_model,
            n_layers=ftheta_layers,
            d_state=ftheta_d_state,
            d_conv=ftheta_d_conv,
            dropout=dropout,
            n_phase_classes=downbeat_phase_classes,
            dilations=ftheta_dilations,
            dilated_dropout=ftheta_dilated_dropout,
        )

        # Fourier PE at the decoder cross-attention: the audio side is the
        # posterior's twelve-harmonic expected Fourier moments, the score side
        # the same twelve harmonics of the bar phase the decode grammar tracks.
        # All projections start at zero so the step-0 model is the
        # coordinate-free one.
        self.coordinate_delivery = coordinate_delivery
        self.fourierpe_token_proj = None
        self.fourierpe_key_proj = None
        self.fourierpe_value_proj = None
        if coordinate_delivery == "cross_attention":
            harmonics = torch.arange(1, 13).float()
            theta = 2.0 * math.pi * torch.arange(downbeat_phase_classes).float() \
                / downbeat_phase_classes
            angles = theta[:, None] * harmonics[None, :]
            self.register_buffer(
                "_fourierpe_basis",
                torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1),
                persistent=False,
            )
            self.register_buffer("_fourierpe_harmonics", harmonics, persistent=False)
            n_feats = 2 * harmonics.numel()

            def zero_linear():
                layer = nn.Linear(n_feats, d_model, bias=False)
                nn.init.zeros_(layer.weight)
                return layer

            if "token" in terms:
                self.fourierpe_token_proj = zero_linear()
            if "key" in terms:
                self.fourierpe_key_proj = nn.ModuleList(
                    [zero_linear() for _ in range(n_layers)]
                )
            if "value" in terms:
                self.fourierpe_value_proj = nn.ModuleList(
                    [zero_linear() for _ in range(n_layers)]
                )

    def _global_features(self, c: torch.Tensor) -> torch.Tensor:
        features = self.memory_pe(c)
        for layer in self.conformer:
            features = layer(features)
        return features

    def _joint_memory(
        self,
        input_spec: torch.Tensor,
        memory_slices: tuple = None,
    ):
        c = self.encode_frames(input_spec)
        downbeat_phase_logits = self.tempo(c)
        memory_key_padding_mask = None
        if memory_slices is not None:
            # The trunk and the phase branch read the whole prefixed window;
            # the decoder memory is each item's own slice of it, right-padded
            # to the batch's longest slice with a key mask.
            starts, ends = memory_slices
            pieces = [
                (c[b, int(s):int(e)], downbeat_phase_logits[b, int(s):int(e)])
                for b, (s, e) in enumerate(zip(starts.tolist(), ends.tolist()))
            ]
            lengths = [piece[0].shape[0] for piece in pieces]
            longest = max(lengths)
            c = torch.stack([
                F.pad(piece[0], (0, 0, 0, longest - piece[0].shape[0]))
                for piece in pieces
            ])
            memory_phase_logits = torch.stack([
                F.pad(piece[1], (0, 0, 0, longest - piece[1].shape[0]))
                for piece in pieces
            ])
            memory_key_padding_mask = (
                torch.arange(longest, device=c.device)[None, :]
                >= torch.tensor(lengths, device=c.device)[:, None]
            )
        else:
            memory_phase_logits = downbeat_phase_logits
        memory = self._global_features(c)
        memory_moments = None
        if self.coordinate_delivery == "cross_attention":
            # Score supervision also trains the coordinate through its Fourier moments.
            probabilities = torch.softmax(memory_phase_logits.float(), dim=-1)
            memory_moments = probabilities @ self._fourierpe_basis
        return (
            memory,
            downbeat_phase_logits,
            memory_key_padding_mask,
            memory_moments,
        )

    def encode(self, input_spec: torch.Tensor) -> torch.Tensor:
        """Return score memory using the predicted coordinate."""
        return self._joint_memory(input_spec)[0]

    def encode_with_fourierpe(self, input_spec: torch.Tensor):
        """Return (memory, audio-side Fourier PE moments) for cross-attention delivery."""
        if input_spec.dim() == 4:
            input_spec = input_spec.squeeze(1)
        out = self._joint_memory(input_spec)
        return out[0], out[3]

    def score_fourierpe_features(self, score_phase: torch.Tensor) -> torch.Tensor:
        """[B, S] bar phase in radians -> [B, S, 24] Fourier features."""
        angles = score_phase.float()[..., None] * self._fourierpe_harmonics
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(
        self,
        input_spec: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None,
        frame_valid: torch.Tensor = None,
        prefix_offsets: torch.Tensor = None,
        memory_start: torch.Tensor = None,
        score_phase: torch.Tensor = None,
    ) -> dict:
        """``prefix_offsets`` / ``memory_start`` (per item, frames): ``input_spec``
        is a prefixed window whose chunk opens at ``prefix_offsets``; the trunk
        and the phase branch read the whole window, the decoder memory is the
        slice [memory_start, prefix_offsets + chunk) of it."""
        if input_spec.dim() == 4:
            input_spec = input_spec.squeeze(1)
        memory_slices = None
        if prefix_offsets is not None:
            if frame_valid is None:
                raise ValueError("prefixed windows require frame_valid")
            chunk_frames = frame_valid.shape[1]
            memory_slices = (memory_start, prefix_offsets + chunk_frames)
        (
            memory,
            downbeat_phase_logits,
            memory_key_padding_mask,
            memory_moments,
        ) = self._joint_memory(
            input_spec,
            memory_slices=memory_slices,
        )

        fourierpe_feats = None
        if self.coordinate_delivery == "cross_attention":
            if score_phase is None:
                raise ValueError("cross-attention coordinate delivery requires score_phase")
            fourierpe_feats = self.score_fourierpe_features(score_phase)
        decoder_hidden = self.decode_hidden(
            memory,
            tgt_ids,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_moments=memory_moments,
            fourierpe_feats=fourierpe_feats,
        )
        score_logits = self.output_proj(decoder_hidden)
        return {
            "score_logits": score_logits,
            "downbeat_phase_logits": downbeat_phase_logits,
        }

def build_piano_model(
    model_cfg: dict,
    vocab_size: int,
    pad_id: int,
    device: str,
) -> PianoModel:
    """Construct the score-only or direct-joint model from one model config."""
    common_args = dict(
        hft_state_dict_path=model_cfg['hft_state_dict'],
        hft_parameter_json=model_cfg.get('hft_parameter_json'),
        n_queries=model_cfg.get('n_queries', 32),
        conformer_kernel=model_cfg.get('conformer_kernel', 65),
        d_model=model_cfg.get('d_model', 256),
        n_heads=model_cfg.get('n_heads', 4),
        ff_dim=model_cfg.get('ff_dim', 256),
        n_layers=model_cfg.get('n_layers', 8),
        dropout=model_cfg.get('dropout', 0.1),
        max_seq_len=model_cfg.get('max_seq_len', 1024),
        vocab_size=vocab_size,
        pad_id=pad_id,
        device=device,
    )
    if not model_cfg.get('phase_model', False):
        return PianoModel(
            **common_args,
            n_conformer_layers=model_cfg.get('n_conformer_layers', 0),
        )
    return JointPianoModel(
        **common_args,
        n_global_layers=model_cfg.get(
            'n_global_layers', model_cfg.get('n_conformer_layers', 2)
        ),
        ftheta_layers=model_cfg.get('ftheta_layers', 6),
        ftheta_d_state=model_cfg.get('ftheta_d_state', 128),
        ftheta_d_conv=model_cfg.get('ftheta_d_conv', 4),
        ftheta_dilations=tuple(model_cfg.get(
            'ftheta_dilations', (1, 1, 2, 4, 8, 16, 32)
        )),
        ftheta_dilated_dropout=model_cfg.get('ftheta_dilated_dropout', 0.1),
        downbeat_phase_classes=model_cfg.get(
            'downbeat_phase_classes', 360
        ),
        coordinate_delivery=model_cfg.get('coordinate_delivery', 'memory'),
        fourierpe_terms=model_cfg.get(
            'fourierpe_terms', 'token,key,value'
        ),
    )
