import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class A2SCollator:

    def __init__(
        self,
        pad_token_id: int = 0,
        max_seq_len: int = 4096,
        pad_to_multiple: int = 32,
        score_phase: bool = False,
    ):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.pad_to_multiple = pad_to_multiple
        # Per-position bar phase of the score itself, replayed from the
        # decode grammar over the teacher-forced tokens.
        self.score_phase = bool(score_phase)
        self._phase_constraints = None

    def _score_phases(self, tokens: List[int]) -> List[float]:
        from src.a2s.piano.inference import (
            _build_decode_constraints, _initial_beam_state, _update_beam_state,
            bar_phase,
        )
        if self._phase_constraints is None:
            from src.a2s.piano.tokenizer import KernTokenizer
            self._phase_constraints = _build_decode_constraints(KernTokenizer(), 'cpu')
        state = _initial_beam_state()
        phases = []
        for tok in tokens[:-1]:
            state = _update_beam_state(state, int(tok), self._phase_constraints)
            phases.append(bar_phase(state))
        return phases

    def _prepare_batch(self, batch):
        """Drop skipped items; a bad item never takes the batch down with it."""
        return [sample for sample in batch if sample is not None]

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, List[int], Dict[str, Any]]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        batch = self._prepare_batch(batch)
        if not batch:
            return None

        mels = []
        mel_lengths = []
        input_ids_list = []
        labels_list = []
        chunk_audio_measures_list = []
        chunk_start_frames_list = []
        chunk_end_frames_list = []
        chunk_kern_list = []
        schema_key_ids_list = []
        clock_targets_list = []
        clock_keys = (
            'beat_target', 'loss_weight', 'frame_valid',
            'gt_phi', 'gt_downbeat_phi', 'gt_omega',
            'phase_supervision_mask',
            'gt_beat_position', 'gt_beat_cycle',
            'beat_position_mask',
            'beat_position_frame_mask',
            'gt_audio_bar_index', 'down_target', 'n_real_frames',
        )

        for mel, tokens, meta in batch:
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

            mels.append(mel)
            mel_lengths.append(mel.shape[-1])

            input_ids_list.append(tokens[:-1])
            labels_list.append(tokens[1:])

            chunk_audio_measures_list.append(meta.get('chunk_audio_measures', None))
            chunk_start_frames_list.append(meta.get('chunk_start_frame', 0))
            chunk_end_frames_list.append(meta.get('chunk_end_frame', mel.shape[-1]))
            chunk_kern_list.append(meta.get('chunk_kern', None))
            schema_key_ids_list.append(meta.get('schema_key_ids', None))
            clock_targets_list.append(
                {key: meta[key] for key in clock_keys if key in meta}
            )

        # Pad mels
        max_mel_len = max(mel_lengths)
        max_mel_len = ((max_mel_len + self.pad_to_multiple - 1)
                       // self.pad_to_multiple * self.pad_to_multiple)

        padded_mels = []
        for mel in mels:
            pad_len = max_mel_len - mel.shape[-1]
            if pad_len > 0:
                mel = F.pad(mel, (0, pad_len), value=0)
            padded_mels.append(mel)

        # Pad token sequences
        seq_lengths = [len(ids) for ids in labels_list]
        max_seq_len = max(seq_lengths)

        padded_input_ids = []
        padded_labels = []
        for inp, lab in zip(input_ids_list, labels_list):
            pad_len = max_seq_len - len(inp)
            padded_input_ids.append(inp + [self.pad_token_id] * pad_len)
            padded_labels.append(lab + [self.pad_token_id] * pad_len)

        result = {
            'mel': torch.stack(padded_mels),
            'mel_lengths': torch.tensor(mel_lengths, dtype=torch.long),
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            **({'score_phase': torch.tensor([
                self._score_phases(tokens)[:max_seq_len]
                + [0.0] * (max_seq_len - min(len(tokens) - 1, max_seq_len))
                for _, tokens, _ in batch
            ], dtype=torch.float32)} if self.score_phase else {}),
            'label_lengths': torch.tensor(seq_lengths, dtype=torch.long),
            'chunk_audio_measures': chunk_audio_measures_list,
            'chunk_start_frames': chunk_start_frames_list,
            'chunk_end_frames': chunk_end_frames_list,
            'chunk_kern_list': chunk_kern_list,
            'schema_key_ids_list': schema_key_ids_list,
        }

        has_clock_targets = [bool(targets) for targets in clock_targets_list]
        if any(has_clock_targets):
            if not all(has_clock_targets):
                raise ValueError(
                    'Clock-supervised and score-only samples cannot share a batch'
                )
            required = (
                'beat_target', 'loss_weight', 'frame_valid',
                'gt_phi', 'gt_downbeat_phi', 'gt_omega',
                'phase_supervision_mask',
                'gt_beat_position', 'gt_beat_cycle',
                'beat_position_mask',
                'beat_position_frame_mask',
                'gt_audio_bar_index', 'down_target', 'n_real_frames',
            )
            for key in required:
                if not all(key in targets for targets in clock_targets_list):
                    raise ValueError(f'Incomplete clock supervision: missing {key}')
            for key in required:
                values = [targets[key] for targets in clock_targets_list]
                result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) \
                    else torch.tensor(values, dtype=torch.long)

        return result


class PrefixCollator(A2SCollator):
    """A2SCollator plus the prefixed window of ``PrefixedChunkedDataset`` items.

    ``mel`` and every chunk-level target are the plain batch.  Added keys:
    ``mel_ext`` [B, 1, n_bin, R + T + 2 * margin] with R the batch's largest
    run-in rounded up to whole hFT windows (each item's window opens ``r_t``
    frames before its barline and runs on into the piece's own audio),
    ``prefix_offsets`` (``r_t``), ``memory_start`` (``r_t - r_s``), and the
    ``*_ext`` phase targets over [0, R + T): the run-in targets, then the
    chunk's own, then an unsupervised tail.
    """

    PHASE_KEYS = ('gt_phi', 'gt_downbeat_phi', 'phase_supervision_mask')

    def __call__(self, batch):
        batch = self._prepare_batch(batch)
        result = super().__call__(batch)
        if result is None:
            return result
        items = [s for s in batch if s is not None]
        if not all('prefix_mel' in meta for _, _, meta in items):
            return result
        from .data import HFT_N_FRAME, HFT_N_MARGIN

        offsets = [int(meta['prefix_frames']) for _, _, meta in items]
        T = int(result['gt_downbeat_phi'].shape[1])
        R = -(-max(offsets) // HFT_N_FRAME) * HFT_N_FRAME
        width = R + T + 2 * HFT_N_MARGIN
        mels = []
        for _, _, meta in items:
            window = meta['prefix_mel'][..., :width]
            if window.dim() == 2:
                window = window.unsqueeze(0)
            mels.append(window)
        result['mel_ext'] = torch.stack(mels)
        result['prefix_offsets'] = torch.tensor(offsets, dtype=torch.long)
        result['memory_start'] = torch.tensor(
            [int(meta['memory_start']) for _, _, meta in items], dtype=torch.long
        )
        for key in self.PHASE_KEYS:
            if key not in result:
                continue
            chunk = result[key]
            ext = chunk.new_full((chunk.shape[0], R + T), 0)
            for b, (r, (_, _, meta)) in enumerate(zip(offsets, items)):
                if r > 0 and f'prefix_{key}' in meta:
                    ext[b, :r] = meta[f'prefix_{key}'].to(chunk.dtype)
                ext[b, r:r + T] = chunk[b]
            result[f'{key}_ext'] = ext
        return result
