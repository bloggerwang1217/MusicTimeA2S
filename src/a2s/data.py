"""
Chunked Dataset for Long Audio
==============================

Wraps a base dataset to split long pieces into overlapping chunks.
Each chunk is treated as an independent sample during training.

Design rationale:
- 8 min piece -> 3 chunks (4 min each, 2 min overlap)
- 10 min piece -> 4 chunks
- <= 4 min -> 1 chunk (original piece)

Benefits:
- Every part of long pieces gets trained
- Overlap regions learn "how to connect"
- No data wasted
- Sample count +35% approximately
"""

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.score.standardize_kern import (
    issue_sidecar_path,
    read_bar_issues,
)

logger = logging.getLogger(__name__)

# Mel frame rate. HFT_MEL: sample_rate / hop_length = 16000 / 256 = 62.5 fps.
# All chunk<->second<->measure alignment below runs at this rate. (Not imported
# from piano.foundation to avoid a src.a2s <-> src.a2s.piano circular import.)
MEL_FPS = 16000 / 256

# hFT margin (frames of context on each side of a chunk for the encoder).
# Duplicated here to avoid circular import with piano.foundation.
HFT_N_MARGIN = 32
HFT_PAD_VALUE = math.log(1e-8)
# hFT tap window: the encoder consumes whole windows, so a prefixed chunk is
# padded up to a multiple of this many frames (plus the margins).
HFT_N_FRAME = 128

def _apply_sidecar_bar_issues(parsed_bars: List[dict], kern_path: Path) -> None:
    """Merge producer-side refusals into tokenizer-local bar errors."""
    for issue in read_bar_issues(kern_path):
        ordinal = issue.get('bar_ordinal')
        label = issue.get('bar_label')
        if label == 'pickup':
            label = ''
        matches = (
            [parsed_bars[ordinal]]
            if isinstance(ordinal, int) and 0 <= ordinal < len(parsed_bars)
            else []
        )
        if not matches:
            matches = [
                bar for bar in parsed_bars
                if label is not None and bar.get('label') == label
            ]
        if not matches:
            index = issue.get('bar_index')
            if isinstance(index, int) and 0 <= index < len(parsed_bars):
                matches = [parsed_bars[index]]
        if not matches:
            message = (
                f"cannot locate producer-side issue in {kern_path}: "
                f"ordinal={ordinal!r}, label={label!r}, "
                f"index={issue.get('bar_index')!r}"
            )
            raise ValueError(message)
        message = issue.get('message') or issue.get('family') \
            or 'noncanonical producer-side bar'
        for bar in matches:
            if bar.get('error') is None:
                bar['error'] = ValueError(message)
            elif message not in str(bar['error']):
                bar['error'] = ValueError(f"{bar['error']}; {message}")

class ChunkedDataset(Dataset):

    def __init__(
        self,
        base_dataset: Dataset,
        tokenizer: Any = None,
        chunk_frames: int = 1280,
        overlap_frames: int = 640,
        max_seq_len: Optional[int] = None,
        include_clock_targets: bool = False,
    ):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self.stride = chunk_frames - overlap_frames
        self.max_seq_len = max_seq_len
        self.include_clock_targets = include_clock_targets

        self.chunks = self._load_or_create_chunks()

        self._token_cache = self._load_or_create_token_cache()

        self.oov_skipped_chunks = 0

        # Version subsampling: 1 version per piece per epoch
        self._version_groups = self._build_version_groups()
        self._active_indices: Optional[List[int]] = None
        if self._version_groups:
            self.set_epoch(0)

        n_active = len(self._active_indices) if self._active_indices is not None else len(self.chunks)
        logger.info(
            f"ChunkedDataset: {len(self.base_dataset)} pieces -> {len(self.chunks)} chunks "
            f"({n_active} active per epoch)"
        )

    def _build_version_groups(self) -> Optional[Dict[str, List[List[int]]]]:
        """Group chunk indices by kern_file, then by version."""
        base = self.base_dataset
        if not hasattr(base, 'aug_meta') or not base.aug_meta:
            return None

        # kern_file → {version → [chunk_indices]}
        groups: Dict[str, Dict[int, List[int]]] = {}
        for ci, (base_idx, *_rest) in enumerate(self.chunks):
            item = base.manifest[base_idx]
            item_id = item.get('id', '')
            aug = (
                base._get_augmentation_metadata(item_id)
                if hasattr(base, '_get_augmentation_metadata')
                else base.aug_meta.get(item_id, {})
            )
            kern_file = aug.get('kern_file', item_id)
            version = aug.get('version', 0)
            groups.setdefault(kern_file, {}).setdefault(version, []).append(ci)

        n_versions = set()
        for kern_file, ver_map in groups.items():
            n_versions.update(ver_map.keys())
        if len(n_versions) <= 1:
            return None

        # kern_file → [[chunk_indices for v0], [chunk_indices for v1], ...]
        result = {}
        for kern_file, ver_map in groups.items():
            result[kern_file] = [ver_map[v] for v in sorted(ver_map.keys())]
        return result

    def set_epoch(self, epoch: int):
        if self._version_groups is None:
            return
        rng = np.random.RandomState(epoch + 42)
        active = []
        for kern_file, versions in self._version_groups.items():
            chosen = rng.randint(len(versions))
            active.extend(versions[chosen])
        active.sort()
        self._active_indices = active

    def _chunk_list_path(self) -> Path:
        """Fixed path for the human-readable chunk list txt.
        Stored alongside the manifest so it's easy to find.
        Layout parameters belong in the key because they change chunk scopes.
        """
        manifest_path = getattr(self.base_dataset, 'manifest_path', None)
        if manifest_path:
            manifest_path = Path(manifest_path)
            return manifest_path.parent / (
                f'chunk_list_{manifest_path.stem}_{self.chunk_frames}_'
                f'{self.overlap_frames}.txt'
            )
        # Fallback: put next to this file
        return Path(__file__).resolve().parent / (
            f'chunk_list_unknown_{self.chunk_frames}_{self.overlap_frames}.txt'
        )

    def _load_or_create_chunks(self) -> List[Tuple[int, int, int, Optional[int], Optional[int]]]:
        """Load a compatible chunk layout or rebuild it from current metadata."""
        txt_path = self._chunk_list_path()
        if txt_path.exists():
            chunks = self._read_chunk_list_txt(txt_path)
            if self._chunks_match_metadata(chunks):
                logger.info(
                    f"ChunkedDataset: loaded {len(chunks)} chunks from "
                    f"{txt_path.name}"
                )
                return chunks
            logger.warning(
                f"ChunkedDataset: rebuilding incompatible chunk cache "
                f"{txt_path.name}"
            )

        chunks = self._create_chunks()
        self._write_chunk_list_txt(chunks, txt_path)
        return chunks

    def _chunks_match_metadata(self, chunks: list) -> bool:
        """Check cached scopes against the authoritative measure boundaries."""
        base = self.base_dataset
        manifest = getattr(base, 'manifest', None)
        if manifest is None:
            return True

        measures_by_piece = {}
        chunks_by_piece = {}
        for piece_idx, start, end, first_m, last_m in chunks:
            if not 0 <= piece_idx < len(manifest) or end <= start:
                return False

            item = manifest[piece_idx]
            chunks_by_piece.setdefault(piece_idx, []).append(
                (piece_idx, start, end, first_m, last_m)
            )
            if piece_idx not in measures_by_piece:
                measures_by_piece[piece_idx] = self._get_audio_measures(piece_idx)
            measures = measures_by_piece[piece_idx]
            if not measures:
                if first_m is not None or last_m is not None:
                    return False
                continue

            if (
                first_m is None
                or last_m is None
                or not 0 <= first_m <= last_m < len(measures)
            ):
                return False

            expected_start = int(measures[first_m]['start_sec'] * MEL_FPS)
            expected_end = int(measures[last_m]['end_sec'] * MEL_FPS)
            n_frames = item.get('n_frames', item.get('duration_frames'))
            if n_frames is not None:
                expected_end = min(expected_end, int(n_frames))
            if start != expected_start or end != expected_end:
                return False

        # Appending a crop-only terminal bar leaves every old boundary valid;
        # compare the layout as well so that those old scopes cannot hide it.
        for piece_idx in range(len(manifest)):
            measures = measures_by_piece.get(piece_idx)
            if measures is None:
                measures = self._get_audio_measures(piece_idx)
            if measures and not measures[-1].get('end_is_annotated', True):
                expected = self._create_bar_aligned_chunks(
                    piece_idx, self._get_audio_length(piece_idx), measures,
                )
                if chunks_by_piece.get(piece_idx, []) != expected:
                    return False

        return True

    def _get_audio_measures(self, piece_idx: int) -> list:
        """Read measure timing through the same merged view as __getitem__."""
        base = self.base_dataset
        if hasattr(base, '_get_metadata'):
            return base._get_metadata(piece_idx).get('audio_measures', [])

        manifest = getattr(base, 'manifest', None)
        if manifest is None:
            return []
        item = manifest[piece_idx]
        measures = item.get('audio_measures', [])
        item_id = item.get('id', Path(item.get('mel_path', '')).stem)
        aug_measures = getattr(base, 'aug_meta', {}).get(item_id, {}).get(
            'audio_measures', []
        )
        return aug_measures or measures

    def _chunk_layout_key(self) -> str:
        """Return a stable identity for the exact scopes consumed by tokens."""
        digest = hashlib.sha1()
        measures_by_piece = {}
        for chunk in self.chunks:
            digest.update(json.dumps(chunk, separators=(',', ':')).encode('utf-8'))
            digest.update(b'\n')
            piece_idx, _, _, _, last_m = chunk
            if piece_idx not in measures_by_piece:
                measures_by_piece[piece_idx] = self._get_audio_measures(piece_idx)
            measures = measures_by_piece[piece_idx]
            if measures and last_m == len(measures) - 1:
                extra = int(measures[-1].get('terminal_score_extra', 0))
                if extra:
                    # Identical audio bounds can now carry a longer terminal score.
                    digest.update(f'score_end:{last_m + extra}\n'.encode('utf-8'))
        return digest.hexdigest()[:10]

    # ------------------------------------------------------------------
    # Token cache: pre-tokenize all chunks once, reuse across epochs
    # ------------------------------------------------------------------

    def _token_cache_path(self) -> Path:
        ordered_vocab = sorted(
            self.tokenizer.vocab.items(), key=lambda item: item[1]
        )
        vocab_key = hashlib.sha1(
            json.dumps(ordered_vocab, separators=(',', ':')).encode('utf-8')
        ).hexdigest()[:10]
        manifest_path = getattr(self.base_dataset, 'manifest_path', None)
        if manifest_path:
            manifest_path = Path(manifest_path)
            return manifest_path.parent / (
                f'token_cache_{manifest_path.stem}_{self.chunk_frames}_'
                f'{self.overlap_frames}_{self._chunk_layout_key()}_'
                f'{vocab_key}.json'
            )
        return Path(__file__).resolve().parent / (
            f'token_cache_unknown_{self.chunk_frames}_{self.overlap_frames}_'
            f'{self._chunk_layout_key()}_{vocab_key}.json'
        )

    def _load_or_create_token_cache(self) -> Optional[List]:
        """Load or build pre-tokenized chunk data."""
        if self.tokenizer is None:
            return None

        cache_path = self._token_cache_path()
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if self._token_cache_matches_chunks(cache):
                logger.info(
                    f"ChunkedDataset: loaded token cache ({len(cache)} entries) "
                    f"from {cache_path.name}"
                )
                return cache
            logger.warning(
                f"ChunkedDataset: rebuilding incompatible token cache "
                f"{cache_path.name}"
            )

        cache = self._pre_tokenize_all()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent DDP ranks build identical content; write-then-replace
        # keeps the on-disk file complete whichever rank lands last.
        tmp_path = cache_path.with_name(f'.{cache_path.name}.{os.getpid()}')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp_path, cache_path)
        logger.info(f"ChunkedDataset: wrote token cache to {cache_path.name}")
        return cache

    def _token_cache_matches_chunks(self, cache: list) -> bool:
        if len(cache) != len(self.chunks):
            return False
        for entry, (*_scope, first_m, last_m) in zip(cache, self.chunks):
            if entry is None:
                continue
            if entry.get('first_m') != first_m or entry.get('last_m') != last_m:
                return False
        return True

    def _pre_tokenize_all(self) -> List:
        """Pre-tokenize every chunk.  Returns a list parallel to self.chunks.

        Each entry is either None (skip) or a dict:
            {tokens, chunk_kern, schema_key_ids,
             is_last_chunk, first_m, last_m}
        """
        base = self.base_dataset
        cache = [None] * len(self.chunks)
        n_skip = 0

        metadata_by_base = {}
        chunks_by_kern = {}
        for idx, chunk in enumerate(self.chunks):
            base_idx = chunk[0]
            if base_idx not in metadata_by_base:
                metadata_by_base[base_idx] = (
                    base._get_metadata(base_idx)
                    if hasattr(base, '_get_metadata') else {}
                )
            kern_path = metadata_by_base[base_idx].get('kern_path', '')
            if not kern_path:
                n_skip += 1
                continue
            chunks_by_kern.setdefault(kern_path, []).append((idx, chunk))

        logger.info(
            f"ChunkedDataset: parsing {len(chunks_by_kern)} kern_gt files "
            f"for {len(self.chunks)} chunks"
        )
        processed_chunks = 0
        n_kern = len(chunks_by_kern)
        for kern_idx, (kern_path, work) in enumerate(
            chunks_by_kern.items(), start=1,
        ):
            try:
                with open(kern_path, 'r', encoding='utf-8') as f:
                    kern_content = f.read()
                parsed_bars = self.tokenizer.parse_kern_bars_isolated(
                    kern_content)
                _apply_sidecar_bar_issues(parsed_bars, Path(kern_path))
            except (ValueError, OSError) as exc:
                if issue_sidecar_path(Path(kern_path)).exists():
                    logger.warning(f"Tokenizer skip {kern_path}: {exc}")
                else:
                    logger.debug(f"Tokenizer skip {kern_path}: {exc}")
                n_skip += len(work)
                processed_chunks += len(work)
                continue

            invalid_bars = [
                index for index, bar in enumerate(parsed_bars)
                if bar.get('error') is not None
            ]
            if invalid_bars:
                logger.info(
                    f"ChunkedDataset: {kern_path} quarantines "
                    f"{len(invalid_bars)} noncanonical bar(s); only "
                    f"overlapping chunks will be skipped"
                )

            all_lines = kern_content.splitlines(keepends=True)
            token_ids_by_scope = {}
            for idx, chunk in work:
                base_idx, start_frame, end_frame, first_m, last_m = chunk
                meta = metadata_by_base[base_idx]
                entry = self._tokenize_one_chunk(
                    idx, base_idx, start_frame, end_frame,
                    first_m, last_m, meta, meta.get('kern_measures', []),
                    kern_content=kern_content,
                    all_lines=all_lines,
                    parsed_bars=parsed_bars,
                    token_ids_by_scope=token_ids_by_scope,
                )
                cache[idx] = entry
                if entry is None:
                    n_skip += 1

            processed_chunks += len(work)
            if kern_idx % 25 == 0 or kern_idx == n_kern:
                logger.info(
                    f"ChunkedDataset: tokenized {kern_idx}/{n_kern} kern_gt "
                    f"files ({processed_chunks}/{len(self.chunks)} chunks)"
                )

        logger.info(f"ChunkedDataset: pre-tokenized {len(cache)} chunks, {n_skip} skipped")
        return cache

    def _tokenize_one_chunk(
        self, idx, base_idx, start_frame, end_frame,
        first_m, last_m, meta, kern_measures, *,
        kern_content=None, all_lines=None, parsed_bars=None,
        token_ids_by_scope=None,
    ) -> Optional[Dict]:
        from .schema import build_per_bar_key_gt

        kern_path = meta.get('kern_path', '')
        if not kern_path or self.tokenizer is None:
            return None

        # No bar info: tokenize full piece (only valid for first chunk)
        if first_m is None or last_m is None:
            if start_frame == 0:
                try:
                    if kern_content is None:
                        with open(kern_path, 'r', encoding='utf-8') as f:
                            kern_content = f.read()
                    if parsed_bars is not None:
                        tokens = self.tokenizer.encode_parsed_chunk(
                            parsed_bars, 0, len(parsed_bars) - 1,
                        )
                    else:
                        tokens = self.tokenizer.encode(kern_content)
                except (ValueError, OSError):
                    return None
                schema_key_ids = build_per_bar_key_gt(kern_content)
                return {
                    'tokens': tokens,
                    'chunk_kern': kern_content,
                    'schema_key_ids': list(schema_key_ids),
                    'is_last_chunk': True,
                    'first_m': None,
                    'last_m': None,
                }
            return None

        if not kern_measures or last_m >= len(kern_measures):
            return None

        audio_measures = meta.get('audio_measures', [])
        is_last_chunk = (last_m == len(audio_measures) - 1)
        score_last_m = last_m
        if is_last_chunk:
            score_last_m += int(audio_measures[-1].get('terminal_score_extra', 0))
        if score_last_m >= len(kern_measures):
            logger.warning(f"Score tail skip {meta.get('id', kern_path)}: score range exceeds kern")
            return None
        line_start = kern_measures[first_m]['line_start']
        line_end = kern_measures[score_last_m]['line_end']

        if all_lines is None:
            with open(kern_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
        if kern_content is None:
            kern_content = ''.join(all_lines)

        line_barline = kern_measures[first_m].get('line_barline')

        # Key and meter in effect where the chunk starts, not whatever the
        # header block happens to declare. Stopping at the first data line
        # hands a chunk no meter at all when a piece opens with unbarred music
        # ahead of its *M, and a stale one to every chunk that starts after a
        # mid-piece meter change.
        body_start = (line_barline if line_barline is not None else line_start) - 1
        key_interp = meter_interp = None
        for hl in all_lines[:body_start]:
            hs = hl.strip()
            if hs.startswith('*k['):
                key_interp = hl
            elif hs.startswith('*M') and '/' in hs:
                meter_interp = hl
        header_interps = [hl for hl in (key_interp, meter_interp) if hl is not None]

        if line_barline is not None:
            chunk_kern = ''.join(header_interps + all_lines[line_barline - 1:line_end])
        else:
            m_num = kern_measures[first_m]['measure']
            synthetic_bar = f'={m_num}\t={m_num}\n'
            chunk_kern = (
                ''.join(header_interps)
                + synthetic_bar
                + ''.join(all_lines[line_start - 1:line_end])
            )

        scope = (first_m, score_last_m)
        if token_ids_by_scope is not None and scope in token_ids_by_scope:
            tokens = token_ids_by_scope[scope]
            if tokens is None:
                return None
        else:
            try:
                # Voice topology is inherited from the complete score, while the
                # selected bars receive their own balanced token scope.
                if parsed_bars is not None:
                    tokens = self.tokenizer.encode_parsed_chunk(
                        parsed_bars, first_m, score_last_m,
                    )
                else:
                    tokens = self.tokenizer.encode_chunk(
                        kern_content, first_m, score_last_m,
                    )
                if token_ids_by_scope is not None:
                    token_ids_by_scope[scope] = tokens
            except ValueError as e:
                logger.debug(f"OOV skip chunk {idx}: {e}")
                if token_ids_by_scope is not None:
                    token_ids_by_scope[scope] = None
                return None

        schema_key_ids = build_per_bar_key_gt(chunk_kern)

        return {
            'tokens': tokens,
            'chunk_kern': chunk_kern,
            'schema_key_ids': list(schema_key_ids),
            'is_last_chunk': is_last_chunk,
            'first_m': first_m,
            'last_m': last_m,
        }

    def _write_chunk_list_txt(self, chunks: list, txt_path: Path):
        """Write a human-readable chunk list to txt_path."""
        items = getattr(self.base_dataset, 'items', None)

        txt_path.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent DDP ranks build identical content; write-then-replace
        # keeps the on-disk file complete whichever rank lands last.
        tmp_path = txt_path.with_name(f'.{txt_path.name}.{os.getpid()}')
        with open(tmp_path, 'w') as f:
            f.write(f"# chunk_list — {len(chunks)} chunks from {len(self.base_dataset)} pieces\n")
            f.write(f"# chunk_frames={self.chunk_frames}  overlap={self.overlap_frames}  "
                    f"max_seq_len={self.max_seq_len}\n")
            f.write(f"# {'idx':>5}  {'piece_idx':>9}  {'start_f':>8}  {'end_f':>8}  "
                    f"{'first_m':>7}  {'last_m':>6}  stem\n")
            for i, (pidx, start, end, fm, lm) in enumerate(chunks):
                stem = ''
                if items is not None and pidx < len(items):
                    audio_path = items[pidx].get('audio', items[pidx].get('mel', ''))
                    stem = Path(audio_path).stem
                fm_s = str(fm) if fm is not None else '-'
                lm_s = str(lm) if lm is not None else '-'
                f.write(f"  {i:>5}  {pidx:>9}  {start:>8}  {end:>8}  "
                        f"{fm_s:>7}  {lm_s:>6}  {stem}\n")
        os.replace(tmp_path, txt_path)
        logger.info(f"ChunkedDataset: wrote chunk list to {txt_path}")

    def _read_chunk_list_txt(self, txt_path: Path) -> list:
        """Read chunks back from a previously written txt file."""
        chunks = []
        with open(txt_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                pidx = int(parts[1])
                start = int(parts[2])
                end = int(parts[3])
                fm = None if parts[4] == '-' else int(parts[4])
                lm = None if parts[5] == '-' else int(parts[5])
                chunks.append((pidx, start, end, fm, lm))
        return chunks

    def _create_chunks(self) -> list:
        chunks = []
        n_bar_aligned = 0
        n_fixed = 0

        for idx in range(len(self.base_dataset)):
            length = self._get_audio_length(idx)
            audio_measures = self._get_audio_measures(idx)

            if audio_measures:
                piece_chunks = self._create_bar_aligned_chunks(
                    idx, length, audio_measures,
                )
                chunks.extend(piece_chunks)
                n_bar_aligned += len(piece_chunks)
            else:
                # Fallback: fixed-frame stride (no bar info)
                start = 0
                while start < length:
                    end = min(start + self.chunk_frames, length)
                    chunks.append((idx, start, end, None, None))
                    n_fixed += 1
                    if end >= length:
                        break
                    start += self.stride

        logger.info(
            f"ChunkedDataset: {n_bar_aligned} bar-aligned + "
            f"{n_fixed} fixed-stride chunks"
        )
        return chunks

    def _create_bar_aligned_chunks(
        self, idx: int, length: int, audio_measures: list,
    ) -> list:
        chunks = []
        stride_sec = self.stride / MEL_FPS
        n_bars = len(audio_measures)
        start_m = 0

        while start_m < n_bars:
            start_frame = int(audio_measures[start_m]['start_sec'] * MEL_FPS)
            chunk_end_sec = (start_frame + self.chunk_frames) / MEL_FPS

            # Collect complete bars within chunk_frames
            last_m = start_m
            for m in range(start_m + 1, n_bars):
                if audio_measures[m]['end_sec'] > chunk_end_sec:
                    break
                last_m = m

            end_frame = min(
                int(audio_measures[last_m]['end_sec'] * MEL_FPS),
                length,
            )
            chunks.append((idx, start_frame, end_frame, start_m, last_m))

            # Stride ≈ overlap_frames, snapped to next bar boundary
            target_next_sec = audio_measures[start_m]['start_sec'] + stride_sec
            next_m = start_m + 1
            while next_m < n_bars:
                if audio_measures[next_m]['start_sec'] >= target_next_sec:
                    break
                next_m += 1

            if next_m <= start_m:
                next_m = start_m + 1
            if next_m >= n_bars:
                # A stride may step over a short crop-only tail that did not
                # fit in this window. Keep its score in a final scope.
                if (last_m < n_bars - 1
                        and not audio_measures[-1].get('end_is_annotated', True)):
                    start_m = last_m + 1
                    continue
                break
            start_m = next_m

        return chunks

    def _get_audio_length(self, idx: int) -> int:
        """Get audio length in frames for a base dataset item.

        This requires the base dataset to support get_mel_length() or similar.
        Falls back to loading the item if not available.
        """
        # Try to get length without loading full data
        if hasattr(self.base_dataset, 'get_mel_length'):
            return self.base_dataset.get_mel_length(idx)
        elif hasattr(self.base_dataset, 'get_audio_length'):
            return self.base_dataset.get_audio_length(idx)
        else:
            # Fallback: load and check shape (less efficient)
            mel, _, _ = self.base_dataset[idx]
            return mel.shape[-1]

    def __len__(self) -> int:
        if self._active_indices is not None:
            return len(self._active_indices)
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
        if self._active_indices is not None:
            idx = self._active_indices[idx]
        base_idx, start_frame, end_frame, first_m, last_m = self.chunks[idx]

        # --- Tokens: from pre-computed cache (no kern I/O, no tokenizer) ---
        if self._token_cache is not None:
            entry = self._token_cache[idx]
            if entry is None:
                return None
            tokens = entry['tokens']
            if self.max_seq_len is not None and len(tokens) > self.max_seq_len:
                self.oov_skipped_chunks += 1
                return None
            chunk_kern = entry.get('chunk_kern')
            is_last_chunk = entry.get('is_last_chunk', True)
            first_m = entry.get('first_m')
            last_m = entry.get('last_m')
            schema_key_ids = entry.get('schema_key_ids')
        else:
            # Fallback: tokenize on the fly (no cache / no tokenizer)
            base_out = self.base_dataset[base_idx]
            if base_out is None:
                return None
            _, tokens, meta_fb = base_out
            chunk_kern = None
            is_last_chunk = True
            first_m = last_m = None
            schema_key_ids = None

        margin = HFT_N_MARGIN
        base = self.base_dataset
        # A mel that cannot be read skips the chunk; a raise here would kill
        # the whole training run from a worker.
        try:
            if hasattr(base, '_get_mel_mmap'):
                mel_mmap = base._get_mel_mmap(base_idx)
                mel_len = mel_mmap.shape[-1]
                meta = base._get_metadata(base_idx)
            else:
                mel_full, _, meta = base[base_idx]
                mel_len = mel_full.shape[-1]
        except Exception as exc:
            self.oov_skipped_chunks += 1
            logger.warning(f"Chunk skip piece {base_idx}: mel unreadable: {exc}")
            return None

        # Guard: if the chunk starts beyond the mel, the slice below is empty
        # and pad_right grows to mel-scale, silently returning a garbage chunk
        # thousands of frames long (OOMs the whole batch at the conv layer).
        # Skipped with a warning: bad data never interrupts a training run.
        if start_frame >= mel_len:
            self.oov_skipped_chunks += 1
            logger.warning(
                f"Chunk skip '{meta.get('name', '?')}': start frame "
                f"{start_frame} >= mel length {mel_len} (mel on disk is "
                f"shorter than audio_measures claims; truncated render?)"
            )
            return None

        sl_start = start_frame - margin
        sl_end = start_frame + self.chunk_frames + margin
        pad_left = max(0, -sl_start)
        pad_right = max(0, sl_end - mel_len)
        sl_start = max(0, sl_start)
        sl_end = min(mel_len, sl_end)

        if hasattr(base, '_get_mel_mmap'):
            mel_chunk = torch.from_numpy(
                np.ascontiguousarray(mel_mmap[..., sl_start:sl_end])
            )
        else:
            mel_chunk = mel_full[..., sl_start:sl_end]

        total_pad = pad_left + pad_right
        if total_pad > 0:
            mel_chunk = torch.nn.functional.pad(
                mel_chunk, (pad_left, pad_right), value=HFT_PAD_VALUE
            )

        # --- Build chunk metadata ---
        audio_measures = meta.get('audio_measures', [])
        chunk_meta = {
            **meta,
            'chunk_start_frame': start_frame,
            'chunk_end_frame': end_frame,
            'chunk_idx': idx,
            'base_idx': base_idx,
            'is_chunked': True,
            'is_last_chunk': is_last_chunk,
        }

        if audio_measures and first_m is not None and last_m is not None:
            chunk_meta['chunk_audio_measures'] = audio_measures[first_m:last_m + 1]
            chunk_meta['chunk_duration_sec'] = (end_frame - start_frame) / MEL_FPS
            if chunk_kern is not None:
                chunk_meta['chunk_kern'] = chunk_kern

        if schema_key_ids is not None:
            chunk_meta['schema_key_ids'] = schema_key_ids

        if self.include_clock_targets:
            if first_m is None or last_m is None:
                self.oov_skipped_chunks += 1
                logger.warning(
                    f"Clock target skip "
                    f"{meta.get('id', meta.get('name', '?'))} chunk "
                    f"[{start_frame}, {end_frame}): no bar-aligned scope"
                )
                return None
            from .piano.tempo_data import build_clock_targets

            n_real = max(0, min(self.chunk_frames, mel_len - start_frame))
            # Data-dependent target failures skip the chunk with a warning;
            # a raise here would kill the whole training run from a worker.
            try:
                clock_targets = build_clock_targets(
                    meta,
                    start=start_frame,
                    scope_end=end_frame,
                    capacity=self.chunk_frames,
                    n_real=n_real,
                    first_measure=first_m,
                    last_measure=last_m,
                    include_downbeat=True,
                )
            except Exception as exc:
                self.oov_skipped_chunks += 1
                logger.warning(
                    f"Clock target skip "
                    f"{meta.get('id', meta.get('name', '?'))} chunk "
                    f"[{start_frame}, {end_frame}): {exc}"
                )
                return None
            if clock_targets is None:
                self.oov_skipped_chunks += 1
                return None
            chunk_meta.update(clock_targets)

        return mel_chunk, tokens, chunk_meta

    def get_chunk_length(self, idx: int) -> int:
        """Get chunk length in frames."""
        return self.chunk_frames


class ManifestDataset(Dataset):
    """Dataset that loads from pre-computed manifest files.

    Expected manifest format (JSON):
    [
        {
            "mel_path": "mel/piece_name.npy",
            "kern_path": "path/to/kern.krn",
            "name": "piece_name",
            "duration_frames": 12345,
            ...
        },
        ...
    ]

    Mel files are stored as .npy and loaded via memory-map (mmap).
    This allows the OS page cache to handle caching across all DataLoader
    workers, and ChunkedDataset to read only the chunk slice from disk.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        tokenizer: Any,  # KernTokenizer
        max_seq_len: int = 4096,
        augmentation_metadata_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize ManifestDataset.

        Args:
            manifest_path: Path to JSON manifest file
            tokenizer: KernTokenizer instance
            max_seq_len: Maximum token sequence length
            augmentation_metadata_path: Path to augmentation_metadata.json
                (provides audio_measures and kern_measures for chunk alignment)

        Note: All paths in the manifest are resolved relative to the manifest's
        parent directory. The manifest already encodes the correct subdirectory
        prefix (e.g. "mel/foo.npy"), so separate mel_dir/kern_dir are not needed.
        """
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mel_dir = None
        self.kern_dir = None

        # mmap references are lightweight (no data in memory); cache all of them.
        # The OS page cache handles the actual data caching across workers.
        self._mel_mmaps: Dict[int, np.memmap] = {}

        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)

        # Load augmentation metadata for alignment info
        self.aug_meta: Dict[str, Any] = {}
        self._aug_meta_by_render: Dict[str, Any] = {}
        if augmentation_metadata_path:
            aug_path = Path(augmentation_metadata_path)
            if aug_path.exists():
                with open(aug_path, 'r', encoding='utf-8') as f:
                    self.aug_meta = json.load(f)
                for source_id, aug_info in self.aug_meta.items():
                    self._aug_meta_by_render[source_id] = aug_info
                    for render in aug_info.get('renders', []):
                        audio_key = render.get('audio_key')
                        if not audio_key:
                            continue
                        previous = self._aug_meta_by_render.get(audio_key)
                        if previous is not None and previous is not aug_info:
                            raise ValueError(
                                f'Duplicate augmentation metadata for {audio_key}'
                            )
                        self._aug_meta_by_render[audio_key] = aug_info
                logger.info(f"ManifestDataset: loaded augmentation metadata ({len(self.aug_meta)} entries)")
            else:
                logger.warning(f"ManifestDataset: augmentation metadata not found at {aug_path}")

        # Track which kern files have OOV (log once per file, not per access)
        self._oov_warned_kerns: set = set()

        logger.info(f"ManifestDataset: loaded {len(self.manifest)} items from {manifest_path}")

    def _get_augmentation_metadata(self, item_id: str) -> Dict[str, Any]:
        """Resolve either a source-version id or one of its rendered audio ids."""
        return self._aug_meta_by_render.get(item_id, {})

    def __len__(self) -> int:
        return len(self.manifest)

    def _get_mel_mmap(self, idx: int) -> np.ndarray:
        """Return mmap'd mel array.  Essentially free — no data is read
        until the caller slices into specific frames.

        Supports both .npy (mmap) and legacy .pt (torch.load fallback).
        """
        if idx not in self._mel_mmaps:
            item = self.manifest[idx]
            mel_path = self._resolve_path(item['mel_path'], self.mel_dir)
            if str(mel_path).endswith('.npy'):
                self._mel_mmaps[idx] = np.load(str(mel_path), mmap_mode='r')
            else:
                # Legacy .pt fallback — no mmap, full load
                self._mel_mmaps[idx] = torch.load(
                    str(mel_path), weights_only=True
                ).numpy()
        return self._mel_mmaps[idx]

    def _get_metadata(self, idx: int) -> Dict[str, Any]:
        """Build metadata dict without loading mel or kern (no I/O)."""
        item = self.manifest[idx]
        mel_path = self._resolve_path(item['mel_path'], self.mel_dir)
        kern_key = 'kern_path' if 'kern_path' in item else 'kern_gt_path'
        kern_path = self._resolve_path(item[kern_key], self.kern_dir)
        item_id = item.get('id', mel_path.stem)

        metadata = {
            'id': item_id,
            'name': item.get('name', mel_path.stem),
            'mel_path': str(mel_path),
            'kern_path': str(kern_path),
            'duration_frames': item.get('duration_frames', item.get('n_frames', 0)),
            'has_oov': False,
        }
        for key in ('audio_beats', 'audio_grid', 'audio_measures'):
            if item.get(key):
                metadata[key] = item[key]

        aug_info = self._get_augmentation_metadata(item_id)
        if aug_info:
            if aug_info.get('audio_measures'):
                metadata['audio_measures'] = aug_info['audio_measures']
            if aug_info.get('kern_measures'):
                metadata['kern_measures'] = aug_info['kern_measures']
            if aug_info.get('duration_sec'):
                metadata['duration_sec'] = aug_info['duration_sec']

        return metadata

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
        """Load mel and kern, return tokenized data.

        For standalone use (without ChunkedDataset).  When wrapped by
        ChunkedDataset, use _get_mel_mmap() + _get_metadata() directly
        to avoid loading the full mel into a tensor.
        """
        item = self.manifest[idx]
        kern_key = 'kern_path' if 'kern_path' in item else 'kern_gt_path'
        kern_path = self._resolve_path(item[kern_key], self.kern_dir)

        # Load mel via mmap, then copy to contiguous tensor
        mel_mmap = self._get_mel_mmap(idx)
        mel = torch.from_numpy(np.array(mel_mmap))

        # Load and tokenize kern
        has_oov = False
        with open(kern_path, 'r', encoding='utf-8') as f:
            kern_content = f.read()

        try:
            tokens = self.tokenizer.encode(kern_content)
        except ValueError as e:
            tokens = []
            has_oov = True
            if kern_path.name not in self._oov_warned_kerns:
                self._oov_warned_kerns.add(kern_path.name)
                logger.warning(f"ManifestDataset: {kern_path.name}: {e} "
                               f"(will attempt chunk-level recovery)")

        if tokens and len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        metadata = self._get_metadata(idx)
        metadata['has_oov'] = has_oov

        return mel, tokens, metadata

    def _resolve_path(self, path: str, base_dir: Optional[Path]) -> Path:
        """Resolve path relative to manifest's parent directory.

        Manifest paths are stored relative to the manifest's parent directory
        (e.g. "mel/foo.npy" -> manifest_parent/mel/foo.npy). base_dir overrides
        this only when files live outside the manifest directory; callers must
        not pass a base_dir that duplicates a prefix already present in path.
        """
        p = Path(path)
        if p.is_absolute():
            return p
        if base_dir:
            return Path(base_dir) / p
        return self.manifest_path.parent / p

    def get_mel_length(self, idx: int) -> int:
        """Get mel length in frames without loading."""
        item = self.manifest[idx]
        if 'n_frames' in item:
            return item['n_frames']
        if 'duration_frames' in item:
            return item['duration_frames']
        # Fallback: mmap header read (no data loaded)
        return self._get_mel_mmap(idx).shape[-1]


class PrefixedChunkedDataset(Dataset):
    """Bar-aligned chunks extended backwards by a run-in of the piece's own audio.

    Every bar-aligned chunk opens on a barline, so the phase branch learns
    "frame 0 = downbeat" and the decoder never opens its memory mid-bar,
    while a self-segmented scope opens wherever the predicted downbeat fell.
    Each item draws a run-in ``r_t`` of up to one bar (the bar before the
    chunk, capped at ``max_frames``) for the trunk and the phase branch, and
    a shorter ``r_s <= min(r_t, half of that bar)`` for the decoder memory, so
    the first downbeat inside the memory is still the token stream's first bar
    even when the phase reads at the half-bar level.  Phase targets
    extend over the run-in; the chunk's own mel, targets and tokens are left
    untouched, so a zero run-in (the first chunk of a piece) is the plain
    item.  The wrapped dataset's counters and caches stay reachable.
    """

    def __init__(
        self,
        chunked: 'ChunkedDataset',
        max_frames: int = 384,
        seed: int = 0,
    ):
        self.chunked = chunked
        self.max_frames = int(max_frames)
        self.seed = seed
        self._epoch = 0
        # The collator crops the window to the batch's own run-in, so every
        # item carries audio up to the largest possible one.
        self.window_frames = (
            -(-self.max_frames // HFT_N_FRAME) * HFT_N_FRAME
        )

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self.chunked.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.chunked)

    def __getattr__(self, name: str):
        # token cache, base dataset, skip counters: read through the wrapped dataset
        if name == 'chunked':
            raise AttributeError(name)
        return getattr(self.chunked, name)

    def _rng(self, idx: int) -> np.random.RandomState:
        key = f'{self.seed}:{self._epoch}:{idx}'.encode('utf-8')
        return np.random.RandomState(
            int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')
        )

    def __getitem__(self, idx: int):
        from .piano.tempo_data import (
            _trusted_measure_clock,
            downbeat_phase_curve,
            gt_curves,
        )

        item = self.chunked[idx]
        if item is None:
            return None
        mel, tokens, meta = item
        raw_idx = (
            self.chunked._active_indices[idx]
            if self.chunked._active_indices is not None else idx
        )
        base_idx, start_frame, _end, first_m, _last = self.chunked.chunks[raw_idx]
        measures = meta.get('audio_measures') or []
        prev_len = 0
        if first_m is not None and 0 < first_m < len(measures):
            prev_len = start_frame - int(
                measures[first_m - 1]['start_sec'] * MEL_FPS
            )
        rng = self._rng(idx)
        r_t = 0
        if prev_len > 0:
            r_t = int(rng.randint(0, min(prev_len, self.max_frames) + 1))
        r_t = min(r_t, start_frame)
        r_s = int(rng.randint(0, min(r_t, prev_len // 2) + 1))

        base = self.chunked.base_dataset
        mel_mmap = base._get_mel_mmap(base_idx)
        mel_len = mel_mmap.shape[-1]
        margin = HFT_N_MARGIN
        capacity = self.chunked.chunk_frames
        lo = start_frame - r_t - margin
        hi = start_frame - r_t + capacity + self.window_frames + margin
        window = torch.from_numpy(
            np.ascontiguousarray(mel_mmap[..., max(lo, 0):min(hi, mel_len)])
        )
        pad_left, pad_right = max(0, -lo), max(0, hi - mel_len)
        if pad_left or pad_right:
            window = torch.nn.functional.pad(
                window, (pad_left, pad_right), value=HFT_PAD_VALUE
            )
        window = window.to(mel.dtype)
        if window.dim() == 3:
            window = window.squeeze(0)

        phi = torch.zeros(r_t)
        downbeat_phi = torch.zeros(r_t)
        mask = torch.zeros(r_t, dtype=torch.bool)
        if r_t > 0:
            beats = meta.get('audio_beats') or []
            grid = meta.get('audio_grid') or beats
            # A run-in without usable annotations stays unsupervised; the
            # chunk itself is untouched, so the item never has to be skipped.
            try:
                _omega, phi, mask = gt_curves(grid, start_frame - r_t, r_t)
                downbeat_phi = downbeat_phase_curve(
                    grid, measures, beats, start_frame - r_t, r_t
                )
                mask = mask.bool()
                # The run-in stays inside the preceding measure.  Its finite
                # placeholder coordinate must not become a teaching target
                # when that measure's clock is not score-grounded.
                previous_measure = first_m - 1
                trusted = _trusted_measure_clock(measures, beats)
                if not (
                    0 <= previous_measure < len(trusted)
                    and trusted[previous_measure]
                ):
                    mask.zero_()
            except Exception as exc:
                logger.warning(
                    f"Run-in unsupervised for "
                    f"{meta.get('id', meta.get('name', '?'))} chunk at "
                    f"{start_frame}: {exc}"
                )
                phi = torch.zeros(r_t)
                downbeat_phi = torch.zeros(r_t)
                mask = torch.zeros(r_t, dtype=torch.bool)

        meta = {
            **meta,
            'prefix_frames': r_t,
            'memory_start': r_t - r_s,
            'prefix_mel': window,
            'prefix_gt_phi': phi.float(),
            'prefix_gt_downbeat_phi': downbeat_phi.float(),
            'prefix_phase_supervision_mask': mask,
        }
        return mel, tokens, meta
