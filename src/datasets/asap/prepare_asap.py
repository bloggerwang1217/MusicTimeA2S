"""
ASAP Data Preparation
=====================
Converts ~/asap-dataset → data/experiments/asap_{split}/
with kern_gt + mel + manifest.

Splits:
  train           train_pieces.txt + valid_pieces.txt (156 + 17 pieces)
  test-zeng25     Zeng 25 pieces / 80 recordings  (test_asap.txt)
  test-clean      ASAP ∩ MAESTRO-test, 39 pieces / 74 recordings (hFT zero-exposure)
  test-asap102    metadata_R rows with source=ASAP × split=test, 46 pieces / 102 recordings

The first three are piece-level, aligned to the MAESTRO piece split.
test-asap102 is defined per recording, so a listed piece contributes only the
performances the list names.

Usage:
  poetry run python src/datasets/asap/prepare_asap.py --split train --workers 8
  poetry run python src/datasets/asap/prepare_asap.py --split train --dry-run
  poetry run python src/datasets/asap/prepare_asap.py --split train --phase 2
  poetry run python src/datasets/asap/prepare_asap.py --split train --phase 2.5
  poetry run python src/datasets/asap/prepare_asap.py --split train --phase 3
  poetry run python src/datasets/asap/prepare_asap.py --split test-clean
  poetry run python src/datasets/asap/prepare_asap.py --split test-zeng25
  poetry run python src/datasets/asap/prepare_asap.py --split test-asap102 --workers 8
  poetry run python src/datasets/asap/prepare_asap.py --split test-clean --skip-mel --skip-kern
"""

import argparse
import csv
import hashlib
import json
import logging
import os

# Keep nested numerical libraries inside the CPU budget implied by --workers.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "4")

import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from src.audio.render_epr import EPR_STYLES
from src.audio.reverb import draw_reverb_plan
from src.audio.render_timing import (
    attach_measure_supervision_flags,
    epr_render_fingerprint,
    epr_seconds_at,
    extract_beat_offsets,
    extract_measure_offsets,
    extract_phase_grid_offsets,
    has_epr_render_fingerprint,
    initial_qpm,
    inject_measure_markers,
    read_beat_times_from_midi,
    read_grid_times_from_midi,
    read_measure_times_from_midi,
)

ASAP_DIR: Path  # the asap-dataset checkout, set from --asap-root in main()
MAESTRO_CSV = REPO / "data/datasets/maestro-v3.0.0.csv"
TEST_LIST = REPO / "data/datasets/asap_test_set/test_asap.txt"
SPLIT_DIR = REPO / "src/datasets/asap"
# ASAP-102 metadata (metadata_R.csv of Liu et al.), taken byte-identical from
# the PM2S repository (github.com/cheriell/PM2S, under dev/metadata/).
ASAP102_METADATA_R = SPLIT_DIR / "metadata_R.csv"

# Synthesised renderings that sit next to the performance audio.
SKIP_STEMS = {"midi_score", "xml_score"}


# ── split definitions ────────────────────────────────────────────────────────

def _load_piece_list(path: Path) -> set[str]:
    with open(path) as f:
        return {l.strip() for l in f if l.strip() and l.strip() != "name"}


def _load_zeng_test_ids() -> set[str]:
    return _load_piece_list(TEST_LIST)


def _load_maestro_test_piece_ids() -> set[str]:
    """ASAP ∩ MAESTRO-test: 39 pieces whose audio is in MAESTRO test split."""
    maestro_test_audio = set()
    with open(MAESTRO_CSV) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                maestro_test_audio.add(row["audio_filename"])

    test_pieces = set()
    with open(ASAP_DIR / "metadata.csv") as f:
        for row in csv.DictReader(f):
            maestro_audio = row.get("maestro_audio_performance", "")
            if not maestro_audio:
                continue
            audio_fn = maestro_audio.replace("{maestro}/", "")
            if audio_fn in maestro_test_audio:
                folder = row["folder"]
                test_pieces.add(folder.replace("/", "#"))

    return test_pieces


def _load_asap102_recordings() -> dict[str, list[str]]:
    """ASAP-102 hold-out: rows of subset_R with source=ASAP and split=test.

    The list names recordings, not piece folders, so it is returned as
    piece_id -> performance stems and the folder contents never widen it.
    """
    from src.evaluation.asap import load_asap102_inventory

    by_piece: dict[str, list[str]] = {}
    for recording in load_asap102_inventory(ASAP102_METADATA_R, ASAP_DIR):
        by_piece.setdefault(recording["piece_id"], []).append(
            recording["performance_id"]
        )
    return {pid: sorted(stems) for pid, stems in sorted(by_piece.items())}


def load_asap_clock(piece_dir: Path, stem: str) -> tuple[list[dict], list[dict]]:
    """Read one performance's beat clock from its ASAP annotation file.

    Each line is `time \t time \t type[,meter[,key]]`, where `db` marks a bar
    start.  Beats are annotated by hand on the real performance, so unlike the
    synthetic corpus there is no render clock to read them back from.
    """
    path = piece_dir / f"{stem}_annotations.txt"
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"{path}: no annotations")
    # A pickup occupies measure 0, so measure numbers index audio_measures
    # directly in both cases.
    pickup = lines[0].split("\t")[2].split(",")[0] != "db"
    beats: list[dict] = []
    measure = 0 if pickup else -1
    beat_index = 0
    for line in lines:
        fields = line.split("\t")
        sec = round(float(fields[0]), 4)
        marker = fields[2].split(",")[0]
        is_downbeat = marker == "db"
        if is_downbeat:
            measure += 1
            beat_index = 0
        else:
            # Beats before the first bar start are a pickup; they count from 1
            # so the beat_index == 0 downbeat test never fires on them.
            beat_index += 1
        beats.append(dict(
            sec=sec,
            measure=measure,
            beat_index=beat_index,
            is_downbeat=is_downbeat,
            # `bR` is the annotators' mark for a beat they could not place
            # against the notation; its bar carries no metric supervision.
            score_derivable=marker != "bR",
        ))
    # Beat annotations alone cannot locate the end of a terminal sustain.
    # Score alignment supplies its audio crop separately when a bar remains.
    starts = [b["sec"] for b in beats if b["is_downbeat"]]
    if pickup:
        starts.insert(0, beats[0]["sec"])
    bounds = starts + [beats[-1]["sec"]]
    measures = [
        dict(start_sec=bounds[i], end_sec=bounds[i + 1])
        for i in range(len(starts))
        if bounds[i + 1] > bounds[i]
    ]
    return beats, measures


@lru_cache(maxsize=1)
def _annotations() -> dict:
    with open(ASAP_DIR / "asap_annotations.json") as f:
        return json.load(f)


def _annotated_path(piece_id: str) -> list[int] | None:
    """The score bars a performance walks, as the annotators mapped them.

    `downbeats_score_map` runs one entry per performance downbeat and names the
    measure it sits on; an entry like `8-0` is a seam where one downbeat closes
    bar 8 and opens bar 0 again.  Every performance of a folder walks the same
    path, so the first one that carries a map answers for the score.
    """
    from src.evaluation.asap import parse_downbeats_score_map

    prefix = f"{piece_id.replace('#', '/')}/"
    for key, entry in _annotations().items():
        if not key.startswith(prefix) or key.count("/") != prefix.count("/"):
            continue
        mapping = parse_downbeats_score_map(entry.get("downbeats_score_map"))
        if mapping is None:
            continue
        bars = [ordinal for group in mapping for ordinal in group]
        if bars:
            return bars
    return None


# Scores whose playback order the corpus wrote out by hand because music21's
# expander cannot follow their D.C. al Fine.  Everywhere else the corpus builds
# `downbeats_score_map` from that same expander, so reading it back would only
# return our own expansion.
HAND_WRITTEN_ORDER = (
    "Beethoven#Piano_Sonatas#11-3",
    "Beethoven#Piano_Sonatas#28-2",
    "Beethoven#Piano_Sonatas#7-3",
)


@lru_cache(maxsize=None)
def official_measure_order(piece_id: str) -> tuple[int, ...] | None:
    """The playback order the corpus wrote by hand, as 0-based measure ordinals.

    `downbeats_score_map` drops the pickup measure from its front, so the order
    is the map with ordinal 0 put back.
    """
    if piece_id not in HAND_WRITTEN_ORDER:
        return None
    path = _annotated_path(piece_id)
    return (0,) + tuple(path) if path else None


@lru_cache(maxsize=None)
def _score_measure_numbers(piece_id: str) -> tuple[int, ...] | None:
    """The first part's measure numbers, or None when they cannot be an index.

    MuseScore writes a split bar as `X1`, `X2`, ... so the number no longer
    identifies a measure; those scores keep reading their own repeat signs.
    """
    xml = ASAP_DIR / piece_id.replace("#", "/") / "xml_score.musicxml"
    body = xml.read_text(errors="ignore").split("<part ", 2)
    if len(body) < 2:
        return None
    found = re.findall(r'<measure number="([^"]+)"', body[1])
    if not all(re.fullmatch(r"-?\d+", n) for n in found):
        return None
    numbers = [int(n) for n in found]
    if len(set(numbers)) != len(numbers):
        return None
    return tuple(numbers)


def _piece_performances(piece_dir: Path) -> list[str]:
    return sorted(w.stem for w in piece_dir.glob("*.wav")
                  if w.stem not in SKIP_STEMS)


def _scan_all_pieces() -> list[dict]:
    pieces = []
    for xml in sorted(ASAP_DIR.glob("**/xml_score.musicxml")):
        rel = xml.parent.relative_to(ASAP_DIR)
        if rel.parts and rel.parts[0] == "asap_test_set":
            continue
        parts = rel.parts
        pieces.append(dict(
            piece_id="#".join(parts),
            composer=parts[0],
            xml_path=xml,
            piece_dir=xml.parent,
            performances=_piece_performances(xml.parent),
        ))
    return pieces


def select_pieces(split: str) -> tuple[list[dict], dict]:
    """Return (pieces, metadata) for the requested split.

    metadata keys: output_dir, manifests (list of (entries, filename) pairs).
    """
    all_pieces = _scan_all_pieces()
    all_by_id = {p["piece_id"]: p for p in all_pieces}

    if split == "test-zeng25":
        target_ids = _load_zeng_test_ids()
        pieces = [all_by_id[pid] for pid in sorted(target_ids) if pid in all_by_id]
        output_dir = REPO / "data/experiments/asap_test"
        # Mel only, same as the other externally grounded hold-out: the ground
        # truth is the reference pipeline's own MIDI.
        return pieces, dict(output_dir=output_dir, split_mode="test",
                            needs_kern=False, needs_clock=True)

    elif split == "test-asap102":
        recordings = _load_asap102_recordings()
        absent_pieces = [pid for pid in recordings if pid not in all_by_id]
        if absent_pieces:
            raise FileNotFoundError(
                f"ASAP-102 pieces missing from the checkout: {absent_pieces}")
        pieces = []
        for pid, stems in recordings.items():
            piece = dict(all_by_id[pid])
            # Fail loud: a hold-out row without audio would otherwise shrink
            # the set silently.
            absent = [s for s in stems if s not in piece["performances"]]
            if absent:
                raise FileNotFoundError(
                    f"{pid}: ASAP-102 recordings missing from the checkout: "
                    f"{absent}")
            piece["performances"] = stems
            pieces.append(piece)
        output_dir = REPO / "data/experiments/asap102"
        # Mel only: the ground truth this hold-out is scored against comes
        # from the reference pipeline's own MIDI, so no kern is produced here.
        return pieces, dict(output_dir=output_dir, split_mode="test",
                            needs_kern=False, needs_clock=True)

    elif split == "test-clean":
        target_ids = _load_maestro_test_piece_ids()
        pieces = [all_by_id[pid] for pid in sorted(target_ids) if pid in all_by_id]
        output_dir = REPO / "data/experiments/asap_test_clean"
        return pieces, dict(output_dir=output_dir, split_mode="test")

    elif split == "train":
        # The two lists are the frozen split: ASAP piece folders whose
        # recordings sit in MAESTRO train / validation, minus every piece
        # named by the two hold-outs (ASAP-102 and Zeng's
        # test_asap list), so training never sees a test score in any
        # performance.  A repeat variant of a hold-out movement counts as
        # the same music and is barred too.  Edit the lists, not a
        # derivation.
        train_ids = _load_piece_list(SPLIT_DIR / "train_pieces.txt")
        valid_ids = _load_piece_list(SPLIT_DIR / "valid_pieces.txt")
        target_ids = train_ids | valid_ids
        pieces = [all_by_id[pid] for pid in sorted(target_ids) if pid in all_by_id]
        output_dir = REPO / "data/experiments/asap_train"
        return pieces, dict(
            output_dir=output_dir,
            split_mode="train",
            train_ids=train_ids,
            valid_ids=valid_ids,
        )

    else:
        raise ValueError(
            f"Unknown split: {split!r}. Use train, test-zeng25, test-clean, "
            f"or test-asap102.")


def train_valid_split(
    pieces: list[dict],
    train_ids: set[str],
    valid_ids: set[str],
) -> tuple[list, list]:
    train = [p for p in pieces if p["piece_id"] in train_ids]
    valid = [p for p in pieces if p["piece_id"] in valid_ids]
    return train, valid


# ── kern conversion ──────────────────────────────────────────────────────────

def _convert_kern_worker(args):
    """xml_score.musicxml → kern/ → kern_gt/ via the shared Phase 1 / 1.5 chain.

    Phase 1 is ASAPProcessor (standardize_xml: cue promotion, sanitize, repeat
    expansion, overfull truncation, meter inference, anacrusis + renumbering,
    clean_kern_sequence); Phase 1.5 is standardize_kern.create_single_gt
    (standardize + typed errors + roundtrip gate + .diff inventory).  Both are
    imported, not copied, so ASAP cannot drift from the Syn pipeline.
    """
    xml_path, safe_id, out_root, order = args
    try:
        from src.preprocessing.asap_processor import ASAPProcessor
        from src.score.standardize_kern import create_single_gt

        out_root = Path(out_root)
        processor = ASAPProcessor(
            output_dir=out_root / "kern",
            visual_dir=out_root / "visual",
            repeat_map_dir=out_root / "repeat_map",
            # The augmentation renders read this copy, so a rendered slot's
            # bars are the bars the kern supervises.
            xml_dir=out_root / "xml",
        )
        kern_path = processor.process_piece(
            Path(xml_path), safe_id,
            measure_order=list(order) if order else None)
        if kern_path is None:
            return safe_id, "phase1_error: processing failed", ""

        return create_single_gt(
            kern_path, out_root / "kern_gt",
            out_root / "canonicalization-diffs",
        )
    except Exception as e:
        return safe_id, f"worker_error: {e}", ""


def convert_kerns(
    pieces: list[dict], output_dir: Path, workers: int = 1,
) -> dict[str, Path]:
    kern_gt_dir = output_dir / "kern_gt"
    kern_gt_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir = output_dir / "canonicalization-diffs"
    inventory_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    result_map = {}
    for p in pieces:
        safe_id = p["piece_id"].replace("#", "__").replace("/", "_")
        kern_path = kern_gt_dir / f"{safe_id}.krn"
        result_map[p["piece_id"]] = kern_path
        if not kern_path.exists():
            tasks.append((str(p["xml_path"]), safe_id, str(output_dir),
                          official_measure_order(p["piece_id"])))

    if not tasks:
        log.info("All kern files already exist, skipping conversion.")
        return result_map

    log.info(f"Converting {len(tasks)} MusicXML → kern (workers={workers})")
    statuses: dict[str, str] = {}
    families: dict[str, str] = {}
    if workers <= 1:
        for t in tqdm(tasks, desc="kern"):
            stem, status, fam = _convert_kern_worker(t)
            statuses[stem] = status
            families[stem] = fam
            if status != "success":
                log.warning(f"{stem}: {status}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_convert_kern_worker, t): t for t in tasks}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="kern"):
                stem, status, fam = fut.result()
                statuses[stem] = status
                families[stem] = fam
                if status != "success":
                    log.warning(f"{stem}: {status}")

    counts = Counter(s.partition(":")[0] for s in statuses.values())
    log.info("Kern done: %s", ", ".join(
        f"{v} {k}" for k, v in sorted(counts.items())))

    def _tsv_cell(value: str) -> str:
        return (value.replace("\t", "\\t")
                .replace("\r", "\\r")
                .replace("\n", "\\n"))

    status_path = output_dir / "canonicalization-status.tsv"
    status_path.write_text(
        "file\tstatus\tdetail\tdiff_families\n" + "".join(
            f"{_tsv_cell(stem)}\t{_tsv_cell(status.partition(':')[0])}\t"
            f"{_tsv_cell(status.partition(':')[2].strip())}\t"
            f"{_tsv_cell(families.get(stem, ''))}\n"
            for stem, status in sorted(statuses.items())
        ),
        encoding="utf-8",
    )
    return result_map


# ── mel computation ──────────────────────────────────────────────────────────


def _mel_id(piece: dict, stem: str) -> str:
    rel = piece["piece_dir"].relative_to(ASAP_DIR)
    return "__".join(rel.parts) + "__" + stem


def compute_mels(
    pieces: list[dict], output_dir: Path, workers: int = 1,
    require_alignable: bool = False,
) -> dict[str, tuple[float, int]]:
    mel_dir = output_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for p in pieces:
        for stem in p["performances"]:
            if require_alignable and not is_alignable(p["piece_id"], stem):
                log.info(f"Skipping {p['piece_id']}/{stem}: bars unalignable")
                continue
            wav = p["piece_dir"] / f"{stem}.wav"
            mel_id = _mel_id(p, stem)
            mel_path = mel_dir / f"{mel_id}.npy"
            tasks.append((mel_id, wav, mel_path))

    return _run_mel_tasks(tasks, workers)


def _run_mel_tasks(
    tasks: list[tuple[str, Path, Path]], workers: int,
) -> dict[str, tuple[float, int]]:
    from src.audio.mel import process_audio_batch
    from src.a2s.piano.foundation import HFT_MEL

    log.info(f"Computing mel for {len(tasks)} wav files (workers={workers})")
    batch = process_audio_batch(
        tasks, spec=HFT_MEL, workers=workers, description="mel")
    results: dict[str, tuple[float, int]] = {}
    failed = 0
    for mel_id, _, mel_path in tasks:
        status, shape = batch[mel_id]
        if status == "skipped":
            shape = np.load(str(mel_path), mmap_mode="r").shape
        if status in ("generated", "skipped") and shape is not None:
            n_frames = shape[-1]
            duration_sec = round(
                n_frames * HFT_MEL.hop_length / HFT_MEL.sample_rate, 4)
            results[mel_id] = (duration_sec, n_frames)
        else:
            failed += 1
            log.warning("FAIL mel %s: %s", mel_id, status)

    log.info(f"Mel done: {len(results)} ok, {failed} failed")
    return results


def compute_train_mels(
    slots: list[dict], valid_pieces: list[dict], output_dir: Path,
    workers: int,
) -> dict[str, tuple[float, int]]:
    """Compute the frozen train slots and untouched validation recordings."""
    mel_dir = output_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (slot_key(slot), slot_audio_path(slot, output_dir),
         mel_dir / f"{slot_key(slot)}.npy")
        for slot in slots
    ]
    tasks += [
        (_mel_id(piece, stem), piece["piece_dir"] / f"{stem}.wav",
         mel_dir / f"{_mel_id(piece, stem)}.npy")
        for piece in valid_pieces
        for stem in piece["performances"]
        if is_alignable(piece["piece_id"], stem)
    ]
    results = _run_mel_tasks(tasks, workers)
    if len(results) != len(tasks):
        raise RuntimeError(
            f"train mel inventory mismatch: {len(results)}/{len(tasks)}")
    return results


def load_existing_mels(
    pieces: list[dict], output_dir: Path,
) -> dict[str, tuple[float, int]]:
    return load_mel_info(
        [_mel_id(piece, stem)
         for piece in pieces for stem in piece["performances"]],
        output_dir,
    )


def load_mel_info(
    mel_ids: list[str], output_dir: Path,
) -> dict[str, tuple[float, int]]:
    from src.a2s.piano.foundation import HFT_MEL

    mel_dir = output_dir / "mel"
    results = {}
    for mel_id in mel_ids:
        mel_path = mel_dir / f"{mel_id}.npy"
        if mel_path.exists():
            mel = np.load(str(mel_path), mmap_mode="r")
            n_frames = mel.shape[-1]
            duration_sec = round(
                n_frames * HFT_MEL.hop_length / HFT_MEL.sample_rate, 4,
            )
            results[mel_id] = (duration_sec, n_frames)
    return results


# ── augmentation slots ───────────────────────────────────────────────────────

AUG_CONFIG = REPO / "src/audio/augmentation.json"
TEMPO_DEFAULTS = SPLIT_DIR / "tempo_defaults.csv"
SLOTS_PER_FAMILY = 4

# A movement family is one score plus the author's repeat realisations of it
# (`_no_repeat`, `_no_2_repeat`, `_no_first_repeat`, `_no_trio`, `_repeat`).
_VARIANT_SUFFIX = re.compile(r"_(?:no_[a-z0-9_]*(?:repeat|trio)|repeat)$")


def family_of(piece_id: str) -> str:
    return _VARIANT_SUFFIX.sub("", piece_id)


@lru_cache(maxsize=1)
def _misaligned_performances() -> frozenset[str]:
    """Performances ASAP itself could not line up with the score.

    `score_and_performance_aligned` is false where the played bar count or
    the bar positions do not follow the notation, so the bar clock cannot be
    trusted anywhere in the file.
    """
    with open(ASAP_DIR / "asap_annotations.json") as f:
        annotations = json.load(f)
    return frozenset(
        key for key, entry in annotations.items()
        if entry.get("score_and_performance_aligned") is False
    )


def is_alignable(piece_id: str, stem: str) -> bool:
    return (
        f"{piece_id.replace('#', '/')}/{stem}.mid"
        not in _misaligned_performances()
    )


@lru_cache(maxsize=None)
def _repeat_marks(piece_id: str) -> int:
    xml = ASAP_DIR / piece_id.replace("#", "/") / "xml_score.musicxml"
    return len(re.findall(r"<repeat\b", xml.read_text(errors="ignore")))


def origin_score(folders: list[str]) -> str:
    """The family member that still writes the repeats out.

    VirtuosoNet plays what the score tells it to play, so its slot reads this
    one. A family is not guaranteed to contain a folder named after itself
    (`9-2` exists only as `9-2_no_trio`), so the pick is by repeat marks, not
    by name.
    """
    return max(folders, key=lambda f: (_repeat_marks(f), f == folders[0], f))


def _stream(*parts: str) -> np.random.Generator:
    digest = hashlib.md5("|".join(parts).encode()).hexdigest()
    return np.random.default_rng(int(digest, 16) % (2 ** 32))


@lru_cache(maxsize=1)
def _performance_audio_index() -> dict[str, dict[str, bool]]:
    """piece_id -> performance stem -> whether MAESTRO carries its recording."""
    index: dict[str, dict[str, bool]] = {}
    with open(ASAP_DIR / "metadata.csv") as f:
        for row in csv.DictReader(f):
            piece_id = row["folder"].replace("/", "#")
            stem = Path(row["midi_performance"]).stem
            index.setdefault(piece_id, {})[stem] = bool(
                row.get("maestro_audio_performance"))
    return index


def select_slots(piece_ids: list[str]) -> list[dict]:
    """Four augmentation slots per movement family.

    Order: performances MAESTRO has no recording of come first (choosing one
    costs nothing at hFT pretrain), then performances with a recording, then
    VirtuosoNet fills what is left. Within a tier the original score is drawn
    from before its repeat variants, and a draw only happens where one score's
    candidates outnumber the slots still open.
    """
    with open(AUG_CONFIG) as f:
        config = json.load(f)
    soundfonts = list(config["soundfonts"]["train"])
    epr = config["epr"]
    audio_index = _performance_audio_index()

    families: dict[str, list[str]] = {}
    for piece_id in piece_ids:
        families.setdefault(family_of(piece_id), []).append(piece_id)

    slots = []
    for family, folders in sorted(families.items()):
        rng = _stream(family, "slots")
        origin = origin_score(sorted(folders))
        order = sorted(folders, key=lambda f: (f != origin, f))
        chosen: list[tuple[str, str, bool]] = []
        for has_wav in (False, True):
            for folder in order:
                open_slots = SLOTS_PER_FAMILY - len(chosen)
                if open_slots <= 0:
                    break
                candidates = sorted(
                    stem for stem, wav in audio_index.get(folder, {}).items()
                    if wav == has_wav and is_alignable(folder, stem)
                )
                if len(candidates) > open_slots:
                    keep = sorted(rng.permutation(len(candidates))[:open_slots])
                    candidates = [candidates[i] for i in keep]
                chosen += [(folder, stem, has_wav) for stem in candidates]

        composer = family.split("#")[0]
        style_rng = _stream(family, "style")
        n_vn = SLOTS_PER_FAMILY - len(chosen)
        family_slots = []
        for index, (folder, stem, has_wav) in enumerate(chosen):
            family_slots.append(dict(
                family=family, slot=index, piece_id=folder,
                source="real_wav" if has_wav else "human_midi",
                performance=stem, score=folder, style=None,
            ))
        for offset in range(n_vn):
            style = (
                composer
                if offset == 0 and composer in EPR_STYLES
                else str(style_rng.choice(epr["style_pool"]))
            )
            family_slots.append(dict(
                family=family, slot=len(chosen) + offset, piece_id=origin,
                source="vn", performance=None, score=origin, style=style,
            ))

        # Timbre rotates from a per-family offset so no soundfont is starved
        # in families whose first slots are taken by real recordings.
        rendered = [s for s in family_slots if s["source"] != "real_wav"]
        start = int(_stream(family, "timbre").integers(len(soundfonts)))
        plan = draw_reverb_plan(family, max(len(rendered), 1))
        for index, slot in enumerate(rendered):
            slot["soundfont"] = soundfonts[(start + index) % len(soundfonts)]
            slot["reverb"] = plan[index] if index < len(plan) else None
        for slot in family_slots:
            slot.setdefault("soundfont", None)
            slot.setdefault("reverb", None)
            slot.setdefault("tempo_scaling", None)
        slots += family_slots
    return _with_slot_tempos(slots, config["tempo"])


def _with_slot_tempos(slots: list[dict], tempo: dict) -> list[dict]:
    """Stratify only generated performances; recorded timing remains intact."""
    if tempo["sampling"] != "stratified_log2":
        raise ValueError(f"Unsupported tempo sampling: {tempo['sampling']!r}")
    half_width = float(tempo["max_log2_deviation"])
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("tempo.max_log2_deviation must be finite and positive")
    result = [dict(slot) for slot in slots]
    families = {}
    for slot in result:
        if slot["source"] == "vn":
            families.setdefault(slot["family"], []).append(slot)
    for family, generated in families.items():
        generated.sort(key=lambda slot: slot["slot"])
        count = len(generated)
        if tempo.get("enabled", True):
            # Each family's generated takes cover its own complete log range;
            # assignment has no shared RNG state with style, timbre or reverb.
            values = -half_width + (2.0 * half_width / count) * (
                np.arange(count) + _stream(family, "tempo-draw").random(count)
            )
            assignment = _stream(family, "tempo-assignment").permutation(count)
            log_scales = values[assignment]
            strata = assignment.tolist()
        else:
            log_scales = np.zeros(count)
            strata = [None] * count
        for slot, log_scale, stratum in zip(generated, log_scales, strata):
            slot.update(
                tempo_scaling=float(2.0 ** log_scale),
                tempo_log2=float(log_scale),
                tempo_stratum=stratum,
            )
    return result


def load_or_build_slots(piece_ids: list[str], output_dir: Path) -> list[dict]:
    """Keep selected performances and refresh only the configured tempo draws."""
    ledger = output_dir / "slot_table.json"
    if ledger.exists():
        original = ledger.read_bytes()
        slots = json.loads(original)
        with open(AUG_CONFIG) as f:
            tempo = json.load(f)["tempo"]
        updated = _with_slot_tempos(slots, tempo)
        if updated != slots:
            # Preserve the previous experiment's draw before publishing new controls.
            digest = hashlib.sha256(original).hexdigest()
            previous = ledger.with_name(f"slot_table.{digest}.json")
            if not previous.exists():
                previous.write_bytes(original)
            tmp = ledger.with_name(f".{ledger.name}")
            tmp.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
            os.replace(tmp, ledger)
            log.info("Updated %d VN tempo draws; previous table: %s",
                     sum(a != b for a, b in zip(slots, updated)), previous)
            slots = updated
        log.info(f"Slot table: {len(slots)} slots read from {ledger.name}")
        return slots
    slots = select_slots(piece_ids)
    ledger.write_text(json.dumps(slots, indent=2, ensure_ascii=False))
    log.info(f"Slot table: {len(slots)} slots written to {ledger.name}")
    return slots


# ── augmentation renders ─────────────────────────────────────────────────────

SOUNDFONT_DIR = REPO / "data/soundfonts/piano"


def slot_key(slot: dict) -> str:
    return f"{slot['family'].replace('#', '__')}__s{slot['slot']}"


def slot_audio_path(slot: dict, output_dir: Path) -> Path:
    """Where a slot's audio lives: the recording itself, or our render."""
    if slot["source"] == "real_wav":
        return ASAP_DIR / slot["piece_id"].replace("#", "/") / f"{slot['performance']}.wav"
    return output_dir / "audio_aug" / f"{slot_key(slot)}.wav"


@lru_cache(maxsize=1)
def _tempo_defaults() -> dict[str, float]:
    """Base qpm for scores that write no tempo of any kind."""
    values: dict[str, float] = {}
    with open(TEMPO_DEFAULTS) as f:
        for row in csv.DictReader(l for l in f if not l.startswith("#")):
            values[row["piece_id"]] = float(row["base_qpm"])
    return values


def _render_slot_worker(args):
    """Render one slot to wav, and for a generated one, its clock as well."""
    slot, out_root = args
    out_root = Path(out_root)
    key = slot_key(slot)
    try:
        from midi2audio import FluidSynth
        from src.audio.synthesis import (
            audio_matches_contract,
            audio_render_fingerprint,
            render_one_midi,
        )

        audio_path = out_root / "audio_aug" / f"{key}.wav"
        if slot["source"] == "human_midi":
            # The recorded performance keeps its own timing; nothing about the
            # file may move, or the annotated beats stop pointing at the audio.
            piece_dir = ASAP_DIR / slot["piece_id"].replace("#", "/")
            midi_path = piece_dir / f"{slot['performance']}.mid"
            beats, measures = load_asap_clock(piece_dir, slot["performance"])
            clock = dict(audio_beats=beats, audio_measures=measures)
        else:
            midi_path, clock = _render_vn_midi(slot, out_root, key)

        soundfont = SOUNDFONT_DIR / slot["soundfont"]
        fingerprint = audio_render_fingerprint(
            midi_path, soundfont, slot["reverb"])
        fingerprint_path = out_root / "audio_aug" / f"{key}.render.json"
        reusable = False
        if audio_matches_contract(audio_path) and fingerprint_path.exists():
            try:
                reusable = (
                    json.loads(fingerprint_path.read_text()).get("fingerprint")
                    == fingerprint
                )
            except (OSError, ValueError, TypeError):
                reusable = False
        if reusable:
            (out_root / "clock_aug" / f"{key}.json").write_text(
                json.dumps(clock), encoding="utf-8")
            return key, "skipped (fingerprint match)", clock

        fs = FluidSynth(str(soundfont), sample_rate=44100)
        render_one_midi(fs, str(midi_path), str(audio_path),
                        reverb=slot["reverb"])
        if not audio_path.exists():
            return key, "error: audio not created", None
        fingerprint_tmp = fingerprint_path.with_name(
            f".{fingerprint_path.name}")
        fingerprint_tmp.write_text(json.dumps({"fingerprint": fingerprint}))
        os.replace(fingerprint_tmp, fingerprint_path)
        (out_root / "clock_aug" / f"{key}.json").write_text(
            json.dumps(clock), encoding="utf-8")
        return key, "success", clock
    except Exception as e:
        return key, f"error: {type(e).__name__}: {e}", None


def _vn_render_recipe(slot: dict, out_root: Path):
    """Resolve the exact score and controls that identify one VN performance."""
    import music21 as m21

    xml_path = out_root / "xml" / f"{slot['score'].replace('#', '__')}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"canonical xml missing: {xml_path}")
    score = m21.converter.parse(str(xml_path))
    base_qpm = _tempo_defaults().get(slot["score"])
    if base_qpm is None:
        base_qpm = initial_qpm(score)
    qpm_primo = base_qpm * slot["tempo_scaling"]
    fingerprint = epr_render_fingerprint(
        xml_path, slot["style"], qpm_primo, interval_in_16th=1)
    return xml_path, score, base_qpm, qpm_primo, fingerprint


def _render_vn_midi(slot: dict, out_root: Path, key: str):
    """Generate the performance, then read its clock back off the markers."""
    from src.audio.render_epr import render_epr

    xml_path, score, base_qpm, qpm_primo, fingerprint = _vn_render_recipe(
        slot, out_root)

    midi_path = out_root / "midi_aug" / f"{key}.mid"
    measure_offsets = extract_measure_offsets(score)
    beat_offsets = extract_beat_offsets(score)
    if not (midi_path.exists()
            and has_epr_render_fingerprint(str(midi_path), fingerprint)):
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        # Slots may share score/style names while requesting different tempos.
        with TemporaryDirectory(prefix=f".{key}.", dir=midi_path.parent) as tmp:
            render = render_epr(
                xml_path, slot["style"], Path(tmp),
                qpm_primo=qpm_primo, interval_in_16th=1)
            inject_measure_markers(
                str(render.midi_path), measure_offsets, beat_offsets,
                seconds_at=lambda qn: epr_seconds_at(render.tempo_points, qn),
                grid_offsets=extract_phase_grid_offsets(
                    beat_offsets,
                    [point.quarter_offset for point in render.tempo_points]),
                epr_fingerprint=fingerprint)
            os.replace(render.midi_path, midi_path)

    measures = attach_measure_supervision_flags(
        read_measure_times_from_midi(str(midi_path)), measure_offsets)
    clock = dict(
        audio_measures=measures,
        audio_beats=read_beat_times_from_midi(str(midi_path)),
        audio_grid=read_grid_times_from_midi(str(midi_path)),
        duration_sec=round(measures[-1]["end_sec"], 4),
        base_qpm=base_qpm, qpm_primo=round(qpm_primo, 6),
        render_fingerprint=fingerprint,
    )
    return midi_path, clock


def render_slots(slots: list[dict], output_dir: Path, workers: int = 1) -> dict:
    """Render every slot that is not a real recording."""
    for name in ("midi_aug", "audio_aug", "clock_aug"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    tasks = [(slot, str(output_dir)) for slot in slots
             if slot["source"] != "real_wav"]
    log.info(f"Rendering {len(tasks)} slots (workers={workers})")
    results: dict[str, str] = {}
    if workers == 1:
        for task in tqdm(tasks, desc="render"):
            key, status, _ = _render_slot_worker(task)
            results[key] = status
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_render_slot_worker, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="render"):
                key, status, _ = fut.result()
                results[key] = status
    ok = sum(1 for v in results.values()
             if v in ("success", "skipped (fingerprint match)"))
    log.info(f"Rendered {ok}/{len(tasks)}")
    for key, status in sorted(results.items()):
        if status.startswith("error"):
            log.warning(f"  {key}: {status}")
    return results


# ── manifest ─────────────────────────────────────────────────────────────────

def build_manifest(
    pieces: list[dict],
    kern_map: dict[str, Path],
    mel_info: dict[str, tuple[float, int]],
    output_dir: Path,
    split_label: str,
    require_kern: bool = True,
    with_clock: bool = False,
    require_alignable: bool = False,
) -> list[dict]:
    entries = []
    kern_measures_cache = {}
    for p in pieces:
        kern_path = kern_map.get(p["piece_id"])
        if require_kern and (kern_path is None or not kern_path.exists()):
            log.warning(f"No kern for {p['piece_id']}, skipping")
            continue
        kern_rel = (kern_path.relative_to(output_dir)
                    if require_kern else None)

        for stem in p["performances"]:
            wav = p["piece_dir"] / f"{stem}.wav"
            mel_id = _mel_id(p, stem)
            if mel_id not in mel_info:
                continue
            duration_sec, n_frames = mel_info[mel_id]
            clock = {}
            if with_clock:
                beats, measures = load_asap_clock(p["piece_dir"], stem)
                if kern_path is not None:
                    _, measures, _ = _aligned_kern_measures(
                        kern_path.name, measures, beats, duration_sec,
                        output_dir, kern_measures_cache)
                clock = dict(audio_beats=beats, audio_measures=measures)
            entries.append(dict(
                id=mel_id,
                mel_path=f"mel/{mel_id}.npy",
                kern_gt_path=str(kern_rel) if kern_rel else None,
                audio_path=str(wav.relative_to(ASAP_DIR)),
                piece_id=p["piece_id"],
                performance_id=wav.stem,
                duration_sec=duration_sec,
                n_frames=n_frames,
                split=split_label,
                **clock,
            ))
    entries.sort(key=lambda x: x["id"])
    return entries


def _slot_clock(slot: dict, output_dir: Path) -> dict:
    if slot["source"] == "vn":
        path = output_dir / "clock_aug" / f"{slot_key(slot)}.json"
        if not path.exists():
            raise FileNotFoundError(f"slot clock missing: {path}")
        clock = json.loads(path.read_text())
        *_, expected = _vn_render_recipe(slot, output_dir)
        if clock.get("render_fingerprint") != expected:
            raise ValueError(f"stale slot clock: {path}; run Phase 2")
        return clock

    piece_dir = ASAP_DIR / slot["piece_id"].replace("#", "/")
    beats, measures = load_asap_clock(piece_dir, slot["performance"])
    return dict(audio_beats=beats, audio_measures=measures)


def _aligned_kern_measures(
    kern_file: str, audio_measures: list, audio_beats: list,
    duration_sec: float, output_dir: Path, cache: dict,
) -> tuple[list, list, dict]:
    """Return paired score/audio measures and their alignment status.

    ChunkedDataset pairs kern_measures[i] with audio_measures[i] by index, so
    a longer kern list is safe only when the surplus sits at the end.  Two
    accepted shapes: equal counts, or exactly one extra kern measure while the
    beat annotation ends on a downbeat before the audio does. The latter gets
    an audio crop whose end is explicitly not a metrical annotation.
    Everything else returns an empty list — the loader skips those chunks —
    with the reason recorded next to the data.
    """
    from src.score.sanitize_kern import extract_kern_measures

    if kern_file not in cache:
        cache[kern_file] = extract_kern_measures(output_dir / "kern_gt" / kern_file)
    kern_measures = cache[kern_file]
    difference = len(kern_measures) - len(audio_measures)
    if difference == 0:
        return kern_measures, audio_measures, {"status": "exact", "difference": 0}
    terminal_span = bool(
        audio_beats
        and audio_measures
        and audio_beats[-1].get("is_downbeat")
        and audio_measures[-1]["end_sec"] == audio_beats[-1]["sec"]
        and float(duration_sec) > float(audio_beats[-1]["sec"])
    )
    if difference == 1 and terminal_span:
        audio_measures = audio_measures + [{
            "start_sec": audio_beats[-1]["sec"],
            "end_sec": float(duration_sec),
            "end_is_annotated": False,
        }]
        return kern_measures, audio_measures, {
            "status": "terminal_score_only", "difference": 0,
        }
    return [], audio_measures, {
        "status": "skip",
        "reason": "unresolved_count_mismatch",
        "difference": difference,
        "terminal_span": terminal_span,
    }


def _aligned_training_clock(
    kern_file: str, clock: dict, duration_sec: float, output_dir: Path,
    cache: dict, piece_id: str, performance: str | None,
    canonical_status: str | None,
) -> tuple[list, dict, dict]:
    """Restore untimed score edges only when the annotated interior is consecutive."""
    from xml.etree import ElementTree

    import mido
    from src.evaluation.asap import parse_downbeats_score_map

    kern_measures, measures, alignment = _aligned_kern_measures(
        kern_file, clock.get("audio_measures", []),
        clock.get("audio_beats", []), duration_sec, output_dir, cache)
    aligned_clock = {**clock, "audio_measures": measures}
    if (alignment["status"] != "skip" or canonical_status != "success"
            or performance is None):
        return kern_measures, aligned_clock, alignment

    piece_dir = ASAP_DIR / piece_id.replace("#", "/")
    annotation = _annotations().get(
        f"{piece_id.replace('#', '/')}/{performance}.mid", {})
    groups = parse_downbeats_score_map(annotation.get("downbeats_score_map"))
    if (annotation.get("score_and_performance_aligned") is not True
            or not groups or any(len(group) != 1 for group in groups)):
        return kern_measures, aligned_clock, alignment
    order = [group[0] for group in groups]
    if order[0] not in (0, 1) or order != list(range(order[0], order[-1] + 1)):
        return kern_measures, aligned_clock, alignment

    # The source ordinals are usable directly only for a preserved written route.
    shape_key = ("written_route", kern_file)
    if shape_key not in cache:
        source = ElementTree.parse(piece_dir / "xml_score.musicxml").getroot()
        counts = [len(part.findall("{*}measure"))
                  for part in source.findall("{*}part")]
        repeat_map = json.loads(
            (output_dir / "repeat_map" / Path(kern_file).with_suffix(".json")).read_text())
        n_score = len(cache[kern_file])
        cache[shape_key] = bool(
            counts and all(count == n_score for count in counts)
            and not repeat_map["has_repeats"]
            and repeat_map["original_measure_count"] == n_score
            and repeat_map["expanded_measure_count"] == n_score)
    if not cache[shape_key]:
        return kern_measures, aligned_clock, alignment
    score_measures = cache[kern_file]
    beats = clock.get("audio_beats", [])
    downbeats = [beat["sec"] for beat in beats if beat["is_downbeat"]]
    if (len(order) != len(downbeats) or order[-1] >= len(score_measures)
            or downbeats != [round(float(t), 4)
                             for t in annotation["performance_downbeats"]]
            or not measures or float(duration_sec) <= beats[-1]["sec"]):
        return kern_measures, aligned_clock, alignment
    has_pickup = not beats[0]["is_downbeat"]
    if has_pickup and order[0] != 1:
        return kern_measures, aligned_clock, alignment

    measures = [dict(measure) for measure in measures]
    opening_added = order[0] == 1 and not has_pickup
    if opening_added:
        first_sound = None
        seconds = 0.0
        for message in mido.MidiFile(piece_dir / f"{performance}.mid"):
            seconds += message.time
            if message.type == "note_on" and message.velocity > 0:
                first_sound = round(seconds, 4)
                break
        if first_sound is None or not 0 <= first_sound < downbeats[0]:
            return kern_measures, aligned_clock, alignment
        measures.insert(0, {
            "start_sec": first_sound, "end_sec": downbeats[0],
            "start_is_annotated": False,
        })
        beats = [{**beat, "measure": beat["measure"] + 1} for beat in beats]
    if beats[-1]["is_downbeat"]:
        measures.append({"start_sec": beats[-1]["sec"], "end_sec": float(duration_sec)})
    else:
        measures[-1]["end_sec"] = float(duration_sec)
    measures[-1]["end_is_annotated"] = False
    if len(measures) != order[-1] + 1:
        return kern_measures, aligned_clock, alignment
    extra = len(score_measures) - len(measures)
    if extra:
        # The remaining score stays in the terminal scope without invented bar times.
        measures[-1]["terminal_score_extra"] = extra
    return score_measures, {**clock, "audio_beats": beats, "audio_measures": measures}, {
        "status": "score_edges", "difference": extra,
        "opening_score_only": opening_added, "terminal_score_extra": extra,
        "mapping_source": "official_consecutive_written_route",
    }


def build_train_metadata(
    slots: list[dict], valid_pieces: list[dict],
    mel_info: dict[str, tuple[float, int]], output_dir: Path,
) -> dict:
    """Describe train slots and untouched validation performances uniformly."""
    metadata = {}
    kern_measures_cache: dict = {}
    with (output_dir / "canonicalization-status.tsv").open() as handle:
        canonical_status = {
            row["file"]: row["status"] for row in csv.DictReader(handle, delimiter="\t")
        }
    for slot in slots:
        key = slot_key(slot)
        if key not in mel_info:
            raise FileNotFoundError(f"slot mel missing: {key}")
        kern_file = f"{slot['piece_id'].replace('#', '__')}.krn"
        if not (output_dir / "kern_gt" / kern_file).exists():
            raise FileNotFoundError(f"slot kern missing: {kern_file}")
        duration_sec, n_frames = mel_info[key]
        clock = _slot_clock(slot, output_dir)
        kern_measures, clock, alignment = _aligned_training_clock(
            kern_file, clock, duration_sec, output_dir, kern_measures_cache,
            slot["piece_id"], slot["performance"],
            canonical_status.get(Path(kern_file).stem))
        metadata[key] = dict(
            kern_file=kern_file,
            split="train",
            duration_sec=duration_sec,
            audio_beats=clock.get("audio_beats", []),
            audio_measures=clock["audio_measures"],
            audio_grid=clock.get("audio_grid", []),
            kern_measures=kern_measures,
            kern_measure_alignment=alignment,
            renders=[dict(
                audio_key=key,
                n_frames=n_frames,
                source=slot["source"],
                audio_path=str(slot_audio_path(slot, output_dir)),
            )],
        )

    for piece in valid_pieces:
        for stem in piece["performances"]:
            if not is_alignable(piece["piece_id"], stem):
                continue
            key = _mel_id(piece, stem)
            if key not in mel_info:
                raise FileNotFoundError(f"validation mel missing: {key}")
            kern_file = f"{piece['piece_id'].replace('#', '__')}.krn"
            if not (output_dir / "kern_gt" / kern_file).exists():
                raise FileNotFoundError(f"validation kern missing: {kern_file}")
            duration_sec, n_frames = mel_info[key]
            beats, measures = load_asap_clock(piece["piece_dir"], stem)
            kern_measures, clock, alignment = _aligned_training_clock(
                kern_file, dict(audio_measures=measures, audio_beats=beats),
                duration_sec, output_dir, kern_measures_cache, piece["piece_id"], stem,
                canonical_status.get(Path(kern_file).stem))
            metadata[key] = dict(
                kern_file=kern_file,
                split="valid",
                duration_sec=duration_sec,
                audio_beats=clock["audio_beats"],
                audio_measures=clock["audio_measures"],
                audio_grid=[],
                kern_measures=kern_measures,
                kern_measure_alignment=alignment,
                renders=[dict(
                    audio_key=key,
                    n_frames=n_frames,
                    source="real_wav",
                    audio_path=str(piece["piece_dir"] / f"{stem}.wav"),
                )],
            )

    statuses = Counter(
        info["kern_measure_alignment"]["status"] for info in metadata.values()
    )
    log.info(f"kern/audio measure alignment: {dict(statuses)}")
    for key, info in sorted(metadata.items()):
        alignment = info["kern_measure_alignment"]
        if alignment["status"] == "skip":
            log.info(
                f"kern_measures skip {key}: {alignment['reason']} "
                f"(difference={alignment['difference']})"
            )
    return metadata


def write_train_metadata(metadata: dict, output_dir: Path) -> None:
    """Publish the slot ledger consumed by the shared manifest compiler."""
    path = output_dir / "augmentation_metadata.json"
    tmp_path = path.with_name(f".{path.name}")
    tmp_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    os.replace(tmp_path, path)


def _slot_audio_reuse_state(slot: dict, output_dir: Path) -> str:
    """Classify an existing render without creating or replacing artifacts."""
    from src.audio.synthesis import audio_matches_contract, audio_render_fingerprint

    key = slot_key(slot)
    soundfont = SOUNDFONT_DIR / slot["soundfont"]
    if not soundfont.exists():
        return "missing soundfont"
    if slot["source"] == "human_midi":
        midi_path = (ASAP_DIR / slot["piece_id"].replace("#", "/")
                     / f"{slot['performance']}.mid")
        if not midi_path.exists():
            return "missing source MIDI"
    else:
        xml_path = output_dir / "xml" / f"{slot['score'].replace('#', '__')}.xml"
        if not xml_path.exists():
            return "missing canonical XML"
        midi_path = output_dir / "midi_aug" / f"{key}.mid"

    audio_path = output_dir / "audio_aug" / f"{key}.wav"
    if not audio_path.exists():
        return "render: audio missing"
    if not audio_matches_contract(audio_path):
        return "render: audio contract mismatch"
    fingerprint_path = output_dir / "audio_aug" / f"{key}.render.json"
    if not fingerprint_path.exists():
        return "render: fingerprint missing"

    if slot["source"] == "vn":
        if not midi_path.exists():
            return "render: VN MIDI missing"
        *_, expected_midi_fingerprint = _vn_render_recipe(slot, output_dir)
        if not has_epr_render_fingerprint(
                str(midi_path), expected_midi_fingerprint):
            return "render: VN MIDI stale"

    try:
        recorded = json.loads(fingerprint_path.read_text())["fingerprint"]
    except (OSError, KeyError, TypeError, ValueError):
        return "render: fingerprint invalid"
    expected = audio_render_fingerprint(
        midi_path, soundfont, slot["reverb"])
    return "reusable" if recorded == expected else "render: fingerprint mismatch"


def dry_run(pieces: list[dict], meta: dict, phase: str) -> None:
    """Report what the formal entry would consume and produce without writes."""
    output_dir = meta["output_dir"]
    run_phase1 = phase in ("1", "all")
    run_phase2 = phase in ("2", "all")
    run_phase2_5 = phase in ("2.5", "all")
    run_phase3 = phase in ("3", "all")
    log.info("DRY RUN: no files will be written")

    needs_kern = meta.get("needs_kern", True)
    if run_phase1:
        if needs_kern:
            kern_dir = output_dir / "kern_gt"
            xml_dir = output_dir / "xml"
            kern_ready = sum(
                (kern_dir / f"{piece['piece_id'].replace('#', '__')}.krn").exists()
                for piece in pieces
            )
            xml_ready = sum(
                (xml_dir / f"{piece['piece_id'].replace('#', '__')}.xml").exists()
                for piece in pieces
            )
            log.info(
                "Phase 1: kern_gt %d/%d existing; canonical XML %d/%d existing",
                kern_ready, len(pieces), xml_ready, len(pieces))
        else:
            log.info("Phase 1: empty for this split")

    if meta["split_mode"] != "train":
        raw_ids = [
            _mel_id(piece, stem)
            for piece in pieces for stem in piece["performances"]
        ]
        if run_phase2:
            log.info("Phase 2: source recordings only; no render jobs")
        if run_phase2_5:
            mel_ready = sum(
                (output_dir / "mel" / f"{key}.npy").exists()
                for key in raw_ids
            )
            log.info("Phase 2.5: mel %d/%d existing; %d pending",
                     mel_ready, len(raw_ids), len(raw_ids) - mel_ready)
        if run_phase3:
            log.info("Phase 3: %d test entries planned", len(raw_ids))
        return

    ledger = output_dir / "slot_table.json"
    if ledger.exists():
        slots = json.loads(ledger.read_text())
        with open(AUG_CONFIG) as f:
            tempo = json.load(f)["tempo"]
        updated = _with_slot_tempos(slots, tempo)
        changed = sum(a != b for a, b in zip(slots, updated))
        log.info("VN tempo draws to update: %d (dry run; ledger unchanged)", changed)
        slots = updated
        ledger_state = "existing selections with configured tempo"
    else:
        slots = select_slots(sorted(meta["train_ids"]))
        ledger_state = "would create ledger"
    _, valid_pieces = train_valid_split(
        pieces, meta["train_ids"], meta["valid_ids"])
    valid_ids = [
        _mel_id(piece, stem)
        for piece in valid_pieces
        for stem in piece["performances"]
        if is_alignable(piece["piece_id"], stem)
    ]
    log.info("Slots: %d across %d families (%s)",
             len(slots), len({slot["family"] for slot in slots}), ledger_state)
    log.info("Slot sources: %s", dict(Counter(
        slot["source"] for slot in slots)))

    if run_phase2:
        rendered = [slot for slot in slots if slot["source"] != "real_wav"]
        states = Counter(
            _slot_audio_reuse_state(slot, output_dir) for slot in rendered)
        real_ready = sum(
            slot_audio_path(slot, output_dir).exists()
            for slot in slots if slot["source"] == "real_wav"
        )
        log.info("Phase 2: %d generated-audio slots; %d/%d real WAV direct",
                 len(rendered), real_ready,
                 sum(slot["source"] == "real_wav" for slot in slots))
        for state, count in sorted(states.items()):
            log.info("  %s: %d", state, count)

    all_mel_ids = [slot_key(slot) for slot in slots] + valid_ids
    if run_phase2_5:
        mel_ready = sum(
            (output_dir / "mel" / f"{key}.npy").exists()
            for key in all_mel_ids
        )
        log.info("Phase 2.5: mel %d/%d existing; %d pending",
                 mel_ready, len(all_mel_ids), len(all_mel_ids) - mel_ready)

    if run_phase3:
        vn_slots = [slot for slot in slots if slot["source"] == "vn"]
        vn_clocks = sum(
            (output_dir / "clock_aug" / f"{slot_key(slot)}.json").exists()
            for slot in vn_slots
        )
        log.info("Phase 3: %d train + %d valid entries planned",
                 len(slots), len(valid_ids))
        log.info("  current VN clocks: %d/%d", vn_clocks, len(vn_slots))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ASAP data preparation")
    ap.add_argument("--asap-root", type=Path, required=True, help="the asap-dataset checkout")
    ap.add_argument("--split", required=True,
                    choices=["train", "test-zeng25", "test-clean",
                             "test-asap102"],
                    help="Which ASAP subset to prepare")
    ap.add_argument("--workers", "-j", type=int, default=4)
    ap.add_argument("--phase", choices=["1", "2", "2.5", "3", "all"],
                    default="all")
    ap.add_argument("--skip-kern", action="store_true")
    ap.add_argument("--skip-mel", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report planned work without writing files")
    ap.add_argument("--dump-slots", action="store_true",
                    help="Write the augmentation slot table and stop")
    args = ap.parse_args()
    global ASAP_DIR
    ASAP_DIR = args.asap_root

    pieces, meta = select_pieces(args.split)
    output_dir = meta["output_dir"]

    log.info(f"Split: {args.split} → {output_dir}")
    log.info(f"Pieces: {len(pieces)}")

    if args.dry_run:
        dry_run(pieces, meta, args.phase)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dump_slots:
        if meta["split_mode"] != "train":
            raise SystemExit("Augmentation slots exist for the train split only")
        slots = load_or_build_slots(sorted(meta["train_ids"]), output_dir)
        counts = Counter(slot["source"] for slot in slots)
        log.info(f"Slots by source: {dict(counts)}")
        return

    run_phase1 = args.phase in ("1", "all")
    run_phase2 = args.phase in ("2", "all")
    run_phase2_5 = args.phase in ("2.5", "all")
    run_phase3 = args.phase in ("3", "all")
    needs_kern = meta.get("needs_kern", True)
    kern_dir = output_dir / "kern_gt"
    kern_map = {
        piece["piece_id"]: kern_dir /
        f"{piece['piece_id'].replace('#', '__').replace('/', '_')}.krn"
        for piece in pieces
    }
    if run_phase1:
        if not needs_kern:
            log.info("Split carries no kern ground truth; Phase 1 is empty.")
        elif args.skip_kern:
            log.info("Skipping kern conversion.")
        else:
            kern_map = convert_kerns(pieces, output_dir, workers=args.workers)

    slots = None
    valid_pieces = None
    if meta["split_mode"] == "train" and (
            run_phase2 or run_phase2_5 or run_phase3):
        _, valid_pieces = train_valid_split(
            pieces, meta["train_ids"], meta["valid_ids"])
        slots = load_or_build_slots(sorted(meta["train_ids"]), output_dir)

    if run_phase2:
        if slots is None:
            log.info("Split uses source recordings; Phase 2 is empty.")
        else:
            render_results = render_slots(slots, output_dir, args.workers)
            errors = {key: status for key, status in render_results.items()
                      if status.startswith("error")}
            if errors:
                raise RuntimeError(
                    f"ASAP Phase 2 failed for {len(errors)} render(s)")

    if slots is not None and (run_phase2_5 or run_phase3) and not run_phase2:
        stale = {
            slot_key(slot): state
            for slot in slots if slot["source"] != "real_wav"
            if (state := _slot_audio_reuse_state(slot, output_dir)) != "reusable"
        }
        if stale:
            for key, state in stale.items():
                log.error("%s: %s", key, state)
            raise RuntimeError(
                f"{len(stale)} slot renders missing or stale; run Phase 2 first")

    mel_info = None
    if run_phase2_5:
        if args.skip_mel:
            log.info("Skipping mel generation.")
        elif slots is not None:
            mel_info = compute_train_mels(
                slots, valid_pieces, output_dir, args.workers)
        else:
            mel_info = compute_mels(pieces, output_dir, workers=args.workers)

    if run_phase3:
        if slots is not None:
            if mel_info is None:
                valid_ids = [
                    _mel_id(piece, stem)
                    for piece in valid_pieces
                    for stem in piece["performances"]
                    if is_alignable(piece["piece_id"], stem)
                ]
                mel_info = load_mel_info(
                    [slot_key(slot) for slot in slots] + valid_ids,
                    output_dir,
                )
            metadata = build_train_metadata(
                slots, valid_pieces, mel_info, output_dir)
            write_train_metadata(metadata, output_dir)
            from src.datasets.manifest import create_manifests_from_metadata
            counts = create_manifests_from_metadata(
                output_dir, splits=("train", "valid"))
            log.info("Train: %d  Valid: %d",
                     counts["train"], counts["valid"])
        else:
            if mel_info is None:
                mel_info = load_existing_mels(pieces, output_dir)
            manifest = build_manifest(
                pieces, kern_map, mel_info, output_dir, "test",
                require_kern=needs_kern,
                with_clock=meta.get("needs_clock", False),
            )
            (output_dir / "test_manifest.json").write_text(
                json.dumps(manifest, indent=2),
            )
            log.info(f"Test manifest: {len(manifest)} entries")

    log.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
