"""
Training Script for piano-base
===================================

Supports:
- Single GPU training
- DDP (Distributed Data Parallel) multi-GPU training
- Mixed precision (BF16/FP16)
- Gradient accumulation
- Wandb logging
- Checkpoint saving/loading

Usage:
    # Single GPU
    python -m src.a2s.piano.train --config configs/piano_2gpu.yaml

    # Multi-GPU DDP (e.g., GPU 1 and 4)
    CUDA_VISIBLE_DEVICES=1,4 torchrun --nproc_per_node=2 -m src.a2s.piano.train --config configs/piano_2gpu.yaml

    # Sanity check (overfit one batch)
    python -m src.a2s.piano.train --config configs/piano_2gpu.yaml --sanity-check
"""

import argparse
import logging
import math
import os
import re
import sys
import queue
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

# Debug: Print CUDA visibility info BEFORE any torch import
_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
print(f"[DEBUG] CUDA_VISIBLE_DEVICES = {_cuda_visible}", flush=True)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

# Debug: Print CUDA device info after torch import
print(f"[DEBUG] torch.cuda.device_count() = {torch.cuda.device_count()}", flush=True)
print(f"[DEBUG] torch.cuda.is_available() = {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[DEBUG] cuda:{i} = {torch.cuda.get_device_name(i)}", flush=True)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.a2s import (
    ChunkedDataset,
    A2SCollator,
    ManifestDataset,
    PrefixCollator,
    PrefixedChunkedDataset,
)
from src.a2s.piano.config import PianoConfig
from src.a2s.piano.model import build_piano_model
from src.a2s.piano.tempo_losses import dense_phase_classification_loss
from src.a2s.piano.tokenizer import (
    VOCAB,
    KernTokenizer,
)
from src.utils.seed import load_rng_state_dict, rng_state_dict, set_seed

logger = logging.getLogger(__name__)


@torch.no_grad()
def _structured_downbeat_counts(predicted, targets, supervision, tolerance):
    active = supervision > 0
    predicted = predicted.bool() & active
    targets = (targets > 0) & active
    kernel = 2 * tolerance + 1
    predicted_near = F.max_pool1d(
        predicted.float().unsqueeze(1), kernel, 1, tolerance
    ).squeeze(1).bool()
    target_near = F.max_pool1d(
        targets.float().unsqueeze(1), kernel, 1, tolerance
    ).squeeze(1).bool()
    return (
        (predicted & target_near).sum(),
        predicted.sum(),
        (targets & predicted_near).sum(),
        targets.sum(),
    )


def _prefetched_batches(loader):
    """Overlap CPU preparation without copying dataset state into worker processes."""
    iterator = iter(loader)
    pending = queue.Queue(maxsize=1)
    stopped = threading.Event()

    def send(kind, value=None):
        while not stopped.is_set():
            try:
                pending.put((kind, value), timeout=0.1)
                return
            except queue.Full:
                continue

    def produce():
        try:
            for batch in iterator:
                if stopped.is_set():
                    break
                send("item", batch)
        except BaseException as error:
            send("error", error)
        finally:
            send("end")

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    try:
        while True:
            kind, value = pending.get()
            if kind == "end":
                break
            if kind == "error":
                raise value
            yield value
    finally:
        stopped.set()
        worker.join()


def compute_training_objective(
    model_output,
    labels: torch.Tensor,
    batch: Dict[str, Any],
    pad_id: int,
    label_smoothing: float,
    loss_cfg: Dict[str, Any],
):
    """Return the active objective, score logits, and unweighted components."""
    score_logits = (
        model_output['score_logits']
        if isinstance(model_output, dict)
        else model_output
    )
    score_loss = F.cross_entropy(
        score_logits.reshape(-1, score_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )
    components = {'loss_score': score_loss}
    if not isinstance(model_output, dict):
        return score_loss, score_logits, components

    required = ('gt_downbeat_phi', 'phase_supervision_mask')
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f'Joint training batch lacks targets: {missing}')

    downbeat_phase_loss = dense_phase_classification_loss(
        model_output['downbeat_phase_logits'],
        batch['gt_downbeat_phi'],
        batch['phase_supervision_mask'],
    )
    components['loss_downbeat_phase'] = downbeat_phase_loss
    total = (
        float(loss_cfg.get('lambda_score', 1.0)) * score_loss
        + float(loss_cfg.get('lambda_downbeat_phase', 10.0))
        * downbeat_phase_loss
    )
    return total, score_logits, components


def setup_logging(rank: int = 0, level: int = logging.INFO):
    """Setup logging (only rank 0 logs to console)."""
    if rank == 0:
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    """Trainer for PianoModel (hFT foundation + Transformer decoder)."""

    def __init__(
        self,
        config: PianoConfig,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        use_wandb: bool = False,
        gradient_accumulation_steps: int = 1,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 0,  # 0 = disabled
        save_every_n_epochs: int = 10,
        loss_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip = gradient_clip
        self.log_every_n_steps = int(
            (loss_cfg or {}).get('log_every_n_steps', 0)
        )
        self.early_stopping_patience = early_stopping_patience
        self.save_every_n_epochs = save_every_n_epochs
        self.loss_cfg = loss_cfg or {}

        # Model
        self.model = model.to(self.device)
        if world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                find_unused_parameters=False,
            )

        # Data loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Optimizer: the frozen hFT has no trainable parameters, so every
        # trainable parameter shares the base LR.
        base_lr = config.learning_rate if hasattr(config, 'learning_rate') else 1e-4
        weight_decay = config.weight_decay if hasattr(config, 'weight_decay') else 0.01
        phase_start = getattr(config, 'phase_lr_decay_start_step', None)
        phase_end = getattr(config, 'phase_lr_decay_end_step', None)
        phase_min = getattr(config, 'phase_lr_min', None)
        self.phase_lr_group_index = None
        if any(v is not None for v in (phase_start, phase_end, phase_min)):
            if (
                any(v is None for v in (phase_start, phase_end, phase_min))
                or not isinstance(phase_start, int)
                or not isinstance(phase_end, int)
                or phase_start < config.warmup_steps
                or phase_end <= phase_start
                or not math.isfinite(phase_min)
                or not 0 < phase_min <= base_lr
            ):
                raise ValueError(
                    'Phase LR requires warmup <= start < end and 0 < min <= base LR'
                )

        NO_DECAY_SUFFIXES = ('bias',)
        NO_DECAY_KEYWORDS = (
            'norm', 'ln_',
            '.embedding', 'token_emb',
        )

        def _is_no_decay(name: str) -> bool:
            if name.endswith(NO_DECAY_SUFFIXES):
                return True
            return any(kw in name for kw in NO_DECAY_KEYWORDS)

        # Two groups: decay / no_decay
        other_decay, other_nodecay = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            name = name.removeprefix('module.')
            (other_nodecay if _is_no_decay(name) else other_decay).append(param)

        if self.rank == 0:
            logger.info(
                f'Optimizer: {len(other_decay)} decay params (wd={weight_decay}), '
                f'{len(other_nodecay)} no-decay params (wd=0) | '
                f'base_lr={base_lr:.2e}'
            )

        param_groups = [
            {'params': other_decay,   'lr': base_lr,  'weight_decay': weight_decay},
            {'params': other_nodecay, 'lr': base_lr,  'weight_decay': 0.0},
        ]

        parameter_names = {
            id(p): name.removeprefix('module.')
            for name, p in self.model.named_parameters()
        }
        # Unsplit ordering recovers Adam state from checkpoints without names.
        self._unsplit_optimizer_param_names = [
            [parameter_names[id(p)] for p in group['params']]
            for group in param_groups
        ]
        if phase_start is not None:
            phase_groups = []
            for group in param_groups[:2]:
                phase_params = [
                    p for p in group['params']
                    if parameter_names[id(p)].startswith('tempo.')
                ]
                group['params'] = [
                    p for p in group['params']
                    if not parameter_names[id(p)].startswith('tempo.')
                ]
                phase_groups.append({**group, 'params': phase_params})
            if not any(group['params'] for group in phase_groups):
                raise ValueError('Phase LR requires trainable tempo.* parameters')
            self.phase_lr_group_index = len(param_groups)
            param_groups.extend(phase_groups)
        for group in param_groups:
            group['param_names'] = [parameter_names[id(p)] for p in group['params']]

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)

        # Learning rate scheduler
        total_steps = len(train_loader) * (config.max_epochs if hasattr(config, 'max_epochs') else 50)
        total_steps = total_steps // gradient_accumulation_steps
        self.total_steps = total_steps
        self.steps_per_epoch = max(
            len(train_loader) // gradient_accumulation_steps, 1
        )
        warmup_steps = config.warmup_steps if hasattr(config, 'warmup_steps') else 1000
        lr_min_ratio = getattr(config, 'lr_min_ratio', 0.1)

        other_lambda = lambda step: self._get_lr_scale(step, base_lr, base_lr * lr_min_ratio, warmup_steps, self.total_steps)
        lr_lambdas = [other_lambda, other_lambda]
        if self.phase_lr_group_index is not None:
            anchor = base_lr * other_lambda(phase_start)
            if phase_min > anchor:
                raise ValueError('Phase LR minimum exceeds its LR at decay start')
            phase_lambda = lambda step: self._get_phase_lr_scale(
                step, base_lr, other_lambda
            )
            lr_lambdas += [phase_lambda, phase_lambda]

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambdas,
        )
        self.base_lr = base_lr
        self.min_lr = base_lr * lr_min_ratio

        # Mixed precision
        # BF16 has sufficient dynamic range and does NOT need loss scaling.
        # GradScaler is only needed for FP16 (to prevent underflow).
        # Using GradScaler with BF16 adds overhead (unscale + inf/NaN scan every step) for no benefit.
        self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = (
            torch.amp.GradScaler('cuda') if self.autocast_dtype == torch.float16 else None
        )

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.warmup_steps = warmup_steps
        # Every arm selects on the unweighted score cross-entropy: 'valid_loss'
        # is the lambda-weighted joint objective, so arms with different
        # auxiliary losses would otherwise be selected by different rules.
        self.selection_metric = 'loss_score'
        self.selection_metric_sign = 1.0
        self.best_selection_loss = float('inf')
        self.epochs_without_improvement = 0

        # Wandb
        self.use_wandb = use_wandb and rank == 0

        # Padding
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        self.pad_id = raw_model.pad_id if hasattr(raw_model, 'pad_id') else 0

        # Build pitch token mask for loss breakdown (pitch vs structure vs duration)
        self._build_token_type_masks()

    def _get_lr_scale(self, step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
        """Return LR scale factor for LambdaLR scheduler.

        Schedule:
          Steps 0            → warmup_steps : linear warmup 0 → 1.0
          Steps warmup_steps → total_steps  : cosine decay 1.0 → min_lr/max_lr
          Steps >= total_steps               : hold at min_lr/max_lr
        """
        import math

        # Linear warmup
        if step < warmup_steps:
            return step / max(warmup_steps, 1)

        # Hold at min after total_steps
        if step >= total_steps:
            return min_lr / max_lr

        # Cosine decay: warmup_steps → total_steps
        decay_steps = max(total_steps - warmup_steps, 1)
        t = (step - warmup_steps) / decay_steps  # 0 → 1
        min_scale = min_lr / max_lr
        return min_scale + 0.5 * (1.0 - min_scale) * (1.0 + math.cos(math.pi * t))

    def _get_phase_lr_scale(self, step, base_lr, original_scale):
        start = self.config.phase_lr_decay_start_step
        end = self.config.phase_lr_decay_end_step
        if step <= start:
            return original_scale(step)
        floor = self.config.phase_lr_min / base_lr
        if step >= end:
            return floor
        progress = (step - start) / (end - start)
        return floor + 0.5 * (original_scale(start) - floor) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _restore_optimizer_schedule(self, checkpoint):
        saved = checkpoint['optimizer_state_dict']
        named = all('param_names' in g for g in saved['param_groups'])
        regroup = self.phase_lr_group_index is not None or named
        if not regroup:
            self.optimizer.load_state_dict(saved)
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return

        saved_names = (
            [g['param_names'] for g in saved['param_groups']]
            if named else self._unsplit_optimizer_param_names
        )
        if len(saved_names) != len(saved['param_groups']) or any(
            len(names) != len(group['params'])
            for names, group in zip(saved_names, saved['param_groups'])
        ):
            raise ValueError('Cannot map checkpoint optimizer groups to parameters')
        by_name = {
            name.removeprefix('module.'): (index, group)
            for names, group in zip(saved_names, saved['param_groups'])
            for name, index in zip(names, group['params'])
        }
        current = self.optimizer.state_dict()
        parameters = {
            name.removeprefix('module.'): param
            for name, param in self.model.named_parameters()
        }
        current_names = {
            name for group in current['param_groups'] for name in group['param_names']
        }
        if set(by_name) - current_names:
            raise ValueError('Checkpoint optimizer parameter names do not match')
        fresh = current_names - set(by_name)
        if fresh:
            # Parameters added since the checkpoint keep the group's schedule
            # and start with empty Adam state.
            logger.info(f'Optimizer: {len(fresh)} new parameters start fresh')
        restored = {'state': {}, 'param_groups': []}
        for group in current['param_groups']:
            names = group['param_names']
            known = [name for name in names if name in by_name]
            source = by_name[known[0]][1] if known else group
            restored['param_groups'].append({
                **source, 'params': group['params'], 'param_names': names,
            })
            for name, index in zip(names, group['params']):
                if name not in by_name:
                    continue
                old_index, _ = by_name[name]
                if old_index in saved['state']:
                    state = saved['state'][old_index]
                    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                        if key in state and state[key].shape != parameters[name].shape:
                            raise ValueError(f'Checkpoint Adam state shape mismatch: {name}')
                    restored['state'][index] = state

        # A shorter continuation must not compress the original score schedule.
        horizon = checkpoint.get('lr_schedule_total_steps')
        saved_config = checkpoint.get('config')
        if horizon is None and saved_config is not None:
            if getattr(saved_config, 'gradient_accumulation_steps', 1) == 1:
                cutoff = checkpoint.get('coordinate_predicted_only_step')
                epochs = getattr(saved_config, 'coordinate_predicted_only_epoch', 0)
                if cutoff is not None and epochs > 0:
                    horizon = (int(cutoff) // epochs) * saved_config.max_epochs
        if horizon is not None:
            self.total_steps = int(horizon)
        self.optimizer.load_state_dict(restored)
        schedule = dict(checkpoint['scheduler_state_dict'])
        schedule['base_lrs'] = [g['initial_lr'] for g in self.optimizer.param_groups]
        schedule['lr_lambdas'] = [None] * len(schedule['base_lrs'])
        self.scheduler.load_state_dict(schedule)
        rates = [
            base * fn(self.scheduler.last_epoch)
            for base, fn in zip(self.scheduler.base_lrs, self.scheduler.lr_lambdas)
        ]
        for group, rate in zip(self.optimizer.param_groups, rates):
            group['lr'] = rate
        self.scheduler._last_lr = rates

    def _build_token_type_masks(self):
        """Build boolean masks over vocab to classify token types.

        Creates four logged 1-D boolean tensors of shape [vocab_size]:
        - pitch_mask: pitch tokens (e.g. C, D#, ee, GG, r)
        - duration_mask: duration tokens (e.g. 4, 8., 16)
        - struct_mask: structural tokens (<bar>, <grid>, brackets, tie, ...)
        - schema_mask: meter and key-signature tokens

        Used for per-type loss breakdown logged to wandb (monitoring only,
        does not affect training).
        """
        tokenizer = KernTokenizer()
        V = self.config.vocab_size

        schema_prefixes = ('<key:', '<num:', '<den:')
        # Hand brackets, voice address, tie, continuation, and bar/grid/tup
        # all count as structure.
        struct_names = {
            '<pad>', '<sos>', '<eos>', '<bar>', '<grid>',
            '<tup>', '</tup>', '<tie>', '</tie>', '<v>',
        }
        struct_prefixes = ('<pl', '</pl', '<pr', '</pr')
        # Durations are the pure-numeric tokens (optional dots).
        duration_re = re.compile(r'\d+\.*')
        # r (rest) stays in pitch: it's the "no pitch" decision, needs audio

        struct_ids = set()
        duration_ids = set()
        pitch_ids = set()
        schema_ids = set()

        for tok_str, tok_id in tokenizer.vocab.items():
            if tok_str.startswith(schema_prefixes):
                schema_ids.add(tok_id)
            elif tok_str in struct_names or tok_str.startswith(struct_prefixes):
                struct_ids.add(tok_id)
            elif duration_re.fullmatch(tok_str):
                duration_ids.add(tok_id)

        # Everything else is a pitch token (note names, r, accidentals)
        classified = struct_ids | duration_ids | schema_ids
        for tok_id in range(V):
            if tok_id not in classified:
                pitch_ids.add(tok_id)

        self._struct_mask = torch.zeros(V, dtype=torch.bool)
        self._duration_mask = torch.zeros(V, dtype=torch.bool)
        self._pitch_mask = torch.zeros(V, dtype=torch.bool)
        self._schema_mask = torch.zeros(V, dtype=torch.bool)
        for i in struct_ids:
            self._struct_mask[i] = True
        for i in duration_ids:
            self._duration_mask[i] = True
        for i in pitch_ids:
            self._pitch_mask[i] = True
        for i in schema_ids:
            self._schema_mask[i] = True

        if self.rank == 0:
            logger.info(
                f'Token type masks: {self._pitch_mask.sum()} pitch, '
                f'{self._duration_mask.sum()} duration, '
                f'{self._struct_mask.sum()} struct, '
                f'{self._schema_mask.sum()} schema'
            )

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move every tensor field in a batch to ``self.device``.

        Non-tensor fields (Python lists for alignment info, etc.) pass through
        untouched.  Uses ``non_blocking=True`` so pinned-memory transfers can
        overlap compute.  This is a no-op for tensors already on the target
        device, so it is safe to call unconditionally from any training loop.
        """
        moved: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device, non_blocking=True)
            else:
                moved[k] = v
        return moved

    @torch.no_grad()
    def _compute_loss_breakdown(
        self,
        logits: torch.Tensor,  # [B, S, V]
        labels: torch.Tensor,  # [B, S]
    ) -> Dict[str, float]:
        """Compute per-token-type loss (monitoring only, detached)."""
        ce = nn.CrossEntropyLoss(reduction='none')
        per_token = ce(logits.view(-1, logits.size(-1)), labels.view(-1))  # [B*S]
        flat_labels = labels.view(-1)

        # Map each label to its type using pre-built masks
        pitch_mask = self._pitch_mask.to(flat_labels.device)
        duration_mask = self._duration_mask.to(flat_labels.device)
        struct_mask = self._struct_mask.to(flat_labels.device)
        schema_mask = self._schema_mask.to(flat_labels.device)

        non_pad = flat_labels != 0
        is_pitch = pitch_mask[flat_labels] & non_pad
        is_duration = duration_mask[flat_labels] & non_pad
        is_struct = struct_mask[flat_labels] & non_pad
        is_schema = schema_mask[flat_labels] & non_pad

        result = {}
        if is_pitch.any():
            result['loss_pitch'] = per_token[is_pitch].mean().item()
        if is_duration.any():
            result['loss_duration'] = per_token[is_duration].mean().item()
        if is_struct.any():
            result['loss_struct'] = per_token[is_struct].mean().item()
        if is_schema.any():
            result['loss_schema'] = per_token[is_schema].mean().item()
        return result

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        accumulated_components: Dict[str, float] = {}
        epoch_ce_loss_accum = 0.0  # running sum of avg_accum_loss for epoch-level reporting
        micro_batch_count = 0  # count successful micro-batches (not raw batch_idx)

        pbar = tqdm(
            _prefetched_batches(self.train_loader),
            total=len(self.train_loader),
            desc=f'Epoch {self.epoch}',
            disable=self.rank != 0,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # DDP-safe skip: all ranks must agree to skip, otherwise
            # the rank that skips misses a gradient all-reduce and NCCL
            # times out waiting for the missing collective.
            if isinstance(self.model, DDP):
                has_data = torch.tensor(
                    [batch is not None], device=self.device, dtype=torch.int32
                )
                dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
                if has_data.item() == 0:
                    continue
            elif batch is None:
                continue

            # Move all tensor fields in one place
            batch = self._to_device(batch)
            input_spec = batch['mel']
            input_ids = batch['input_ids']
            labels = batch['labels']
            tgt_key_padding_mask = input_ids == self.pad_id
            micro_batch_count += 1
            objective_batch = batch
            if 'mel_ext' in batch:
                # Prefixed window: the phase branch is scored on the whole
                # window, so its targets are the extended ones.
                input_spec = batch['mel_ext']
                objective_batch = dict(batch)
                for key in PrefixCollator.PHASE_KEYS:
                    if f'{key}_ext' in batch:
                        objective_batch[key] = batch[f'{key}_ext']

            # Skip DDP all-reduce on non-update micro-batches to save
            # communication overhead during gradient accumulation.
            is_update_step = micro_batch_count % self.gradient_accumulation_steps == 0
            maybe_no_sync = (
                self.model.no_sync() if isinstance(self.model, DDP) and not is_update_step
                else nullcontext()
            )

            with maybe_no_sync:
                with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                    model_output = self.model(
                        input_spec, input_ids,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        frame_valid=batch.get('frame_valid'),
                        prefix_offsets=batch.get('prefix_offsets'),
                        memory_start=batch.get('memory_start'),
                        score_phase=batch.get('score_phase'),
                    )
                    objective, score_logits, loss_components = compute_training_objective(
                        model_output,
                        labels,
                        objective_batch,
                        self.pad_id,
                        self.config.label_smoothing,
                        self.loss_cfg,
                    )
                    if 'prefix_offsets' in batch:
                        loss_components['prefix_frames'] = (
                            batch['prefix_offsets'].float().mean()
                        )
                        loss_components['memory_prefix_frames'] = (
                            (batch['prefix_offsets'] - batch['memory_start'])
                            .float().mean()
                        )
                    total_loss = objective / self.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

            accumulated_loss += objective.item()
            for name, value in loss_components.items():
                accumulated_components[name] = (
                    accumulated_components.get(name, 0.0) + value.item()
                )

            # Update weights every gradient_accumulation_steps
            if is_update_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                all_params = [
                    p for group in self.optimizer.param_groups
                    for p in group['params']
                ]

                # Per-component grad norms (pre-clip)
                component_norms = {}
                raw_model = (
                    self.model.module if isinstance(self.model, DDP) else self.model
                )
                for comp_name in (
                    "converter",
                    "conformer",
                    "tempo",
                    "token_embedding",
                    "decoder",
                    "output_proj",
                    "memory_pe",
                    "pos_encoding",
                ):
                    module = getattr(raw_model, comp_name, None)
                    if module is None:
                        continue
                    grads = [
                        p.grad.detach().float().pow(2).sum()
                        for p in module.parameters()
                        if p.grad is not None
                    ]
                    if not grads:
                        continue
                    component_norms[comp_name] = torch.stack(grads).sum().sqrt()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_params,
                    self.gradient_clip,
                )
                component_norms["total"] = grad_norm
                self._component_grad_norms = dict(
                    zip(
                        component_norms,
                        torch.stack(list(component_norms.values())).cpu().tolist(),
                    )
                )

                if self.scaler is not None:
                    old_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    # Only step scheduler if optimizer actually updated
                    # (scaler skips on inf/NaN gradients)
                    if self.scaler.get_scale() >= old_scale:
                        self.scheduler.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()

                self.optimizer.zero_grad()

                # Update metrics (average over accumulation steps)
                avg_accum_loss = accumulated_loss / self.gradient_accumulation_steps
                epoch_ce_loss_accum += avg_accum_loss
                num_batches += 1
                self.global_step += 1

                # Log to wandb
                if self.use_wandb:
                    last_lrs = self.scheduler.get_last_lr()
                    log_dict = {
                        'train/loss': avg_accum_loss,
                        'train/lr_base': last_lrs[0],
                        'train/epoch': self.epoch,
                        'system/gpu_memory_gb': torch.cuda.max_memory_allocated(self.device) / 1024**3,
                    }

                    for name, value in accumulated_components.items():
                        log_dict[f'train/{name}'] = (
                            value / self.gradient_accumulation_steps
                        )

                    if self.phase_lr_group_index is not None:
                        log_dict['train/lr_phase'] = last_lrs[self.phase_lr_group_index]

                    breakdown = self._compute_loss_breakdown(
                        score_logits.detach(), labels
                    )
                    for k, v in breakdown.items():
                        log_dict[f'train/{k}'] = v

                    for comp_name, comp_gn in self._component_grad_norms.items():
                        log_dict[f'grad_norm/{comp_name}'] = comp_gn

                    wandb.log(log_dict, step=self.global_step)

                # Update progress bar
                postfix = {
                    'loss': f'{avg_accum_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'grad': f'{grad_norm:.2f}' if hasattr(grad_norm, '__float__') else 'N/A',
                }
                if self.phase_lr_group_index is not None:
                    postfix['lr_phase'] = f'{self.scheduler.get_last_lr()[self.phase_lr_group_index]:.2e}'
                pbar.set_postfix(postfix)

                # The same per-step components wandb receives, on stdout, so a
                # run's stop condition can be read without a tracking service.
                if (self.rank == 0 and self.log_every_n_steps > 0
                        and self.global_step % self.log_every_n_steps == 0):
                    parts = [f'{k}={v / self.gradient_accumulation_steps:.4f}'
                             for k, v in sorted(accumulated_components.items())]
                    parts += [f'grad_norm/{k}={v:.4f}'
                              for k, v in sorted(self._component_grad_norms.items())]
                    logger.info(
                        f'step {self.global_step}: loss={avg_accum_loss:.4f} '
                        f'grad={float(grad_norm):.4f} ' + ' '.join(parts)
                    )

                accumulated_loss = 0.0
                accumulated_components = {}

        if num_batches == 0:
            logger.warning('Training epoch skipped: no usable chunks; no loss was measured')
            return {}
        avg_loss = epoch_ce_loss_accum / num_batches

        result = {'train_loss': avg_loss}
        return result

    def validate(self) -> Dict[str, float]:
        return self._validate_loader(self.valid_loader)

    @torch.no_grad()
    def _validate_loader(self, loader: Optional[DataLoader]) -> Dict[str, float]:
        if loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        component_sums: Dict[str, float] = {}
        type_loss_sums = {
            'loss_pitch': 0.0,
            'loss_duration': 0.0,
            'loss_struct': 0.0,
            'loss_schema': 0.0,
        }
        type_loss_counts = {key: 0 for key in type_loss_sums}

        for batch_idx, batch in enumerate(_prefetched_batches(loader)):
            if isinstance(self.model, DDP):
                has_data = torch.tensor(
                    [batch is not None], device=self.device, dtype=torch.int32
                )
                dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
                if has_data.item() == 0:
                    continue
            elif batch is None:
                continue

            batch = self._to_device(batch)
            input_spec = batch['mel']
            input_ids = batch['input_ids']
            labels = batch['labels']
            tgt_key_padding_mask = input_ids == self.pad_id

            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                model_output = self.model(
                    input_spec, input_ids,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    frame_valid=batch.get('frame_valid'),
                    score_phase=batch.get('score_phase'),
                )
                objective, score_logits, loss_components = compute_training_objective(
                    model_output,
                    labels,
                    batch,
                    self.pad_id,
                    self.config.label_smoothing,
                    self.loss_cfg,
                )

            total_loss += objective.item()
            num_batches += 1
            for name, value in loss_components.items():
                component_sums[name] = component_sums.get(name, 0.0) + value.item()

            breakdown = self._compute_loss_breakdown(score_logits, labels)
            for k, v in breakdown.items():
                if k in type_loss_sums:
                    type_loss_sums[k] += v
                    type_loss_counts[k] += 1

        if num_batches == 0:
            logger.warning('Validation skipped: no usable chunks; checkpoint selection is unchanged')
            return {}
        avg_loss = total_loss / num_batches

        result = {'valid_loss': avg_loss}
        for name, value in component_sums.items():
            result[name] = value / num_batches
        for k in type_loss_sums:
            if type_loss_counts[k] > 0:
                result[k] = type_loss_sums[k] / type_loss_counts[k]
        return result

    def _gather_rng_states(self):
        """Per-rank RNG streams for exact resume; a collective under DDP."""
        state = rng_state_dict()
        if self.world_size <= 1 or not dist.is_initialized():
            return [state]
        gathered = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(state, gathered, dst=0)
        return gathered

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save checkpoint."""
        # Collective: every rank contributes its RNG streams, so this must
        # run before the rank-0 short-circuit.
        rng_states = self._gather_rng_states()
        if self.rank != 0:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'lr_schedule_total_steps': self.total_steps,
            'selection_metric': self.selection_metric,
            'best_selection_loss': self.best_selection_loss,
            # Retained for readers of checkpoints written before the selection
            # criterion was named explicitly.
            'best_valid_loss': self.best_selection_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'config': self.config,
            'rng_state': rng_states,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f'Saved checkpoint to {path}')

        if is_best:
            best_path = path.parent / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f'Saved best model to {best_path}')

    def load_checkpoint(self, path: Path):
        """Load checkpoint."""
        if not path.exists():
            logger.warning(f'Checkpoint not found: {path}')
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        state_dict = checkpoint['model_state_dict']
        missing, unexpected = model.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            logger.info(f'New params (not in checkpoint): {missing}')
        if unexpected:
            logger.warning(f'Unexpected params in checkpoint: {unexpected}')

        try:
            self._restore_optimizer_schedule(checkpoint)
        except (ValueError, KeyError, RuntimeError) as e:
            if self.phase_lr_group_index is not None:
                raise RuntimeError(
                    'Cannot resume phase LR schedule without preserving Adam state'
                ) from e
            logger.warning(
                f'Optimizer/scheduler state mismatch ({e}); starting fresh '
                f'optimizer (model weights still loaded). Common when new '
                f'params have been added since the checkpoint.'
            )

        self.epoch = checkpoint['epoch'] + 1  # resume from NEXT epoch
        self.global_step = checkpoint['global_step']
        saved_selection_metric = checkpoint.get('selection_metric')
        if saved_selection_metric == self.selection_metric:
            self.best_selection_loss = checkpoint.get(
                'best_selection_loss', checkpoint['best_valid_loss']
            )
            self.epochs_without_improvement = checkpoint.get(
                'epochs_without_improvement', 0
            )
        else:
            # A legacy checkpoint's best_valid_loss tracked the weighted joint
            # objective. Re-evaluate these weights under the score criterion so
            # resuming cannot compare quantities with different meanings.
            valid_metrics = self.validate()
            if self.selection_metric not in valid_metrics:
                raise KeyError(
                    f'Validation lacks selection metric {self.selection_metric}'
                )
            self.best_selection_loss = (
                self.selection_metric_sign
                * valid_metrics[self.selection_metric]
            )
            self.epochs_without_improvement = 0
            logger.info(
                f'Initialized best {self.selection_metric} from resumed '
                f'checkpoint: '
                f'{self.selection_metric_sign * self.best_selection_loss:.4f}'
            )
            if self.rank == 0:
                checkpoint['selection_metric'] = self.selection_metric
                checkpoint['best_selection_loss'] = self.best_selection_loss
                checkpoint['best_valid_loss'] = self.best_selection_loss
                checkpoint['epochs_without_improvement'] = 0
                best_path = path.parent / 'best.pt'
                torch.save(checkpoint, best_path)
                logger.info(
                    f'Rebased best model at {best_path} under '
                    f'valid_{self.selection_metric}'
                )

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        rng_states = checkpoint.get('rng_state')
        if rng_states is None:
            logger.warning(
                'Checkpoint carries no RNG state; shuffle/dropout streams '
                'restart from the base seed instead of continuing the '
                'interrupted run')
        elif len(rng_states) != self.world_size:
            logger.warning(
                f'Checkpoint RNG state covers {len(rng_states)} ranks but '
                f'world_size={self.world_size}; shuffle/dropout streams '
                f'restart from the base seed')
        else:
            try:
                load_rng_state_dict(rng_states[self.rank])
                logger.info('Restored RNG streams from checkpoint')
            except (KeyError, ValueError, RuntimeError) as e:
                logger.warning(
                    f'RNG state restore failed ({e}); shuffle/dropout '
                    f'streams restart from the base seed')

        logger.info(f'Loaded checkpoint from {path} (epoch {self.epoch}, step {self.global_step})')

    def train(self, max_epochs: int, checkpoint_dir: Path):
        """Full training loop."""
        model_for_info = self.model.module if isinstance(self.model, DDP) else self.model
        n_trainable = sum(p.numel() for p in model_for_info.parameters() if p.requires_grad)
        logger.info(f'Starting training for {max_epochs} epochs')
        logger.info(f'Model params: {n_trainable:,} trainable')
        logger.info(f'Gradient accumulation steps: {self.gradient_accumulation_steps}')
        if self.phase_lr_group_index is not None:
            logger.info(
                'Phase LR: tempo.* only, cosine steps %d..%d, floor %.2e',
                self.config.phase_lr_decay_start_step,
                self.config.phase_lr_decay_end_step,
                self.config.phase_lr_min,
            )
        logger.info(f'Effective batch size: {self.train_loader.batch_size * self.world_size * self.gradient_accumulation_steps}')
        if self.early_stopping_patience > 0:
            logger.info(
                'Early stopping: patience=%d', self.early_stopping_patience
            )

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch

            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
                # Recalculate after dataset.set_epoch changed __len__
                import math as _math
                sampler = self.train_loader.sampler
                sampler.num_samples = _math.ceil(len(self.train_loader.dataset) / sampler.num_replicas)
                sampler.total_size = sampler.num_samples * sampler.num_replicas

            # Train
            train_metrics = self.train_epoch()

            # Log OOV skipped chunks (if any)
            if self.rank == 0 and hasattr(self.train_loader.dataset, 'oov_skipped_chunks'):
                skipped = self.train_loader.dataset.oov_skipped_chunks
                if skipped > 0:
                    total = len(self.train_loader.dataset)
                    logger.info(
                        f'Epoch {epoch}: skipped {skipped}/{total} chunks '
                        f'due to OOV tokens/targets '
                        f'({skipped/total*100:.1f}%)'
                    )
                    self.train_loader.dataset.oov_skipped_chunks = 0
            # Validate at end of epoch
            valid_metrics = self.validate()

            # Log
            if self.rank == 0:
                logger.info(
                    f'Epoch {epoch}: '
                    f'train_loss={train_metrics.get("train_loss", "N/A")}, '
                    f'valid_loss={valid_metrics.get("valid_loss", "N/A")}'
                )

                if self.use_wandb:
                    epoch_log = {}
                    if 'train_loss' in train_metrics:
                        epoch_log['epoch/train_loss'] = train_metrics['train_loss']
                    if 'valid_loss' in valid_metrics:
                        epoch_log['epoch/valid_loss'] = valid_metrics['valid_loss']
                    for k in (
                        'loss_score',
                        'loss_downbeat_phase',
                        'loss_pitch', 'loss_duration', 'loss_struct',
                        'loss_schema',
                    ):
                        if k in valid_metrics:
                            epoch_log[f'epoch/valid_{k}'] = valid_metrics[k]
                    for k in ('frame_loss', 'frame_precision', 'frame_recall', 'frame_f1'):
                        vk = f'valid_{k}'
                        if vk in valid_metrics:
                            epoch_log[f'epoch/valid_{k}'] = valid_metrics[vk]
                    wandb.log(epoch_log, step=self.global_step)

            # Save checkpoint and check early stopping
            is_best = False
            if self.selection_metric in valid_metrics:
                selection_loss = (
                    self.selection_metric_sign
                    * valid_metrics[self.selection_metric]
                )
                if selection_loss < self.best_selection_loss:
                    self.best_selection_loss = selection_loss
                    is_best = True
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

            # Validation is not sharded or reduced, so ranks can disagree on
            # is_best by kernel noise; the save is a collective, so every
            # rank must follow rank 0's verdict.
            if self.world_size > 1 and dist.is_initialized():
                flag = torch.tensor([int(is_best)], device=self.device)
                dist.broadcast(flag, src=0)
                is_best = bool(flag.item())

            if (epoch + 1) % self.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(
                    checkpoint_dir / f'epoch_{epoch:03d}.pt',
                    is_best=is_best,
                )

            # Early stopping check
            if (
                self.early_stopping_patience > 0
                and self.epochs_without_improvement >= self.early_stopping_patience
            ):
                if self.rank == 0:
                    logger.info(
                        f'Early stopping triggered: no improvement for {self.epochs_without_improvement} epochs. '
                        f'Best valid_{self.selection_metric}: '
                        f'{self.selection_metric_sign * self.best_selection_loss:.4f}'
                    )
                break

        # Save final checkpoint
        self.save_checkpoint(checkpoint_dir / 'last.pt')

        if self.use_wandb:
            wandb.finish()


def sanity_check(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    loss_cfg: Optional[Dict[str, Any]] = None,
):
    """Overfit one batch to verify pipeline works."""
    logger.info('Running sanity check (overfit one batch)...')

    model.train()
    model = model.to(device)

    batch = next((item for item in train_loader if item is not None), None)
    if batch is None:
        logger.warning('Sanity check has no usable chunks')
        return
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    input_spec = batch.get('mel_ext',batch['mel'])
    objective_batch = dict(batch)
    if 'mel_ext' in batch:
        for key in PrefixCollator.PHASE_KEYS:
            if key+'_ext' in batch:
                objective_batch[key] = batch[key+'_ext']
    input_ids = batch['input_ids']
    labels = batch['labels']
    logger.info(f'Batch shapes: mel={input_spec.shape}, input_ids={input_ids.shape}, labels={labels.shape}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pad_id = model.pad_id if hasattr(model, 'pad_id') else 0
    tgt_key_padding_mask = (input_ids == pad_id).to(device)

    for step in range(100):
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            model_output = model(
                input_spec, input_ids,
                tgt_key_padding_mask=tgt_key_padding_mask,
                frame_valid=batch.get('frame_valid'),
                prefix_offsets=batch.get('prefix_offsets'),
                memory_start=batch.get('memory_start'),
            )
            objective, _, components = compute_training_objective(
                model_output,
                labels,
                objective_batch,
                pad_id,
                label_smoothing=0.0,
                loss_cfg=loss_cfg or {},
            )

        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            component_text = ' '.join(
                f'{name}={value.item():.4f}'
                for name, value in components.items()
            )
            logger.info(
                f'Step {step}: loss={objective.item():.4f} {component_text}'
            )

    import math
    score_loss = components['loss_score'].item()
    logger.info(
        f'Final loss: {objective.item():.4f}  '
        f'(score perplexity={math.exp(score_loss):.2f})'
    )

    if objective.item() < 0.1:
        logger.info('Sanity check PASSED: model can overfit one batch')
    else:
        logger.warning('Sanity check: loss still high, may need more steps')


def main():
    # Handle Ctrl+C gracefully - finish wandb and kill child processes
    import signal

    def cleanup_and_exit(signum, frame):
        print("\n[INFO] Caught signal, cleaning up...", flush=True)
        # Finish wandb run (so it doesn't stay "running" on server)
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish(exit_code=1)
        except Exception:
            pass
        # Kill all processes in our process group
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    parser = argparse.ArgumentParser(description='Train PianoModel (hFT + Transformer decoder)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML (single source of truth for all hyperparameters)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run sanity check (overfit one batch)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='piano-a2s',
                        help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity (team/user); defaults to wandb default entity if not set')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    setup_logging(rank, logging.DEBUG if args.debug else logging.INFO)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    logger.info(f'Rank {rank}/{world_size}, local_rank={local_rank}')

    try:
        # Load config
        config = PianoConfig.from_yaml(args.config)

        # Set seed for reproducibility
        training_seed = getattr(config, 'training_seed', 1234)
        set_seed(training_seed)
        logger.info(f'Set training seed: {training_seed}')

        tokenizer = KernTokenizer()
        config.vocab_size = tokenizer.vocab_size
        logger.info(f'Vocab size: {config.vocab_size}')

        # Create model
        import yaml
        with open(args.config) as _f:
            raw_cfg = yaml.safe_load(_f)
        m_cfg = raw_cfg['model']
        phase_model = bool(m_cfg.get('phase_model', False))
        loss_cfg = raw_cfg.get('loss', {})
        # Per-step components on stdout; 0 leaves a run's output unchanged.
        loss_cfg['log_every_n_steps'] = int(
            raw_cfg.get('training', {}).get('log_every_n_steps', 0)
        )
        if phase_model:
            if not getattr(config, 'chunk_frames', 0):
                raise ValueError(
                    'Direct joint training requires bar-aligned chunking'
                )
            for key, default in (
                ('lambda_score', 1.0),
                ('lambda_downbeat_phase', 10.0),
            ):
                value = float(loss_cfg.get(key, default))
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'{key} must be finite and non-negative')
                loss_cfg[key] = value
        model = build_piano_model(
            m_cfg,
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.vocab['<pad>'],
            device=f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu',
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Model created: {n_params:,} trainable params')
        config.model_config = dict(m_cfg)

        # Create datasets
        manifest_dir = Path(config.manifest_dir)
        train_manifest = manifest_dir / 'train_manifest.json'
        valid_manifest = manifest_dir / 'valid_manifest.json'

        if not train_manifest.exists():
            logger.error(f'Train manifest not found: {train_manifest}')
            logger.info('Please run the data preparation script first.')
            return

        aug_metadata_path = manifest_dir / 'augmentation_metadata.json'

        def _build_pool(pool_manifest: Path, pool_aug_path: Path, overlap: int):
            pool = ManifestDataset(
                pool_manifest,
                tokenizer=tokenizer,
                max_seq_len=config.max_seq_len,
                augmentation_metadata_path=pool_aug_path,
            )
            if hasattr(config, 'chunk_frames') and config.chunk_frames > 0:
                pool = ChunkedDataset(
                    pool,
                    tokenizer=tokenizer,
                    chunk_frames=config.chunk_frames,
                    overlap_frames=overlap,
                    max_seq_len=config.max_seq_len,
                    include_clock_targets=phase_model,
                )
            return pool

        train_dataset = _build_pool(train_manifest, aug_metadata_path, config.overlap_frames)

        valid_dataset = None
        if valid_manifest.exists():
            valid_dataset = _build_pool(valid_manifest, aug_metadata_path, 0)

        # Create collator
        score_phase = m_cfg.get('coordinate_delivery', 'memory') == 'cross_attention'
        collator = A2SCollator(
            pad_token_id=tokenizer.vocab['<pad>'],
            max_seq_len=config.max_seq_len,
            score_phase=score_phase,
        )
        train_collator = collator

        # Prefixed windows: each training chunk opens up to a bar early for
        # the trunk and phase branch, up to half a bar early for the decoder
        # memory.  Off by default so existing runs are unchanged.
        prefix_cfg = raw_cfg.get('data', {}).get('prefix') or {}
        if bool(prefix_cfg.get('enabled', False)):
            if not phase_model or not isinstance(train_dataset, ChunkedDataset):
                raise ValueError(
                    'data.prefix requires model.phase_model=true and '
                    'bar-aligned chunking'
                )
            train_dataset = PrefixedChunkedDataset(
                train_dataset,
                max_frames=int(prefix_cfg.get('max_frames', 384)),
                seed=training_seed,
            )
            train_collator = PrefixCollator(
                pad_token_id=tokenizer.vocab['<pad>'],
                max_seq_len=config.max_seq_len,
                score_phase=score_phase,
            )
            if rank == 0:
                logger.info(
                    f"Prefixed windows: run-in up to "
                    f"{train_dataset.max_frames} frames (one bar), decoder "
                    f"memory capped at half the bar"
                )

        # Create data loaders
        if world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            )
        else:
            train_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            collate_fn=train_collator,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

        valid_loader = None
        if valid_dataset is not None:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collator,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=4 if args.num_workers > 0 else None,
            )
        # Sanity check
        if args.sanity_check:
            device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
            sanity_check(
                model, train_loader, device, loss_cfg=loss_cfg
            )
            return

        # Setup wandb
        use_wandb = args.wandb and HAS_WANDB and rank == 0
        if args.wandb and not HAS_WANDB:
            logger.warning('wandb not installed, skipping logging')
        if use_wandb:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=f'train-{world_size}gpu',
                config={
                    'batch_size': config.batch_size,
                    'world_size': world_size,
                    'effective_batch_size': config.batch_size * world_size * config.gradient_accumulation_steps,
                    'gradient_accumulation_steps': config.gradient_accumulation_steps,
                    'gradient_clip': config.gradient_clip,
                    'max_epochs': config.max_epochs,
                    'learning_rate': config.learning_rate,
                    'manifest_dir': config.manifest_dir,
                    'phase_lr_decay_start_step': config.phase_lr_decay_start_step,
                    'phase_lr_decay_end_step': config.phase_lr_decay_end_step,
                    'phase_lr_min': config.phase_lr_min,
                    'model_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                },
                tags=['training', f'{world_size}gpu'],
            )
            logger.info(f'Wandb: {wandb.run.url}')

        # Create trainer
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            use_wandb=use_wandb,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_clip=config.gradient_clip,
            early_stopping_patience=config.early_stopping_patience,
            save_every_n_epochs=config.save_every_n_epochs,
            loss_cfg=loss_cfg,
        )

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(Path(args.resume))

        # Train
        trainer.train(
            max_epochs=config.max_epochs,
            checkpoint_dir=Path(config.checkpoint_dir),
        )

    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
