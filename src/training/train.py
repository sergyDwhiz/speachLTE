"""
A small, readable training loop for our ASR experiments—no mystery helpers, just PyTorch.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol, Tuple

import torch
from torch import Tensor, nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard optional during scaffolding.
    SummaryWriter = None  # type: ignore[assignment]

import logging

LOGGER = logging.getLogger(__name__)


class Batch(Protocol):
    """Protocol defining the items required by the training loop."""

    audio_features: Tensor
    feature_lengths: Tensor
    tokens: Tensor
    token_lengths: Tensor


@dataclass(slots=True)
class TrainerConfig:
    """Hyperparameters and runtime toggles for training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    batch_size: int = 8
    grad_clip: float = 5.0
    log_every: int = 25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = torch.cuda.is_available()
    output_dir: str = "artifacts"


class TrainingHarness:
    """Wraps the model, optimizer, and loop bookkeeping."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        writer: Optional["SummaryWriter"] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        requested_device = torch.device(config.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable; falling back to CPU.")
            requested_device = torch.device("cpu")

        self.device = requested_device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        if writer:
            self.writer = writer
        elif SummaryWriter:
            self.writer = SummaryWriter(log_dir=config.output_dir)
        else:
            self.writer = None
        self.amp_enabled = bool(config.mixed_precision and self.device.type == "cuda" and torch.cuda.is_available())
        if config.mixed_precision and not self.amp_enabled:
            LOGGER.info("Mixed precision requested but CUDA unavailable; disabling AMP.")
        self.scaler = (
            torch.amp.GradScaler(device_type="cuda", enabled=True) if self.amp_enabled else None
        )
        self.checkpoint_metadata = checkpoint_metadata or {}

    def train_epoch(self, dataloader: Iterable[Batch], epoch: int) -> float:
        """Run a single training epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(dataloader, start=1):
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1
            if step % self.config.log_every == 0 and self.writer:
                self.writer.add_scalar("train/loss", loss, epoch * 1000 + step)
        return total_loss / max(1, num_batches)

    def _train_step(self, batch: Batch) -> float:
        """Single optimization step with optional AMP."""
        self.optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.amp.autocast(device_type="cuda", enabled=True) if self.amp_enabled else nullcontext()
        )
        with autocast_context:
            logits, logit_lengths = self.model(batch.audio_features.to(self.device), batch.feature_lengths)
            loss = self.criterion(
                logits.log_softmax(dim=-1),
                batch.tokens.to(self.device),
                logit_lengths,
                batch.token_lengths,
            )
        if self.amp_enabled and self.scaler:
            self.scaler.scale(loss).backward()
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return float(loss.detach().cpu())

    def fit(
        self,
        train_loader: Iterable[Batch],
        val_loader: Optional[Iterable[Batch]] = None,
    ) -> Tuple[nn.Module, Optional["SummaryWriter"]]:
        """High-level training loop across epochs."""
        best_loss = float("inf")
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch}/{self.config.epochs} - Train Loss: {train_loss:.4f}", flush=True)
            if self.writer:
                self.writer.add_scalar("train/epoch_loss", train_loss, epoch)
            if val_loader:
                val_loss = self.evaluate(val_loader)
                print(f"Epoch {epoch}/{self.config.epochs} - Val Loss: {val_loss:.4f} (Best: {best_loss:.4f})", flush=True)
                if val_loss < best_loss:
                    best_loss = val_loss
                    self._checkpoint(epoch, val_loss)
                    print(f"Checkpoint saved at epoch {epoch}", flush=True)
        return self.model, self.writer

    def evaluate(self, dataloader: Iterable[Batch]) -> float:
        """Simple evaluation loop calculating average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                logits, logit_lengths = self.model(batch.audio_features.to(self.device), batch.feature_lengths)
                loss = self.criterion(
                    logits.log_softmax(dim=-1),
                    batch.tokens.to(self.device),
                    logit_lengths,
                    batch.token_lengths,
                )
                total_loss += float(loss.cpu())
                num_batches += 1
        return total_loss / max(1, num_batches)

    def _checkpoint(self, epoch: int, val_loss: float) -> None:
        """Persist the best model state."""
        checkpoint_dir = Path(self.config.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        payload.update(self.checkpoint_metadata)
        torch.save(payload, checkpoint_dir / "best.ckpt")
