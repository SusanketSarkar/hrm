from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .hrm_model import HRM, HRMConfig


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    segments: int = 3
    act: bool = False
    act_threshold: float = 0.99
    log_interval: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


class HRMTrainer:
    def __init__(self, model: HRM, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 config: Optional[TrainConfig] = None):
        self.model = model.to(config.device if config else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config or TrainConfig()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def train(self) -> None:
        device = torch.device(self.cfg.device)
        best_val_acc = 0.0

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}")
            running_loss = 0.0
            running_acc = 0.0
            count = 0
            for step, (x, y) in pbar:
                x = x.to(device)
                y = y.to(device)

                # initialize state per batch
                state = self.model.init_state(x.size(0), x.device)

                # Deep supervision: run multiple segments, stepping optimizer each time
                batch_loss = 0.0
                batch_correct = 0
                for seg in range(self.cfg.segments):
                    state, logits, _ = self.model(x, state=state, segments=1,
                                                  act_threshold=self.cfg.act_threshold if self.cfg.act else None)
                    loss = self.criterion(logits, y)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                    # detach state between segments
                    state = (state[0].detach(), state[1].detach())

                    batch_loss += loss.item()
                    preds = logits.argmax(dim=-1)
                    batch_correct += (preds == y).sum().item()

                    if self.cfg.act:
                        # If model decided to halt in this segment for most samples, skip remaining segments
                        with torch.no_grad():
                            _, _, halt = self.model(x, state=state, segments=0)
                        # Note: segments=0 returns previous state; ignore here

                running_loss += batch_loss
                running_acc += batch_correct / x.size(0) / max(1, self.cfg.segments)
                count += 1
                if (step + 1) % self.cfg.log_interval == 0:
                    pbar.set_postfix({"loss": f"{running_loss / count:.4f}", "acc": f"{running_acc / count:.4f}"})

            val_acc = 0.0
            if self.val_loader is not None:
                val_acc = self.evaluate(self.val_loader, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint("checkpoints/best.pt")

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, device: torch.device) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            state = self.model.init_state(x.size(0), x.device)
            state, logits, _ = self.model(x, state=state, segments=1)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
        acc = correct / max(1, total)
        return acc

    def save_checkpoint(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.model.config.__dict__,
        }, path) 