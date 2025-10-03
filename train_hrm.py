from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.benchmarks import ParityDataset
from src.hrm_model import HRM, HRMConfig
from src.trainer import HRMTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a minimal HRM on a toy task")
    p.add_argument("--task", type=str, default="parity", choices=["parity"], help="Toy task to train on")
    p.add_argument("--seq_len", type=int, default=32, help="Sequence length for parity task")
    p.add_argument("--train_samples", type=int, default=20000)
    p.add_argument("--val_samples", type=int, default=2000)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--hidden_low", type=int, default=256)
    p.add_argument("--hidden_high", type=int, default=256)
    p.add_argument("--cycles_n", type=int, default=3)
    p.add_argument("--steps_per_cycle_t", type=int, default=4)

    p.add_argument("--segments", type=int, default=3)
    p.add_argument("--use_act", action="store_true")
    p.add_argument("--act_threshold", type=float, default=0.99)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    return p.parse_args()


def make_dataloaders(args: argparse.Namespace):
    if args.task == "parity":
        train_ds = ParityDataset(args.train_samples, seq_len=args.seq_len)
        val_ds = ParityDataset(args.val_samples, seq_len=args.seq_len, seed=7)
        input_dim = args.seq_len
        output_dim = 2
    else:
        raise ValueError(f"Unknown task: {args.task}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, input_dim, output_dim


def main():
    args = parse_args()

    train_loader, val_loader, input_dim, output_dim = make_dataloaders(args)

    model_cfg = HRMConfig(
        input_dim=input_dim,
        hidden_dim_low=args.hidden_low,
        hidden_dim_high=args.hidden_high,
        embed_dim=args.embed_dim,
        output_dim=output_dim,
        cycles_n=args.cycles_n,
        steps_per_cycle_t=args.steps_per_cycle_t,
        use_act=args.use_act,
        act_threshold=args.act_threshold,
    )
    model = HRM(model_cfg)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        segments=args.segments,
        act=args.use_act,
        act_threshold=args.act_threshold,
        device=args.device,
        num_workers=args.num_workers,
    )

    trainer = HRMTrainer(model, train_loader, val_loader, train_cfg)
    trainer.train()


if __name__ == "__main__":
    main() 