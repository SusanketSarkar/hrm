# Hierarchical Reasoning Model (HRM) — Minimal Implementation

This repository contains a minimal, runnable PyTorch implementation of the Hierarchical Reasoning Model (HRM) inspired by the paper "Hierarchical Reasoning Model" (arXiv:2506.21734).

It includes:
- HRM architecture with high-level and low-level recurrent modules operating at different time scales
- 1-step gradient approximation (no BPTT) for efficient training
- Deep supervision training over multiple "segments" (state is detached between segments)
- Optional Adaptive Computation Time (ACT) halting head (disabled by default)
- A toy dataset (Parity) to verify training end-to-end


## Environment setup

You may use your own Python environment or a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
pip install --upgrade pip
pip install -r requirements.txt
```


## Train the model (toy Parity task)

The Parity task generates random binary sequences and asks the model to predict the parity (even/odd) of the bit sum.

```bash
python train_hrm.py \
  --task parity \
  --seq_len 32 \
  --train_samples 20000 \
  --val_samples 2000 \
  --epochs 10 \
  --batch_size 128 \
  --lr 1e-3 \
  --cycles_n 3 \
  --steps_per_cycle_t 4 \
  --segments 3
```

Arguments you may tune:
- `--seq_len`: input length (and input dimension)
- `--cycles_n`: number of high-level cycles N
- `--steps_per_cycle_t`: low-level steps per cycle T
- `--segments`: number of deep-supervision segments (state is detached between segments)
- `--use_act`: enable halting head (Adaptive Computation Time)

Trained checkpoints are saved to `checkpoints/best.pt` when validation accuracy improves.


## Code layout

- `src/hrm_model.py`: HRM model definition (input embedding f_I, L-module, H-module, output head f_O). Implements the 1-step gradient approximation by running all but the last two updates under `torch.no_grad()`.
- `src/trainer.py`: Training loop using deep supervision across multiple segments.
- `src/benchmarks.py`: Toy `ParityDataset` for quick verification.
- `train_hrm.py`: CLI to configure data, model, and training.


## Notes

- This implementation focuses on clarity and minimalism so you can train and extend the HRM quickly. It does not replicate the full suite of tasks from the paper (e.g., ARC, Sudoku, Maze), but provides the architectural core and training scheme to build upon.
- To adapt HRM to your own task, implement a dataset returning `(features, label)` and set `input_dim` and `output_dim` accordingly.


## Reference

- Hierarchical Reasoning Model (HRM): arXiv:2506.21734 