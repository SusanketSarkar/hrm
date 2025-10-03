from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class HRMConfig:
    input_dim: int
    hidden_dim_low: int = 256
    hidden_dim_high: int = 256
    embed_dim: int = 256
    output_dim: int = 2
    cycles_n: int = 3
    steps_per_cycle_t: int = 4
    use_act: bool = False
    act_threshold: float = 0.99


class HRM(nn.Module):
    """
    Minimal Hierarchical Reasoning Model (HRM)

    - Input embedding f_I
    - Low-level recurrent module f_L (GRUCell)
    - High-level recurrent module f_H (GRUCell)
    - Output head f_O

    Implements: 1-step gradient approximation and deep supervision-friendly step API.
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.embed_dim),
            nn.GELU(),
        )

        self.low_cell = nn.GRUCell(config.embed_dim + config.hidden_dim_high + config.hidden_dim_low,
                                   config.hidden_dim_low)
        self.high_cell = nn.GRUCell(config.hidden_dim_low + config.hidden_dim_high,
                                    config.hidden_dim_high)

        self.output_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim_high),
            nn.Linear(config.hidden_dim_high, config.output_dim),
        )

        if config.use_act:
            self.halt_head = nn.Linear(config.hidden_dim_high, 1)
        else:
            self.halt_head = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.GRUCell):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        z_h = torch.zeros(batch_size, self.config.hidden_dim_high, device=device)
        z_l = torch.zeros(batch_size, self.config.hidden_dim_low, device=device)
        return z_h, z_l

    @torch.no_grad()
    def _roll_until_last_step(self, z_h: torch.Tensor, z_l: torch.Tensor, x_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run N*T-1 timesteps without tracking gradients (approximate gradient scheme).
        Update order per timestep i:
          - Update L given (z_l, z_h, x_embed)
          - If (i+1) % T == 0: update H given (z_h, z_l)
        """
        N = self.config.cycles_n
        T = self.config.steps_per_cycle_t
        total = N * T

        for i in range(total - 1):
            l_input = torch.cat([x_embed, z_h, z_l], dim=-1)
            z_l = self.low_cell(l_input, z_l)
            if (i + 1) % T == 0:
                h_input = torch.cat([z_l, z_h], dim=-1)
                z_h = self.high_cell(h_input, z_h)
        return z_h, z_l

    def hrm_step(self, z_h: torch.Tensor, z_l: torch.Tensor, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform one HRM "segment" forward pass with 1-step gradient approximation.

        Returns: ((z_h_new, z_l_new), logits, halting_prob_optional)
        """
        x_embed = self.input_proj(x)

        # No-grad roll for first (N*T - 1) steps
        z_h_ng, z_l_ng = self._roll_until_last_step(z_h, z_l, x_embed)

        # Track gradients only for the final two updates
        l_input = torch.cat([x_embed, z_h_ng, z_l_ng], dim=-1)
        z_l_new = self.low_cell(l_input, z_l_ng)

        h_input = torch.cat([z_l_new, z_h_ng], dim=-1)
        z_h_new = self.high_cell(h_input, z_h_ng)

        logits = self.output_head(z_h_new)

        halt_prob = None
        if self.halt_head is not None:
            halt_prob = torch.sigmoid(self.halt_head(z_h_new))  # (B,1)

        return (z_h_new, z_l_new), logits, halt_prob

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                segments: int = 1, act_threshold: Optional[float] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Run multiple HRM segments with deep supervision-friendly API.
        If use_act is enabled, early stop when mean halt probability exceeds threshold.

        Returns final_state, last_logits, last_halt_prob
        """
        device = x.device
        batch = x.size(0)
        if state is None:
            state = self.init_state(batch, device)
        z_h, z_l = state

        last_logits: torch.Tensor = None  # type: ignore
        last_halt: Optional[torch.Tensor] = None
        threshold = act_threshold if act_threshold is not None else self.config.act_threshold

        for _ in range(segments):
            (z_h, z_l), logits, halt_prob = self.hrm_step(z_h, z_l, x)
            last_logits = logits
            last_halt = halt_prob
            if self.halt_head is not None and halt_prob is not None:
                if halt_prob.mean().item() >= threshold:
                    break
            # detach the state between segments (deep supervision training step will also detach externally)
        return (z_h, z_l), last_logits, last_halt 