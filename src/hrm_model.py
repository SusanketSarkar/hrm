import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from einops import rearrange, repeat
import numpy as np


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization as used in the paper"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Pre-compute rotation matrices
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        
        self.register_buffer('cos_cached', torch.cos(freqs))
        self.register_buffer('sin_cached', torch.sin(freqs))

    def forward(self, x, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        
        return self.apply_rotary_pos_emb(x, cos, sin)

    def apply_rotary_pos_emb(self, x, cos, sin):
        # Split x into first half and second half
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        
        # Apply rotation
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated


class GLUVariant(nn.Module):
    """GLU variant activation as mentioned in the paper"""
    def __init__(self, input_dim: int, hidden_dim: int, activation='swish'):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, input_dim, bias=False)
        
        if activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE"""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.rope = RoPEEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Project to q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization"""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.mlp = GLUVariant(hidden_size, hidden_size * mlp_ratio)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Pre-norm attention
        normed_x = self.norm1(x)
        attn_out = self.attention(normed_x, mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm MLP
        normed_x = self.norm2(x)
        mlp_out = self.mlp(normed_x)
        x = x + self.dropout(mlp_out)
        
        return x


class AdaptiveComputationTime(nn.Module):
    """Adaptive Computation Time (ACT) mechanism"""
    def __init__(self, hidden_size: int, max_steps: int = 16, threshold: float = 0.99):
        super().__init__()
        self.max_steps = max_steps
        self.threshold = threshold
        
        self.halt_predictor = nn.Linear(hidden_size, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, compute_fn):
        batch_size, seq_len, hidden_size = x.shape
        
        # Initialize variables
        running_state = x
        halting_probs = []
        states = []
        cumulative_probs = torch.zeros(batch_size, seq_len, 1, device=x.device)
        step_probs = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        for step in range(self.max_steps):
            # Compute halting probability
            halt_logits = self.halt_predictor(running_state) + self.bias
            halt_prob = torch.sigmoid(halt_logits)
            
            # Check if we should halt
            still_running = (cumulative_probs + halt_prob) < self.threshold
            if step == self.max_steps - 1:
                # Force halt on last step
                step_prob = 1.0 - cumulative_probs
            else:
                step_prob = torch.where(still_running, halt_prob, 1.0 - cumulative_probs)
            
            # Update cumulative probabilities
            cumulative_probs += step_prob
            step_probs = step_prob
            
            # Compute new state
            new_state = compute_fn(running_state)
            states.append(new_state)
            halting_probs.append(step_probs)
            
            # Update running state
            running_state = new_state
            
            # Check if all sequences have halted
            if torch.all(cumulative_probs >= self.threshold):
                break
        
        # Combine states using halting probabilities
        output = torch.zeros_like(x)
        for i, (state, prob) in enumerate(zip(states, halting_probs)):
            output += state * prob
        
        # Compute ponder cost (number of steps taken)
        ponder_cost = sum(halting_probs).mean()
        
        return output, ponder_cost


class HierarchicalModule(nn.Module):
    """Hierarchical module (either L-module or H-module)"""
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        is_high_level: bool = False,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_act: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_high_level = is_high_level
        self.use_act = use_act
        
        # Build transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_size)
        
        # ACT mechanism
        if use_act:
            self.act = AdaptiveComputationTime(hidden_size)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if self.use_act:
            def compute_fn(state):
                for layer in self.layers:
                    state = layer(state, mask)
                return self.norm(state)
            
            output, ponder_cost = self.act(x, compute_fn)
            return output, ponder_cost
        else:
            for layer in self.layers:
                x = layer(x, mask)
            return self.norm(x), 0.0


class HierarchicalReasoningModel(nn.Module):
    """
    Hierarchical Reasoning Model (HRM) as described in the paper.
    
    Key features:
    - Two-level hierarchy with L-module (low-level) and H-module (high-level)
    - Multi-timescale processing
    - Adaptive Computation Time (ACT)
    - Brain-inspired feedback loops
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        timescale_ratio: int = 4,  # H-module updates every 4 L-module steps
        use_act: bool = True,
        mlp_ratio: int = 4
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.timescale_ratio = timescale_ratio
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # L-module (low-level, faster timescale)
        self.l_module = HierarchicalModule(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            is_high_level=False,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_act=use_act
        )
        
        # H-module (high-level, slower timescale)
        self.h_module = HierarchicalModule(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            is_high_level=True,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_act=use_act
        )
        
        # Cross-module connections
        self.l_to_h = nn.Linear(hidden_size, hidden_size)
        self.h_to_l = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using truncated LeCun Normal as mentioned in paper"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # LeCun Normal initialization
                fan_in = module.in_features
                std = math.sqrt(1.0 / fan_in)
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Initialize states
        z_l = x  # Low-level state
        z_h = torch.zeros_like(x)  # High-level state
        
        l_states = []
        h_states = []
        total_ponder_cost = 0.0
        
        # Multi-timescale processing
        for step in range(seq_len):
            # L-module always updates
            l_input = z_l + self.h_to_l(z_h)  # Feedback from H-module
            z_l_new, l_ponder = self.l_module(l_input.unsqueeze(1), attention_mask)
            z_l = z_l_new.squeeze(1)
            total_ponder_cost += l_ponder
            
            # H-module updates at slower timescale
            if step % self.timescale_ratio == 0:
                h_input = z_h + self.l_to_h(z_l)  # Input from L-module
                z_h_new, h_ponder = self.h_module(h_input.unsqueeze(1), attention_mask)
                z_h = z_h_new.squeeze(1)
                total_ponder_cost += h_ponder
            
            if return_states:
                l_states.append(z_l.clone())
                h_states.append(z_h.clone())
        
        # Final state combination
        final_state = z_l + z_h
        
        # Output projection
        logits = self.output_projection(final_state)
        
        result = {
            'logits': logits,
            'ponder_cost': total_ponder_cost,
            'final_l_state': z_l,
            'final_h_state': z_h
        }
        
        if return_states:
            result['l_states'] = torch.stack(l_states, dim=1)
            result['h_states'] = torch.stack(h_states, dim=1)
        
        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate sequences using the HRM model"""
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    logits[logits < values[:, -1, None]] = -float('inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS token (assuming EOS token id is 2)
                if torch.all(next_token == 2):
                    break
        
        return generated

    def compute_participation_ratio(self, states: torch.Tensor) -> float:
        """
        Compute Participation Ratio (PR) for analyzing dimensionality hierarchy
        as mentioned in the brain correspondence section
        """
        # Flatten states for covariance computation
        states_flat = states.view(-1, states.shape[-1])
        
        # Compute covariance matrix
        cov_matrix = torch.cov(states_flat.T)
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(cov_matrix).real
        eigenvals = torch.clamp(eigenvals, min=1e-12)  # Avoid numerical issues
        
        # Compute participation ratio
        pr = (eigenvals.sum() ** 2) / (eigenvals ** 2).sum()
        
        return pr.item()


def create_hrm_model(
    vocab_size: int,
    hidden_size: int = 512,
    num_layers: int = 8,
    **kwargs
) -> HierarchicalReasoningModel:
    """Factory function to create HRM model with default parameters from paper"""
    return HierarchicalReasoningModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        **kwargs
    ) 