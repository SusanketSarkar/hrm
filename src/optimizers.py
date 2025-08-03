import torch
import torch.optim as optim
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import math


class AdamAtan2(optim.Optimizer):
    """
    Adam-atan2 optimizer: A scale-invariant variant of Adam as mentioned in the paper.
    
    This optimizer uses the atan2 function to make the updates scale-invariant,
    which can help with training stability in hierarchical models.
    """
    
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
        max_grad_norm: Optional[float] = None
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Gradient clipping
                if group['max_grad_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_([p], group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Apply warmup to learning rate
                lr = group['lr']
                if state['step'] <= group['warmup_steps']:
                    lr = lr * state['step'] / group['warmup_steps']

                # Compute step size with bias correction
                step_size = lr / bias_correction1

                # Compute the denominator (second moment with bias correction)
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])

                # Scale-invariant update using atan2
                # This is the key difference from standard Adam
                update_direction = exp_avg / denom
                
                # Use atan2 for scale invariance
                # atan2(y, x) gives the angle of vector (x, y) from x-axis
                # This makes the update scale-invariant
                norm_update = torch.norm(update_direction, dim=-1, keepdim=True)
                norm_param = torch.norm(p.data, dim=-1, keepdim=True)
                
                # Avoid division by zero
                norm_update = torch.clamp(norm_update, min=group['eps'])
                norm_param = torch.clamp(norm_param, min=group['eps'])
                
                # Scale-invariant step using atan2-inspired scaling
                scale_factor = torch.atan2(norm_update, norm_param) / (norm_update + group['eps'])
                scaled_update = update_direction * scale_factor

                # Apply update
                p.data.add_(scaled_update, alpha=-step_size)

        return loss


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    optimizer_type: str = "adam_atan2",
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer for HRM training.
    
    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient  
        warmup_steps: Number of warmup steps for learning rate
        optimizer_type: Type of optimizer ('adam_atan2', 'adam', 'adamw')
        **kwargs: Additional optimizer arguments
    
    Returns:
        Configured optimizer
    """
    # Separate parameters for different weight decay
    # Exclude bias and normalization parameters from weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Don't apply weight decay to bias terms and normalization layers
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    if optimizer_type.lower() == "adam_atan2":
        return AdamAtan2(
            param_groups,
            lr=lr,
            warmup_steps=warmup_steps,
            **kwargs
        )
    elif optimizer_type.lower() == "adam":
        return optim.Adam(
            param_groups,
            lr=lr,
            **kwargs
        )
    elif optimizer_type.lower() == "adamw":
        return optim.AdamW(
            param_groups,
            lr=lr,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class WarmupScheduler:
    """
    Learning rate scheduler with linear warmup as mentioned in the paper.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_lr: float,
        final_lr: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.final_lr = final_lr if final_lr is not None else base_lr
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Constant learning rate after warmup (as mentioned in paper)
            lr = self.final_lr
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


def get_parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get parameter count breakdown for the model.
    
    Args:
        model: The model to analyze
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = 0
    trainable_params = 0
    
    # Count by module type
    module_counts = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_type = type(module).__name__
            param_count = sum(p.numel() for p in module.parameters())
            
            if module_type not in module_counts:
                module_counts[module_type] = 0
            module_counts[module_type] += param_count
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'module_breakdown': module_counts
    } 