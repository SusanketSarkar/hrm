import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from tqdm import tqdm
import wandb
import os
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

from .hrm_model import HierarchicalReasoningModel
from .optimizers import create_optimizer, WarmupScheduler, get_parameter_count


@dataclass
class TrainingConfig:
    """Configuration for HRM training"""
    # Model parameters
    vocab_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    max_seq_len: int = 2048
    timescale_ratio: int = 4
    use_act: bool = True
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    max_steps: Optional[int] = None
    grad_clip_norm: float = 1.0
    
    # Loss parameters
    ponder_loss_weight: float = 0.01  # Weight for ACT ponder cost
    
    # Evaluation parameters
    eval_every: int = 500
    save_every: int = 1000
    eval_steps: int = 100
    
    # Logging
    log_every: int = 50
    use_wandb: bool = False
    project_name: str = "hrm-training"
    run_name: Optional[str] = None
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class HRMTrainer:
    """Trainer class for the Hierarchical Reasoning Model"""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[HierarchicalReasoningModel] = None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        if model is None:
            self.model = HierarchicalReasoningModel(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                max_seq_len=config.max_seq_len,
                timescale_ratio=config.timescale_ratio,
                use_act=config.use_act
            )
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Data loaders
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Initialize optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps
        )
        
        self.scheduler = WarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            base_lr=config.learning_rate
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize logging
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=asdict(config)
            )
        
        # Log model parameters
        param_info = get_parameter_count(self.model)
        print(f"Model parameter count: {param_info}")
        
        if config.use_wandb:
            wandb.log(param_info)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch of data.
        
        Args:
            batch: Dictionary containing 'input_ids' and 'labels'
        
        Returns:
            Dictionary with loss components
        """
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs['logits']
        ponder_cost = outputs['ponder_cost']
        
        # Compute cross-entropy loss
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute cross-entropy loss (ignore padding tokens)
        ce_loss = F.cross_entropy(
            shift_logits, 
            shift_labels, 
            ignore_index=-100, 
            reduction='mean'
        )
        
        # Compute total loss with ponder cost
        total_loss = ce_loss + self.config.ponder_loss_weight * ponder_cost
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'ponder_cost': ponder_cost,
            'logits': logits
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step"""
        self.model.train()
        
        # Forward pass and loss computation
        loss_dict = self.compute_loss(batch)
        loss = loss_dict['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Convert to float for logging
        return {k: v.item() if torch.is_tensor(v) else v 
                for k, v in loss_dict.items() if k != 'logits'}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset"""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_ponder_cost = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.eval_dataloader):
                if i >= self.config.eval_steps:
                    break
                
                loss_dict = self.compute_loss(batch)
                batch_size = batch['input_ids'].size(0)
                
                total_loss += loss_dict['loss'].item() * batch_size
                total_ce_loss += loss_dict['ce_loss'].item() * batch_size
                total_ponder_cost += loss_dict['ponder_cost'].item() * batch_size
                total_samples += batch_size
        
        return {
            'eval_loss': total_loss / total_samples,
            'eval_ce_loss': total_ce_loss / total_samples,
            'eval_ponder_cost': total_ponder_cost / total_samples,
            'eval_perplexity': np.exp(total_ce_loss / total_samples)
        }

    def analyze_hierarchy(self) -> Dict[str, float]:
        """
        Analyze the dimensionality hierarchy of L and H modules.
        This implements the brain correspondence analysis from the paper.
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        l_states_list = []
        h_states_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.eval_dataloader):
                if i >= 10:  # Analyze a small subset
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                outputs = self.model(input_ids, return_states=True)
                
                l_states_list.append(outputs['l_states'].cpu())
                h_states_list.append(outputs['h_states'].cpu())
        
        # Concatenate all states
        l_states = torch.cat(l_states_list, dim=0)  # [batch * seq_len, hidden_size]
        h_states = torch.cat(h_states_list, dim=0)
        
        # Compute participation ratios
        l_pr = self.model.compute_participation_ratio(l_states)
        h_pr = self.model.compute_participation_ratio(h_states)
        
        return {
            'l_module_pr': l_pr,
            'h_module_pr': h_pr,
            'hierarchy_ratio': h_pr / (l_pr + 1e-8)  # Avoid division by zero
        }

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint"""
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir, 
                f"checkpoint_step_{self.global_step}.pt"
            )
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': {
                'step_count': self.scheduler.step_count,
                'warmup_steps': self.scheduler.warmup_steps,
                'base_lr': self.scheduler.base_lr,
                'final_lr': self.scheduler.final_lr
            },
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': asdict(self.config),
            'best_eval_loss': self.best_eval_loss
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state
        scheduler_state = checkpoint['scheduler_state_dict']
        self.scheduler.step_count = scheduler_state['step_count']
        self.scheduler.warmup_steps = scheduler_state['warmup_steps']
        self.scheduler.base_lr = scheduler_state['base_lr']
        self.scheduler.final_lr = scheduler_state['final_lr']
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint['best_eval_loss']
        
        print(f"Checkpoint loaded from {path}")

    def train(self):
        """Main training loop"""
        if self.train_dataloader is None:
            raise ValueError("Training dataloader is required for training")
        
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        training_stats = []
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            epoch_stats = []
            
            # Training loop
            self.model.train()
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.config.max_epochs}"
            )
            
            for batch in progress_bar:
                # Training step
                step_stats = self.train_step(batch)
                step_stats['learning_rate'] = self.scheduler.get_lr()
                step_stats['epoch'] = epoch
                step_stats['global_step'] = self.global_step
                
                epoch_stats.append(step_stats)
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_stats['loss']:.4f}",
                    'lr': f"{step_stats['learning_rate']:.2e}"
                })
                
                # Logging
                if self.global_step % self.config.log_every == 0:
                    if self.config.use_wandb:
                        wandb.log(step_stats, step=self.global_step)
                
                # Evaluation
                if self.global_step % self.config.eval_every == 0:
                    eval_stats = self.evaluate()
                    hierarchy_stats = self.analyze_hierarchy()
                    
                    combined_stats = {**eval_stats, **hierarchy_stats}
                    
                    if self.config.use_wandb:
                        wandb.log(combined_stats, step=self.global_step)
                    
                    print(f"\nStep {self.global_step} Evaluation:")
                    for k, v in combined_stats.items():
                        print(f"  {k}: {v:.4f}")
                    
                    # Save best model
                    if eval_stats.get('eval_loss', float('inf')) < self.best_eval_loss:
                        self.best_eval_loss = eval_stats['eval_loss']
                        self.save_checkpoint(
                            os.path.join(self.config.checkpoint_dir, "best_model.pt")
                        )
                
                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()
                
                # Check max steps
                if (self.config.max_steps is not None and 
                    self.global_step >= self.config.max_steps):
                    print(f"Reached maximum steps ({self.config.max_steps})")
                    return training_stats
            
            # End of epoch
            training_stats.extend(epoch_stats)
        
        print("Training completed!")
        return training_stats

    def generate_sample(
        self,
        prompt: str,
        tokenizer,
        max_length: int = 100,
        temperature: float = 0.8,
        **kwargs
    ) -> str:
        """Generate a sample from the model given a prompt"""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                **kwargs
            )
        
        # Decode
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text

    def visualize_attention_patterns(self, batch: Dict[str, torch.Tensor], save_path: str):
        """Visualize attention patterns in the model"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = batch['input_ids'][:1].to(self.device)  # Take first sample
            outputs = self.model(input_ids, return_states=True)
            
            # Get states for visualization
            l_states = outputs['l_states'][0].cpu().numpy()  # [seq_len, hidden_size]
            h_states = outputs['h_states'][0].cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # L-module states heatmap
            sns.heatmap(l_states.T, ax=axes[0, 0], cmap='viridis')
            axes[0, 0].set_title('L-module States')
            axes[0, 0].set_xlabel('Sequence Position')
            axes[0, 0].set_ylabel('Hidden Dimension')
            
            # H-module states heatmap
            sns.heatmap(h_states.T, ax=axes[0, 1], cmap='viridis')
            axes[0, 1].set_title('H-module States')
            axes[0, 1].set_xlabel('Sequence Position')
            axes[0, 1].set_ylabel('Hidden Dimension')
            
            # State norms over time
            l_norms = np.linalg.norm(l_states, axis=1)
            h_norms = np.linalg.norm(h_states, axis=1)
            
            axes[1, 0].plot(l_norms, label='L-module', color='blue')
            axes[1, 0].plot(h_norms, label='H-module', color='red')
            axes[1, 0].set_title('State Norms Over Time')
            axes[1, 0].set_xlabel('Sequence Position')
            axes[1, 0].set_ylabel('L2 Norm')
            axes[1, 0].legend()
            
            # Dimensionality analysis
            l_pr = self.model.compute_participation_ratio(torch.tensor(l_states))
            h_pr = self.model.compute_participation_ratio(torch.tensor(h_states))
            
            bars = axes[1, 1].bar(['L-module', 'H-module'], [l_pr, h_pr], 
                                color=['blue', 'red'], alpha=0.7)
            axes[1, 1].set_title('Participation Ratio (Effective Dimensionality)')
            axes[1, 1].set_ylabel('Participation Ratio')
            
            # Add value labels on bars
            for bar, value in zip(bars, [l_pr, h_pr]):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Attention visualization saved to {save_path}")


def create_trainer(
    config: TrainingConfig,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    model: Optional[HierarchicalReasoningModel] = None
) -> HRMTrainer:
    """Factory function to create HRM trainer"""
    return HRMTrainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    ) 