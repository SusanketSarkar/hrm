#!/usr/bin/env python3
"""
Training script for Hierarchical Reasoning Model (HRM)

This script demonstrates how to train the HRM model on reasoning benchmarks
as described in the paper "Hierarchical Reasoning Model".

Usage:
    python train_hrm.py --benchmark sudoku --epochs 10 --batch_size 8
    python train_hrm.py --benchmark maze --use_wandb --hidden_size 256
    python train_hrm.py --benchmark arc --difficulty easy --max_steps 5000
"""

import argparse
import torch
import wandb
import os
import json
from dataclasses import asdict

# Import HRM components
from src.hrm_model import HierarchicalReasoningModel, create_hrm_model
from src.trainer import HRMTrainer, TrainingConfig, create_trainer
from src.benchmarks import create_benchmark_dataloader, evaluate_benchmark_performance
from src.optimizers import get_parameter_count


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Hierarchical Reasoning Model")
    
    # Benchmark settings
    parser.add_argument("--benchmark", type=str, default="sudoku", 
                       choices=["sudoku", "maze", "arc"],
                       help="Benchmark to train on")
    parser.add_argument("--difficulty", type=str, default="hard",
                       choices=["easy", "hard", "extreme"],
                       help="Difficulty level for benchmark")
    
    # Model parameters
    parser.add_argument("--hidden_size", type=int, default=512,
                       help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=8,
                       help="Number of layers in each module")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--timescale_ratio", type=int, default=4,
                       help="Timescale ratio between H and L modules")
    parser.add_argument("--use_act", action="store_true", default=True,
                       help="Use Adaptive Computation Time")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Warmup steps for learning rate")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                       help="Gradient clipping norm")
    
    # Data parameters
    parser.add_argument("--train_samples", type=int, default=1000,
                       help="Number of training samples")
    parser.add_argument("--eval_samples", type=int, default=200,
                       help="Number of evaluation samples")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Logging and saving
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--project_name", type=str, default="hrm-training",
                       help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None,
                       help="WandB run name")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    
    # Evaluation
    parser.add_argument("--eval_every", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log_every", type=int, default=50,
                       help="Log every N steps")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def get_vocab_size(benchmark_type: str) -> int:
    """Get vocabulary size for different benchmarks"""
    vocab_sizes = {
        "sudoku": 100,
        "maze": 20,
        "arc": 30
    }
    return vocab_sizes.get(benchmark_type, 100)


def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Get vocabulary size for the benchmark
    vocab_size = get_vocab_size(args.benchmark)
    
    # Create model
    print(f"Creating HRM model for {args.benchmark} benchmark...")
    model = create_hrm_model(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        timescale_ratio=args.timescale_ratio,
        use_act=args.use_act
    )
    
    # Print model information
    param_info = get_parameter_count(model)
    print(f"Model created with {param_info['total_params']:,} parameters")
    print(f"Trainable parameters: {param_info['trainable_params']:,}")
    
    # Create dataloaders
    print(f"Creating {args.benchmark} datasets...")
    
    # Benchmark-specific kwargs
    dataset_kwargs = {}
    if args.benchmark == "sudoku":
        dataset_kwargs["difficulty"] = args.difficulty
    elif args.benchmark == "maze":
        dataset_kwargs["maze_size"] = 30
    elif args.benchmark == "arc":
        dataset_kwargs["grid_size"] = 8
    
    train_dataloader = create_benchmark_dataloader(
        benchmark_type=args.benchmark,
        split="train",
        batch_size=args.batch_size,
        num_samples=args.train_samples,
        max_seq_len=args.max_seq_len,
        **dataset_kwargs
    )
    
    eval_dataloader = create_benchmark_dataloader(
        benchmark_type=args.benchmark,
        split="eval",
        batch_size=args.batch_size,
        num_samples=args.eval_samples,
        max_seq_len=args.max_seq_len,
        **dataset_kwargs
    )
    
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Evaluation samples: {len(eval_dataloader.dataset)}")
    
    # Create training configuration
    config = TrainingConfig(
        # Model parameters
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        timescale_ratio=args.timescale_ratio,
        use_act=args.use_act,
        
        # Training parameters
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        grad_clip_norm=args.grad_clip_norm,
        
        # Evaluation parameters
        eval_every=args.eval_every,
        save_every=args.save_every,
        
        # Logging
        log_every=args.log_every,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        run_name=args.run_name,
        
        # Paths
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        
        # Device
        device=device
    )
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"Configuration saved to {config_path}")
    
    # Create trainer
    trainer = create_trainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    print("\n" + "="*50)
    print("STARTING HIERARCHICAL REASONING MODEL TRAINING")
    print("="*50)
    print(f"Benchmark: {args.benchmark}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Model size: {param_info['total_params']:,} parameters")
    print(f"Training samples: {args.train_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {device}")
    if args.use_wandb:
        print(f"WandB project: {args.project_name}")
    print("="*50)
    
    try:
        # Train the model
        training_stats = trainer.train()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        # Final evaluation
        print("Running final evaluation...")
        final_eval = evaluate_benchmark_performance(
            model=trainer.model,
            dataloader=eval_dataloader,
            benchmark_type=args.benchmark,
            device=device
        )
        
        print("Final evaluation results:")
        for metric, value in final_eval.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save final model
        final_model_path = os.path.join(args.checkpoint_dir, "final_model.pt")
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save training stats
        stats_path = os.path.join(args.output_dir, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump(training_stats, f, indent=2)
        print(f"Training statistics saved to {stats_path}")
        
        # Generate sample visualization
        if eval_dataloader:
            try:
                sample_batch = next(iter(eval_dataloader))
                vis_path = os.path.join(args.output_dir, "model_visualization.png")
                trainer.visualize_attention_patterns(sample_batch, vis_path)
            except Exception as e:
                print(f"Could not generate visualization: {e}")
        
        # Log final results to wandb
        if args.use_wandb:
            wandb.log(final_eval)
            wandb.finish()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save checkpoint
        interrupt_path = os.path.join(args.checkpoint_dir, "interrupted_checkpoint.pt")
        trainer.save_checkpoint(interrupt_path)
        print(f"Checkpoint saved to {interrupt_path}")
        
        if args.use_wandb:
            wandb.finish()
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        if args.use_wandb:
            wandb.finish()
        raise


def demo_model():
    """Demo function showing model capabilities"""
    print("="*50)
    print("HRM MODEL DEMO")
    print("="*50)
    
    # Create a small model for demo
    model = create_hrm_model(
        vocab_size=100,
        hidden_size=256,
        num_layers=4,
        num_heads=4
    )
    
    print(f"Demo model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create sample input
    batch_size = 2
    seq_len = 50
    sample_input = torch.randint(1, 50, (batch_size, seq_len))
    
    print(f"Sample input shape: {sample_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(sample_input, return_states=True)
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Ponder cost: {outputs['ponder_cost']:.4f}")
    
    # Analyze dimensionality hierarchy
    l_states = outputs['l_states']
    h_states = outputs['h_states']
    
    l_pr = model.compute_participation_ratio(l_states.view(-1, l_states.shape[-1]))
    h_pr = model.compute_participation_ratio(h_states.view(-1, h_states.shape[-1]))
    
    print(f"L-module participation ratio: {l_pr:.2f}")
    print(f"H-module participation ratio: {h_pr:.2f}")
    print(f"Hierarchy ratio (H/L): {h_pr/l_pr:.2f}")
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_model()
    else:
        main() 