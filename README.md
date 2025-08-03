# Hierarchical Reasoning Model (HRM)

Implementation of the **Hierarchical Reasoning Model** from the paper "Hierarchical Reasoning Model" - a brain-inspired neural architecture for complex reasoning tasks.

## 🧠 Overview

The Hierarchical Reasoning Model (HRM) is a novel neural architecture that leverages hierarchical structure and multi-timescale processing to achieve substantial computational depth while maintaining training stability. The model is inspired by neuroscientific principles and demonstrates exceptional performance on complex reasoning tasks.

### Key Features

- **🏗️ Hierarchical Architecture**: Two-level hierarchy with L-module (low-level) and H-module (high-level)
- **⏱️ Multi-timescale Processing**: Different modules operate at different temporal scales
- **🔄 Adaptive Computation Time (ACT)**: Dynamic computation allocation based on task difficulty
- **🧬 Brain-inspired Design**: Incorporates neuroscientific principles like dimensionality hierarchy
- **🎯 Superior Performance**: Achieves state-of-the-art results on ARC-AGI, Sudoku, and Maze benchmarks
- **📈 Data Efficiency**: Trained with only ~1000 examples per task without pretraining

## 📊 Performance

According to the paper, HRM achieves:

- **ARC-AGI-2**: Significant improvement over chain-of-thought methods
- **Sudoku-Extreme**: Solves complex puzzles that challenge traditional approaches  
- **Maze-Hard**: Successfully navigates 30×30 mazes with optimal pathfinding
- **Parameter Efficiency**: ~27M parameters achieving results comparable to much larger models

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd hrm

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Demo the Model

```bash
python train_hrm.py demo
```

#### 2. Train on Sudoku

```bash
python train_hrm.py --benchmark sudoku --epochs 10 --batch_size 8
```

#### 3. Train on Maze Navigation

```bash
python train_hrm.py --benchmark maze --hidden_size 256 --use_wandb
```

#### 4. Train on ARC-like Tasks

```bash
python train_hrm.py --benchmark arc --difficulty easy --max_steps 5000
```

## 🏗️ Architecture Details

### Core Components

1. **L-Module (Low-level)**:
   - Operates at faster timescale
   - Lower-dimensional representations
   - Handles fine-grained processing

2. **H-Module (High-level)**:
   - Operates at slower timescale (4x slower by default)
   - Higher-dimensional representations  
   - Manages abstract reasoning and planning

3. **Adaptive Computation Time (ACT)**:
   - Dynamically allocates computation based on input complexity
   - Prevents overfitting and improves efficiency

4. **Cross-module Connections**:
   - Bidirectional information flow between L and H modules
   - Enables hierarchical reasoning and feedback

### Technical Innovations

- **RMSNorm**: Root Mean Square Layer Normalization
- **RoPE**: Rotary Position Embedding
- **GLU Variants**: Gated Linear Units with SwiGLU activation
- **Adam-atan2**: Scale-invariant optimizer variant
- **Participation Ratio Analysis**: Measures effective dimensionality

## 📁 Project Structure

```
hrm/
├── src/
│   ├── __init__.py
│   ├── hrm_model.py       # Core HRM architecture
│   ├── trainer.py         # Training utilities and loops
│   ├── optimizers.py      # Adam-atan2 and other optimizers
│   └── benchmarks.py      # Sudoku, Maze, and ARC datasets
├── docs/
│   └── 2506.21734v2.pdf  # Original paper
├── train_hrm.py           # Main training script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🎯 Benchmarks

### Sudoku

The implementation includes Sudoku puzzle generation with varying difficulty levels:

```python
from src.benchmarks import create_benchmark_dataloader

# Create Sudoku dataloader
dataloader = create_benchmark_dataloader(
    benchmark_type="sudoku",
    difficulty="hard",  # easy, hard, extreme
    batch_size=16,
    num_samples=1000
)
```

### Maze Navigation

30×30 maze navigation with optimal pathfinding:

```python
# Create Maze dataloader
dataloader = create_benchmark_dataloader(
    benchmark_type="maze",
    maze_size=30,
    batch_size=16,
    num_samples=1000
)
```

### ARC-like Tasks

Simplified ARC-style pattern recognition:

```python
# Create ARC dataloader
dataloader = create_benchmark_dataloader(
    benchmark_type="arc",
    grid_size=8,
    batch_size=16,
    num_samples=1000
)
```

## 🔧 Model Configuration

### Basic Model Creation

```python
from src.hrm_model import create_hrm_model

model = create_hrm_model(
    vocab_size=100,
    hidden_size=512,
    num_layers=8,
    num_heads=8,
    timescale_ratio=4,  # H-module updates every 4 L-module steps
    use_act=True        # Enable Adaptive Computation Time
)
```

### Training Configuration

```python
from src.trainer import TrainingConfig

config = TrainingConfig(
    # Model parameters
    hidden_size=512,
    num_layers=8,
    timescale_ratio=4,
    
    # Training parameters
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    
    # Use Adam-atan2 optimizer
    use_wandb=True,  # Enable Weights & Biases logging
    device="cuda"
)
```

## 📈 Training and Evaluation

### Training Loop

```python
from src.trainer import HRMTrainer

trainer = HRMTrainer(
    config=config,
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader
)

# Start training
training_stats = trainer.train()
```

### Evaluation Metrics

The implementation tracks several key metrics:

- **Cross-entropy Loss**: Standard language modeling loss
- **Ponder Cost**: ACT computational overhead
- **Participation Ratio**: Dimensionality hierarchy analysis
- **Accuracy**: Task-specific performance metrics

### Brain Correspondence Analysis

```python
# Analyze dimensionality hierarchy (from paper's brain correspondence section)
hierarchy_stats = trainer.analyze_hierarchy()

print(f"L-module PR: {hierarchy_stats['l_module_pr']:.2f}")
print(f"H-module PR: {hierarchy_stats['h_module_pr']:.2f}")
print(f"Hierarchy ratio: {hierarchy_stats['hierarchy_ratio']:.2f}")
```

## 🎨 Visualization

Generate visualizations of model behavior:

```python
# Visualize attention patterns and state evolution
trainer.visualize_attention_patterns(sample_batch, "attention_viz.png")
```

This creates plots showing:
- L-module and H-module state evolution
- State norms over time
- Participation ratio comparison
- Dimensionality hierarchy

## 🧪 Advanced Usage

### Custom Benchmarks

Create your own reasoning tasks:

```python
from src.benchmarks import BenchmarkDataset, BenchmarkSample

class CustomDataset(BenchmarkDataset):
    def _generate_samples(self, num_samples):
        samples = []
        for i in range(num_samples):
            # Generate your input/output pairs
            input_seq, target_seq = your_generation_logic()
            
            samples.append(BenchmarkSample(
                input_sequence=input_seq,
                target_sequence=target_seq,
                metadata={"custom_field": "value"}
            ))
        return samples
```

### Model Analysis

```python
# Compute participation ratios for dimensionality analysis
l_pr = model.compute_participation_ratio(l_states)
h_pr = model.compute_participation_ratio(h_states)

# The paper shows H-module should have higher PR than L-module
print(f"Hierarchy established: {h_pr > l_pr}")
```

### Generation

```python
# Generate sequences using the trained model
generated_text = trainer.generate_sample(
    prompt="Your prompt here",
    tokenizer=your_tokenizer,
    max_length=100,
    temperature=0.8
)
```

## 📋 Command Line Options

The training script supports extensive configuration:

```bash
python train_hrm.py --help
```

Key options:
- `--benchmark`: Choose between sudoku, maze, arc
- `--hidden_size`: Model hidden dimension (default: 512)
- `--num_layers`: Number of transformer layers per module (default: 8)
- `--timescale_ratio`: H-module update frequency (default: 4)
- `--use_act`: Enable Adaptive Computation Time
- `--use_wandb`: Enable Weights & Biases logging
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate for Adam-atan2 optimizer
- `--epochs`: Number of training epochs
- `--device`: Device to use (auto, cpu, cuda)

## 🔬 Research Features

### Neuroscientific Validation

The implementation includes tools to validate the brain-inspired design:

1. **Participation Ratio Analysis**: Measures effective dimensionality
2. **Timescale Separation**: Validates multi-timescale processing
3. **Hierarchical Organization**: Confirms emergence of hierarchy during training

### Ablation Studies

Easy configuration for ablation studies:

```python
# Disable ACT
model_without_act = create_hrm_model(use_act=False)

# Change timescale ratio
model_faster_h = create_hrm_model(timescale_ratio=2)

# Smaller model
model_small = create_hrm_model(hidden_size=256, num_layers=4)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- New benchmark implementations
- Architecture improvements
- Bug fixes
- Documentation enhancements

## 📚 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{hrm2025,
  title={Hierarchical Reasoning Model},
  author={Wang, Guan and Li, Jin and Sun, Yuhao and Chen, Xing and Liu, Changling and Wu, Yue and Lu, Meng and Song, Sen and Yadkori, Yasin Abbasi},
  journal={arXiv preprint arXiv:2506.21734},
  year={2025}
}
```

## 📄 License

This implementation is provided for research purposes. Please refer to the original paper for licensing terms.

## 🚨 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for larger models)
- GPU with 6GB+ VRAM (for default configuration)

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` or `hidden_size`
2. **Slow Training**: Enable CUDA and use appropriate batch size
3. **Convergence Issues**: Adjust learning rate or warmup steps
4. **Import Errors**: Ensure all dependencies are installed

### Performance Tips

- Use mixed precision training for faster computation
- Adjust `timescale_ratio` based on your task complexity
- Enable wandb logging for better experiment tracking
- Use gradient accumulation for effective larger batch sizes

## 📞 Support

For questions about the implementation, please:

1. Check the issues section for existing solutions
2. Create a new issue with detailed description
3. Include system information and error logs

---

**Note**: This implementation is based on the research paper and aims to reproduce the key architectural innovations. Some implementation details may differ from the original due to the need for practical considerations and code clarity. 