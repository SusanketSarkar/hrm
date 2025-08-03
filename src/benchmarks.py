import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any
import json
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BenchmarkSample:
    """Base class for benchmark samples"""
    input_sequence: List[int]
    target_sequence: List[int]
    metadata: Dict[str, Any]


class BenchmarkDataset(Dataset, ABC):
    """Abstract base class for benchmark datasets"""
    
    def __init__(self, samples: List[BenchmarkSample], max_seq_len: int = 2048):
        self.samples = samples
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Pad or truncate sequences
        input_ids = self._pad_sequence(sample.input_sequence)
        labels = self._pad_sequence(sample.target_sequence)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'metadata': sample.metadata
        }
    
    def _pad_sequence(self, sequence: List[int]) -> List[int]:
        """Pad or truncate sequence to max_seq_len"""
        if len(sequence) > self.max_seq_len:
            return sequence[:self.max_seq_len]
        else:
            # Pad with -100 (ignore index for loss computation)
            return sequence + [-100] * (self.max_seq_len - len(sequence))


class SudokuDataset(BenchmarkDataset):
    """Sudoku benchmark dataset implementation"""
    
    def __init__(self, num_samples: int = 1000, difficulty: str = "hard", max_seq_len: int = 2048):
        self.difficulty = difficulty
        self.vocab_size = 100  # Special tokens + digits + separators
        
        # Generate samples
        samples = self._generate_samples(num_samples)
        super().__init__(samples, max_seq_len)
    
    def _generate_samples(self, num_samples: int) -> List[BenchmarkSample]:
        """Generate Sudoku puzzle samples"""
        samples = []
        
        for i in range(num_samples):
            # Generate a valid Sudoku solution
            solution = self._generate_valid_sudoku()
            
            # Create puzzle by removing cells
            puzzle = self._create_puzzle_from_solution(solution)
            
            # Convert to sequence format
            input_seq, target_seq = self._sudoku_to_sequences(puzzle, solution)
            
            samples.append(BenchmarkSample(
                input_sequence=input_seq,
                target_sequence=target_seq,
                metadata={
                    'puzzle': puzzle.tolist(),
                    'solution': solution.tolist(),
                    'difficulty': self.difficulty,
                    'sample_id': i
                }
            ))
        
        return samples
    
    def _generate_valid_sudoku(self) -> np.ndarray:
        """Generate a valid 9x9 Sudoku solution"""
        # Start with a base valid Sudoku and randomize it
        base = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [4, 5, 6, 7, 8, 9, 1, 2, 3],
            [7, 8, 9, 1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7, 8, 9, 1],
            [5, 6, 7, 8, 9, 1, 2, 3, 4],
            [8, 9, 1, 2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8, 9, 1, 2],
            [6, 7, 8, 9, 1, 2, 3, 4, 5],
            [9, 1, 2, 3, 4, 5, 6, 7, 8]
        ])
        
        # Apply random transformations to create variety
        return self._randomize_sudoku(base)
    
    def _randomize_sudoku(self, sudoku: np.ndarray) -> np.ndarray:
        """Apply random valid transformations to a Sudoku"""
        # Random digit permutation
        perm = np.random.permutation(9) + 1
        mapping = {i+1: perm[i] for i in range(9)}
        randomized = np.vectorize(mapping.get)(sudoku)
        
        # Random row swaps within bands (3x3 blocks)
        for band in range(3):
            rows = list(range(band * 3, (band + 1) * 3))
            np.random.shuffle(rows)
            randomized[band*3:(band+1)*3] = randomized[rows]
        
        return randomized
    
    def _create_puzzle_from_solution(self, solution: np.ndarray) -> np.ndarray:
        """Create puzzle by removing cells from solution"""
        puzzle = solution.copy()
        
        # Remove cells based on difficulty
        if self.difficulty == "easy":
            cells_to_remove = 35
        elif self.difficulty == "hard":
            cells_to_remove = 50
        else:  # extreme
            cells_to_remove = 65
        
        positions = [(i, j) for i in range(9) for j in range(9)]
        remove_positions = random.sample(positions, cells_to_remove)
        
        for i, j in remove_positions:
            puzzle[i, j] = 0
        
        return puzzle
    
    def _sudoku_to_sequences(self, puzzle: np.ndarray, solution: np.ndarray) -> Tuple[List[int], List[int]]:
        """Convert Sudoku puzzle and solution to token sequences"""
        # Special tokens
        START_TOKEN = 1
        END_TOKEN = 2
        SEP_TOKEN = 3
        EMPTY_TOKEN = 10
        
        # Input sequence: START + flattened puzzle + SEP + solution prefix
        input_seq = [START_TOKEN]
        
        # Add puzzle
        for i in range(9):
            for j in range(9):
                if puzzle[i, j] == 0:
                    input_seq.append(EMPTY_TOKEN)
                else:
                    input_seq.append(puzzle[i, j] + 10)  # Offset to avoid special tokens
        
        input_seq.append(SEP_TOKEN)
        
        # Target sequence: solution + END
        target_seq = []
        for i in range(9):
            for j in range(9):
                target_seq.append(solution[i, j] + 10)
        
        target_seq.append(END_TOKEN)
        
        return input_seq, target_seq


class MazeDataset(BenchmarkDataset):
    """Maze navigation benchmark dataset"""
    
    def __init__(self, num_samples: int = 1000, maze_size: int = 30, max_seq_len: int = 2048):
        self.maze_size = maze_size
        self.vocab_size = 20  # Special tokens + directions + cell types
        
        samples = self._generate_samples(num_samples)
        super().__init__(samples, max_seq_len)
    
    def _generate_samples(self, num_samples: int) -> List[BenchmarkSample]:
        """Generate maze navigation samples"""
        samples = []
        
        for i in range(num_samples):
            # Generate random maze
            maze = self._generate_maze()
            
            # Find start and goal positions
            start, goal = self._place_start_goal(maze)
            
            # Find optimal path
            path = self._find_optimal_path(maze, start, goal)
            
            if path is None:
                continue  # Skip if no solution
            
            # Convert to sequences
            input_seq, target_seq = self._maze_to_sequences(maze, start, goal, path)
            
            samples.append(BenchmarkSample(
                input_sequence=input_seq,
                target_sequence=target_seq,
                metadata={
                    'maze': maze.tolist(),
                    'start': start,
                    'goal': goal,
                    'path': path,
                    'path_length': len(path),
                    'sample_id': i
                }
            ))
        
        return samples
    
    def _generate_maze(self) -> np.ndarray:
        """Generate a random maze using simple algorithm"""
        maze = np.random.choice([0, 1], size=(self.maze_size, self.maze_size), p=[0.7, 0.3])
        
        # Ensure borders are walls
        maze[0, :] = 1
        maze[-1, :] = 1
        maze[:, 0] = 1
        maze[:, -1] = 1
        
        return maze
    
    def _place_start_goal(self, maze: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Place start and goal in free cells"""
        free_cells = [(i, j) for i in range(1, self.maze_size-1) 
                      for j in range(1, self.maze_size-1) if maze[i, j] == 0]
        
        start, goal = random.sample(free_cells, 2)
        return start, goal
    
    def _find_optimal_path(self, maze: np.ndarray, start: Tuple[int, int], 
                          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find optimal path using BFS"""
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            (x, y), path = queue.popleft()
            
            if (x, y) == goal:
                return path
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.maze_size and 0 <= ny < self.maze_size and
                    maze[nx, ny] == 0 and (nx, ny) not in visited):
                    
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        
        return None  # No path found
    
    def _maze_to_sequences(self, maze: np.ndarray, start: Tuple[int, int], 
                          goal: Tuple[int, int], path: List[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
        """Convert maze and path to token sequences"""
        START_TOKEN = 1
        END_TOKEN = 2
        SEP_TOKEN = 3
        WALL_TOKEN = 4
        FREE_TOKEN = 5
        START_POS_TOKEN = 6
        GOAL_POS_TOKEN = 7
        
        # Direction tokens
        UP, DOWN, LEFT, RIGHT = 11, 12, 13, 14
        
        input_seq = [START_TOKEN]
        
        # Encode maze
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if (i, j) == start:
                    input_seq.append(START_POS_TOKEN)
                elif (i, j) == goal:
                    input_seq.append(GOAL_POS_TOKEN)
                elif maze[i, j] == 1:
                    input_seq.append(WALL_TOKEN)
                else:
                    input_seq.append(FREE_TOKEN)
        
        input_seq.append(SEP_TOKEN)
        
        # Target sequence: sequence of moves
        target_seq = []
        for i in range(1, len(path)):
            prev_pos = path[i-1]
            curr_pos = path[i]
            
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            
            if dx == 1:
                target_seq.append(DOWN)
            elif dx == -1:
                target_seq.append(UP)
            elif dy == 1:
                target_seq.append(RIGHT)
            elif dy == -1:
                target_seq.append(LEFT)
        
        target_seq.append(END_TOKEN)
        
        return input_seq, target_seq


class SimpleARCDataset(BenchmarkDataset):
    """Simplified ARC-like dataset for testing"""
    
    def __init__(self, num_samples: int = 1000, grid_size: int = 8, max_seq_len: int = 2048):
        self.grid_size = grid_size
        self.vocab_size = 30  # Colors + special tokens
        
        samples = self._generate_samples(num_samples)
        super().__init__(samples, max_seq_len)
    
    def _generate_samples(self, num_samples: int) -> List[BenchmarkSample]:
        """Generate simple pattern transformation samples"""
        samples = []
        
        for i in range(num_samples):
            # Generate input/output pattern pairs
            input_grid, output_grid = self._generate_pattern_pair()
            
            # Convert to sequences
            input_seq, target_seq = self._arc_to_sequences(input_grid, output_grid)
            
            samples.append(BenchmarkSample(
                input_sequence=input_seq,
                target_sequence=target_seq,
                metadata={
                    'input_grid': input_grid.tolist(),
                    'output_grid': output_grid.tolist(),
                    'transformation': 'simple_pattern',
                    'sample_id': i
                }
            ))
        
        return samples
    
    def _generate_pattern_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simple input/output pattern pair"""
        input_grid = np.random.randint(0, 5, size=(self.grid_size, self.grid_size))
        
        # Apply simple transformation (e.g., color shift)
        output_grid = (input_grid + 1) % 5
        
        return input_grid, output_grid
    
    def _arc_to_sequences(self, input_grid: np.ndarray, 
                         output_grid: np.ndarray) -> Tuple[List[int], List[int]]:
        """Convert ARC grids to sequences"""
        START_TOKEN = 1
        END_TOKEN = 2
        SEP_TOKEN = 3
        
        input_seq = [START_TOKEN]
        
        # Flatten input grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                input_seq.append(input_grid[i, j] + 10)  # Offset colors
        
        input_seq.append(SEP_TOKEN)
        
        # Target sequence: flattened output grid
        target_seq = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                target_seq.append(output_grid[i, j] + 10)
        
        target_seq.append(END_TOKEN)
        
        return input_seq, target_seq


def create_benchmark_dataloader(
    benchmark_type: str,
    split: str = "train",
    batch_size: int = 16,
    num_samples: int = 1000,
    **kwargs
) -> DataLoader:
    """
    Create dataloader for specified benchmark.
    
    Args:
        benchmark_type: Type of benchmark ('sudoku', 'maze', 'arc')
        split: Data split ('train', 'eval', 'test')
        batch_size: Batch size
        num_samples: Number of samples to generate
        **kwargs: Additional dataset-specific arguments
    
    Returns:
        DataLoader for the benchmark
    """
    
    if benchmark_type.lower() == "sudoku":
        dataset = SudokuDataset(num_samples=num_samples, **kwargs)
    elif benchmark_type.lower() == "maze":
        dataset = MazeDataset(num_samples=num_samples, **kwargs)
    elif benchmark_type.lower() == "arc":
        dataset = SimpleARCDataset(num_samples=num_samples, **kwargs)
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=torch.cuda.is_available()
    )


def evaluate_benchmark_performance(
    model,
    dataloader: DataLoader,
    benchmark_type: str,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate model performance on a specific benchmark.
    
    Args:
        model: Trained HRM model
        dataloader: DataLoader for evaluation
        benchmark_type: Type of benchmark
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            logits = outputs['logits']
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            
            # Mask out padding tokens
            mask = labels != -100
            correct = (predictions == labels) & mask
            
            total_correct += correct.sum().item()
            total_samples += mask.sum().item()
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            total_loss += loss.item()
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(dataloader)
    
    return {
        f'{benchmark_type}_accuracy': accuracy,
        f'{benchmark_type}_loss': avg_loss,
        f'{benchmark_type}_perplexity': np.exp(avg_loss)
    } 