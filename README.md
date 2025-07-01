# CODE FOR SINGLE-AGENT RL-APPROACH FOR PATH PLANNING AND FLOOR PLAN DESIGN IN DYNAMIC ENVIRONMENTS

This repository contains the implementation of a reinforcement learning framework for investigating agent-driven architectural design through dynamic door placement in escape scenarios. The codebase supports the research presented in "Single-Agent RL-Approach for Path Planning and Floor Plan Design in Dynamic Environments."

## Research Context

Traditional architectural escape path planning relies on static analysis methods that require iterative design modifications. This framework enables agents to learn navigation strategies while simultaneously modifying their environment through door placement, creating a feedback loop between pathfinding behavior and architectural design decisions.

## Implementation Overview

The framework is implemented using MiniWorld for 3D simulation, Gymnasium for standardized RL interfaces, and Stable Baselines3 for PPO-based training. The system enables real-time environmental modifications during episode execution, allowing agents to place doors based on learned navigation policies.

## Core Scripts

### Training Environments

#### `env_sarl_v1.py` - Initial Implementation
The foundational single-agent environment implementing the core research concept. This script establishes the basic framework for agent-driven door placement within a multi-room escape scenario.

#### `env_sarl_v2.py` - Enhanced Implementation  
An optimized version of the environment with refined reward structures and improved observation spaces. This implementation incorporates lessons learned from initial experiments and provides enhanced training stability.


### Analysis and Visualization Tools

#### `env_sarl_load_paths.py` - Trajectory Analysis System
A comprehensive evaluation and visualization framework for analyzing trained agent behavior. This script loads saved models and generates detailed trajectory analyses across multiple training iterations.

#### `vis_freq_multi.py` - Multi-Run Comparison Analysis
Statistical analysis tool for comparing episode length patterns across different experimental configurations. Designed for systematic evaluation of training parameter effects.

#### `vis_freq_multiruns.py` - Multi-Agent Analysis Framework
Specialized visualization tool for multi-agent scenarios and comparative performance analysis. Supports detailed examination of agent interaction patterns and performance correlations.


## Installation and Dependencies

```bash
# Core RL dependencies
pip install stable-baselines3>=2.0.0
pip install gymnasium>=0.29.0
pip install miniworld>=1.2.0

# Analysis and visualization
pip install matplotlib>=3.5.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install pandas>=1.3.0

# Optional: Experiment tracking
pip install wandb>=0.15.0
```

## Basic Usage

### Training Execution
```bash
# Basic training with default parameters
python env_sarl_v1.py

# Enhanced training with optimized parameters
python env_sarl_v2.py
```

### Model Evaluation and Visualization
```bash
# Generate trajectory analysis for saved model
python env_sarl_load_paths.py model_checkpoint.zip --num-episodes 1000

# Compare multiple experimental runs
python vis_freq_multi.py experiment_data.csv iteration_log.txt --output comparison.png

# Multi-agent performance analysis
python vis_freq_multiruns.py --navigator-reward nav_data.json --door-reward door_data.json
```

## Configuration Parameters

### Environment Configuration
```python
environment_params = {
    'max_episode_steps': 500,
    'steps_until_hallway': 500,
    'door_width': 1.0,
    'wall_touching_threshold': 0.1,
    'agent_radius': 0.25
}
```

### Training Configuration
```python
training_params = {
    'timesteps': 12000,
    'iterations': 1000,
    'learning_rate': 0.0003,
    'batch_size': 2048,
    'gamma': 0.99
}
```

### Reward Scaling Configuration
```python
reward_scales = {
    'reward_orientation_scale': 1.0,
    'penalty_stagnation_scale': 1.0,
    'reward_hallway_scale': 1.0,
    'reward_terminal_scale': 1.0,
    'wall_collision_scale': 1.0
}
```

## Data Output Formats

### Training Outputs
- **Model checkpoints**: `.zip` files containing trained PPO models
- **Training logs**: CSV files with episode-level performance metrics
- **Door creation logs**: Detailed records of door placement events with positional data
- **Terminal achievement logs**: Records of successful escape events

### Visualization Outputs
- **Trajectory maps**: High-resolution PNG images showing agent paths
- **Heat maps**: Frequency visualizations of door creation patterns
- **Statistical plots**: Performance comparison charts with confidence intervals
- **Analysis reports**: Text-based summaries of experimental outcomes


## Citation

tba.

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
