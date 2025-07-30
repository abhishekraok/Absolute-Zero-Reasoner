# Absolute Zero Reasoner - Codebase Documentation

## Overview

Absolute Zero Reasoner (AZR) is a self-play reinforcement learning system that trains language models to improve their reasoning abilities without any external training data. The key innovation is a two-phase approach where the model generates its own reasoning tasks and then learns to solve them.

## Core Algorithm: Two-Phase Self-Play

### Phase 1: PROPOSE (Generation Tasks)
The model generates reasoning tasks of different types. In the code, these are implemented as "generation" tasks (prefixed with `gen_`).

### Phase 2: SOLVE (Prediction Tasks)  
The model attempts to solve the self-generated tasks. In the code, these are implemented as "prediction" tasks (prefixed with `pred_`).

## Problem Types

The system supports four types of code-based reasoning problems:

1. **`code_i` (Input Prediction - Abductive Reasoning)**
   - Given: Code snippet + Expected output
   - Task: Find an input that produces this output
   - Reasoning type: Working backwards from observations

2. **`code_o` (Output Prediction - Deductive Reasoning)**
   - Given: Code snippet + Input
   - Task: Predict the output
   - Reasoning type: Forward reasoning from premises

3. **`code_f` (Function Synthesis - Inductive Reasoning)**
   - Given: Input/output example pairs
   - Task: Write a function that matches the pattern
   - Reasoning type: Generalizing from specific cases

4. **`code_e` (Error Prediction)**
   - Given: Code snippet + Input
   - Task: Predict if/what error will occur
   - Note: This type is supported but not commonly used

## Directory Structure

```
absolute_zero_reasoner/
├── main_azr_ppo.py                    # Main training entry point
├── configs/azr_ppo_trainer.yaml       # Comprehensive training configuration
├── trainer/ppo/
│   ├── azr_ray_trainer.py             # Core AZR trainer implementing the algorithm
│   └── reason_rl_ray_trainer.py       # Base reinforcement learning trainer
├── rewards/
│   ├── reward_managers.py             # Manages reward calculation and distribution
│   ├── code_reward.py                 # Code execution and verification rewards
│   └── custom_evaluate.py             # Custom evaluation metrics
├── data_construction/
│   ├── constructor.py                 # Data generation logic for all problem types
│   ├── prompts.py                     # Prompt templates for each problem type
│   └── process_*.py                   # Data processing and validation utilities
└── utils/
    ├── code_utils/
    │   ├── python_executor.py         # Direct Python execution (research only!)
    │   ├── sandboxfusion_executor.py  # Safer Docker-based execution
    │   ├── parsers.py                 # Code parsing and AST utilities
    │   └── templates.py               # Execution templates
    └── dataset/rl_dataset.py          # RL dataset management with thread safety

evaluation/
├── code_eval/
│   ├── coding/LiveCodeBench/          # Live coding benchmark evaluation
│   ├── coding/evalplus/               # HumanEval+ and MBPP+ evaluation
│   └── scripts/                       # Evaluation automation scripts
└── math_eval/
    ├── eval/data/                     # 25+ math datasets (GSM8K, MATH, etc.)
    └── *.py                           # Math evaluation implementation

scripts/
├── seeding/                           # Scripts to generate initial seed data
└── selfplay/                          # Main training scripts by model size

data/                                  # Pre-collected seed datasets
└── *_seed_io.jsonl                   # Seed data for different model variants
```

## Key Technical Components

### 1. Distributed Training Infrastructure
- **Ray Framework**: Manages distributed computing across multiple GPUs
- **veRL Integration**: Built on Volcano Engine's reinforcement learning framework
- **Hydra Configuration**: Comprehensive YAML-based configuration system
- **vLLM**: Efficient inference engine for rollouts

### 2. Core Training Algorithm (`azr_ray_trainer.py`)
- **CodeIORayPPOTrainer**: Main trainer class implementing the AZR algorithm
- **DatasetManager**: Thread-safe distributed dataset management with step tracking
- **Problem Type Management**: Handles generation and prediction for each problem type
- **Executor Integration**: Supports both QwQ (direct) and SandboxFusion (containerized) executors
- **Dynamic Batch Management**: Automatically adjusts batch sizes based on problem types

### 3. Data Generation System (`constructor.py` + `prompts.py`)
- **Reference-Based Generation**: Uses existing problems as examples for generating new ones
- **Composite Functions**: Optional support for multi-function problems
- **Banned Keywords**: Extensive list preventing unsafe code generation
- **Content Filtering**: Multiple validation strategies (all, non_one, non_extremes)
- **Dynamic Prompting**: Different templates for each reasoning type and instruction format

### 4. Code Execution System
- **PythonExecutor**: Direct Python execution with process isolation (research only - not secure!)
  - Uses `pebble` for process pool management
  - Implements timeout protection (default: 10 seconds)
  - Basic AST validation and banned import checking
  
- **SandboxfusionExecutor**: Safer Docker-based execution environment
  - Containerized execution with resource limits
  - Network isolation
  - Memory limits configurable

### 5. Reward System (`reward_managers.py` + `code_reward.py`)
- **Binary Rewards**: Success/failure for code execution correctness
- **Intrinsic Rewards** (all configurable):
  - Complexity reward (code complexity metrics)
  - Diversity reward (answer variation across samples)
  - Halstead complexity metrics
  - Mean edit distance reward
- **Learnability Scoring**: Estimates task difficulty for curriculum learning
- **Multi-Sample Evaluation**: Tests multiple solutions per problem (default: 8 samples)

### 6. Safety and Validation
- **Banned Keywords**: Prevents dangerous operations
  ```
  logging, random, multiprocessing, subprocess, threading, 
  datetime, time, hashlib, hmac, bcrypt, os.sys, os.path, 
  sys.exit, os.environ, calendar
  ```
- **AST Validation**: Parses code to ensure structural validity
- **Import Restrictions**: Blocks potentially harmful imports
- **Execution Timeouts**: Prevents infinite loops
- **Memory Limits**: Configurable resource constraints

## Training Flow

### 1. Initialization
```python
# Load base model (Qwen, CodeQwen, Llama, etc.)
# Setup Ray cluster for distributed training
# Initialize executors and reward managers
# Load or generate seed datasets
```

### 2. Self-Play Training Loop
For each epoch and problem type:
1. **Generate Phase** (PROPOSE):
   - Model creates new reasoning tasks using prompt templates
   - Validate generated code with executor
   - Filter based on execution results and AST checks
   - Calculate generation rewards (format, complexity, diversity)

2. **Solve Phase** (SOLVE):
   - Model attempts to solve the generated tasks
   - Execute solutions and verify correctness
   - Calculate solving rewards based on accuracy

3. **PPO Update**:
   - Use TRR++ (reinforce_plus_plus) advantage estimation
   - Update model parameters with PPO algorithm
   - Track metrics with Weights & Biases

### 3. Evaluation
- **Code Benchmarks**: 
  - LiveCodeBench (real-time coding challenges)
  - HumanEval+ (enhanced HumanEval with stronger tests)
  - MBPP+ (enhanced MBPP benchmark)
  
- **Math Benchmarks**: 
  - 25+ datasets including GSM8K, MATH, AIME, Olympiad problems

## GPU Requirements

| Model Size | GPUs Required | Memory per GPU |
|------------|---------------|----------------|
| 3B models  | 2 GPUs        | 80GB           |
| 7B models  | 4 GPUs        | 80GB           |
| 14B models | 8 GPUs        | 80GB           |

## Configuration System

### Key Configuration Options
```yaml
azr:
  executor: qwq                        # or sandboxfusion
  problem_types: [code_i, code_o, code_f]
  reward:
    n_samples: 8                       # Solutions per problem
    generation_reward_config:
      complexity_reward:
        enabled: true
        coef: 0.1
        max: 0.5
  data_selection_strategy:
    valid_program_filter: all          # all, non_one, non_extremes
    composite_function_n_max: 3        # Max composite functions
    
trainer:
  total_epochs: 30
  wandb_run_id: null                   # For resuming runs
```

### Prompt Template
Uses DeepSeek R1 format with structured reasoning:
```
<think> reasoning process here </think>
<answer> answer here </answer>
```

## Important Notes and Warnings

### Security Warning ⚠️
The Python executor is designed for **research purposes only** and is **not secure for production environments**. It executes arbitrary code with only basic safety measures. For production use, consider:
- Using the SandboxFusion executor
- Implementing additional sandboxing
- Running in isolated environments
- Adding network restrictions

### Branch Management
- **`main` branch**: Regularly updated with latest veRL versions (under active development)
- **`paper` branch**: Stable version for reproducing paper results

### Special Considerations
- **Qwen3 Models**: Require removing untrained `<think>` tokens using provided script
- **Resume Training**: Full checkpoint support with W&B integration
- **Data Management**: Sophisticated filtering and selection strategies
- **Composite Functions**: Advanced feature for generating multi-function problems

## Training Tips

1. **Start with Seed Data**: Use provided seed datasets or generate your own
2. **Monitor Executor Load**: Clean up executor periodically (configurable frequency)
3. **Adjust Sampling**: Tune `n_samples` based on GPU memory
4. **Enable Intrinsic Rewards**: Experiment with different reward combinations
5. **Use Validation Filtering**: Choose appropriate `valid_program_filter` strategy

## Extending the System

To add new problem types:
1. Define prompt template in `prompts.py`
2. Add generation logic in `constructor.py`
3. Implement reward calculation in `reward_managers.py`
4. Update supported tasks in `azr_ray_trainer.py`

To add new intrinsic rewards:
1. Implement calculation in `code_reward.py`
2. Add configuration in `azr_ppo_trainer.yaml`
3. Integrate into reward manager

---

