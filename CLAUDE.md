# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda env create -f azr_env.yml
conda activate azr
pip install -r flashattn_requirements.txt
```

### Training Commands
```bash
# Data processing for CruxEval/LiveCodeBench evaluation during self-play
python -m absolute_zero_reasoner.data_construction.process_code_reasoning_data

# Seeding (optional) - create seed datasets
export OUTPUT_SEED_PATH=data/<new_ded_abd_seed_data_name>.jsonl
export OUTPUT_CODE_F_SEED_PATH=data/<new_ind_seed_data_name>.jsonl
bash scripts/seeding/<7b|14b|coder3b|coder7b|coder14b|llama>.sh

# Self-play training
bash scripts/selfplay/<7b|14b|coder3b|coder7b|coder14b|llama>.sh

# Convert veRL checkpoints to HuggingFace format
python -m absolute_zero_reasoner.utils.convert2hf <veRL_ckpt_path>/actor <veRL_ckpt_path>/actor/huggingface/ <hf_ckpt_path>
```

### Testing Commands
```bash
# Run basic tests
python -m pytest tests/

# Math evaluation
cd evaluation/math_eval
bash run.sh

# Code evaluation - LiveCodeBench
# First download the data
git clone https://hf-mirror.com/datasets/livecodebench/code_generation_lite evaluation/code_eval/coding/LiveCodeBench/code_generation_lite
bash evaluation/code_eval/scripts/run_lcb_gen.sh --model <model_path>

# Code evaluation - EvalPlus (requires separate conda env)
conda create -n evalplus python=3.11
conda activate evalplus
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus@d362e933265c3e7e3df8101c930a89c3c470cd9f"
bash evaluation/code_eval/scripts/run_evalplus.sh 0 <humaneval|mbpp> <model_path>
```

## Architecture Overview

### Core Algorithm
Absolute Zero Reasoner implements a self-play reinforcement learning approach with two main phases:

1. **PROPOSE**: The model generates reasoning tasks from three types:
   - Abduction: Reasoning backwards from observations
   - Deduction: Reasoning forwards from premises  
   - Induction: Reasoning from specific cases to general rules
   
2. **SOLVE**: The model attempts to solve self-generated tasks, with solutions verified through Python execution

The system uses **TRR++** (Test-Time Reward Reasoning) for continuous improvement without external training data.

### Key Components

#### Main Training Pipeline
- `absolute_zero_reasoner/main_azr_ppo.py`: Main training entry point using Ray and Hydra
- `absolute_zero_reasoner/trainer/ppo/azr_ray_trainer.py`: Core AZR Ray-based PPO trainer
- `absolute_zero_reasoner/configs/azr_ppo_trainer.yaml`: Comprehensive training configuration

#### Reward System
- `absolute_zero_reasoner/rewards/reward_managers.py`: Manages reward calculation and distribution
- `absolute_zero_reasoner/rewards/code_reward.py`: Code execution and verification rewards
- `absolute_zero_reasoner/rewards/custom_evaluate.py`: Custom evaluation metrics

#### Code Execution
- `absolute_zero_reasoner/utils/code_utils/python_executor.py`: Python code execution (research-only, not production-secure)
- `absolute_zero_reasoner/utils/code_utils/sandboxfusion_executor.py`: Sandbox-Fusion executor integration
- `absolute_zero_reasoner/utils/code_utils/parsers.py`: Code parsing and validation utilities

#### Data Construction
- `absolute_zero_reasoner/data_construction/constructor.py`: Main data construction logic
- `absolute_zero_reasoner/data_construction/prompts.py`: Prompting templates for different reasoning types
- `absolute_zero_reasoner/data_construction/process_code_reasoning_data.py`: Code reasoning data processing

### Configuration System
Uses Hydra configuration management with extensive YAML configs supporting:
- Multi-GPU distributed training with FSDP/FSDP2
- veRL framework integration for RL training
- vLLM integration for efficient inference
- Configurable reward systems with intrinsic rewards (complexity, diversity, etc.)
- Multiple executor backends (qwq, sandboxfusion)

### GPU Requirements
- 3B models: 2 × 80GB GPUs
- 7B/8B models: 4 × 80GB GPUs  
- 14B models: 8 × 80GB GPUs

### Prompt Template
Uses DeepSeek R1 format with `<think>` and `<answer>` tags:
```
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}\nAssistant: <think>
```

## Important Notes

- **Security Warning**: The Python executor is for research purposes only and is not secure for production environments
- **Qwen3 Models**: New Qwen3 base models have untrained `<think>` token embeddings. Use `python absolute_zero_reasoner/utils/remove_think_qwen3_tokenizer.py --model_name <Qwen3ModelName>` to remove these tokens
- **Branch Usage**: Use `paper` branch to replicate original paper results; `main` branch uses updated veRL versions and is regularly updated
- **Testing Status**: The `main` branch is currently under testing - use `paper` branch for stable reproduction of paper results
- **Resuming Training**: Add original wandb run ID to script: `trainer.wandb_run_id=<run_id>`
- **Sandbox-Fusion**: For using sandbox-fusion executor, use Docker and set `azr.executor=sandboxfusion`
- **Custom Intrinsic Rewards**: Add custom rewards to `azr.reward.generation_reward_config` - check existing implementations like diversity and complexity rewards
- **Math Evaluation**: Refer to `evaluation/math_eval/README.md` for detailed math evaluation instructions