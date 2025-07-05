import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'selfplay' / 'coder1_5b.sh'


def test_script_exists():
    assert SCRIPT_PATH.is_file(), f"{SCRIPT_PATH} should exist"


def test_script_syntax():
    subprocess.check_call(['bash', '-n', str(SCRIPT_PATH)])


def test_model_path():
    content = SCRIPT_PATH.read_text()
    assert 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' in content


def test_single_gpu_configuration():
    content = SCRIPT_PATH.read_text()
    assert 'trainer.n_gpus_per_node=1' in content
    assert 'actor_rollout_ref.rollout.tensor_model_parallel_size=1' in content
