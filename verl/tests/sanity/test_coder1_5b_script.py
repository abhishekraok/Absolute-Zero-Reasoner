# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import subprocess

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "selfplay" / "coder1_5b.sh"


def test_script_exists():
    assert SCRIPT_PATH.is_file()


def test_script_syntax():
    result = subprocess.run(["bash", "-n", str(SCRIPT_PATH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


SCRIPT_CONTENT = SCRIPT_PATH.read_text()


def test_model_path():
    assert "actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-1.5B" in SCRIPT_CONTENT


def test_gpu_config():
    assert "trainer.n_gpus_per_node=1" in SCRIPT_CONTENT
    assert "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1" in SCRIPT_CONTENT
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in SCRIPT_CONTENT
