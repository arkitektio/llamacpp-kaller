"""Named setups from a YAML or JSON file, and how they meet the environment."""

import json

import pytest

from localkaller import setups
from localkaller.settings import Settings

FILE = """
default: big
defaults:
  server_bin: /bin/ls
  threads: 48
  model: /models/shared.gguf
setups:
  big:
    model: /models/Big-Model-Q4_K_M-00001-of-00003.gguf
    ctx: 131072
    args: [--temp, "1.0", --fit-target, "10240"]
  small:
    hf: org/Small-GGUF:Q8_0
    args: -ngl 999
"""


@pytest.fixture
def setup_file(tmp_path, monkeypatch):
    path = tmp_path / "setups.yaml"
    path.write_text(FILE)
    monkeypatch.setattr(setups, "SETUPS_FILE", path)
    return path


def test_the_default_setup_is_used(setup_file):
    settings = Settings.from_env({})

    assert settings.model == "/models/Big-Model-Q4_K_M-00001-of-00003.gguf"
    assert settings.alias == "big-model-q4_k_m" and settings.ctx == 131072 and settings.threads == 48
    assert settings.extra_args == ("--temp", "1.0", "--fit-target", "10240")


def test_a_named_setup_replaces_the_defaults_model_source(setup_file):
    settings = Settings.from_env({"LLAMA_PROFILE": "small"})

    assert settings.model is None and settings.hf == "org/Small-GGUF:Q8_0"
    assert settings.extra_args == ("-ngl", "999") and settings.ctx == 32768


def test_the_environment_wins_over_the_setup(setup_file):
    settings = Settings.from_env({"LLAMA_CTX": "4096", "LLAMA_ARGS": "--parallel 1"})

    assert settings.ctx == 4096 and settings.extra_args == ("--parallel", "1")
    assert settings.model.endswith("Big-Model-Q4_K_M-00001-of-00003.gguf")


def test_json_is_accepted(tmp_path):
    path = tmp_path / "setups.json"
    path.write_text(json.dumps({"setups": {"only": {"hf": "a/B-GGUF", "port": 9000}}}))

    settings = Settings.from_env({"LLAMA_SETUPS": str(path), "LLAMA_SERVER_BIN": "/bin/ls"})

    assert settings.hf == "a/B-GGUF" and settings.port == 9000


def test_mistakes_are_named(setup_file, tmp_path):
    with pytest.raises(setups.SetupError, match="no setup 'huge'.*big, small"):
        Settings.from_env({"LLAMA_PROFILE": "huge"})
    bad = tmp_path / "bad.yaml"
    bad.write_text("setups:\n  x:\n    modle: /m.gguf\n")
    with pytest.raises(setups.SetupError, match="unknown keys \\['modle'\\]"):
        Settings.from_env({"LLAMA_SETUPS": str(bad)})
    with pytest.raises(setups.SetupError, match="does not exist"):
        Settings.from_env({"LLAMA_SETUPS": str(tmp_path / "missing.yaml")})
