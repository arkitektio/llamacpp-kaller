"""The configuration, as a fresh node would read it."""

import pytest

from localkaller import settings as S
from localkaller.settings import Settings, alias_of, alias_of_hf


def test_a_file_model_and_its_alias():
    settings = Settings.from_env({"LLAMA_MODEL": "/m/DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00001-of-00008.gguf", "LLAMA_SERVER_BIN": "/bin/ls"})

    assert settings.alias == alias_of(settings.model) == "deepseek-v4.1-flash-1017-ud-iq3_s"
    assert settings.command()[1:3] == ["-m", settings.model]


def test_a_hugging_face_model_needs_no_file():
    settings = Settings.from_env({"LLAMA_HF": "unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL", "LLAMA_SERVER_BIN": "/bin/ls"})

    assert settings.model is None and settings.alias == "qwen3.8-flash-next"
    assert settings.command()[1:3] == ["-hf", "unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL"]
    assert alias_of_hf("org/Model-GGUF") == "model"


def test_without_a_local_build_llama_server_is_taken_from_path(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "DEFAULT_BIN", str(tmp_path / "missing"))
    monkeypatch.setattr(S.shutil, "which", lambda name: "/usr/local/bin/llama-server")

    assert Settings.from_env({"LLAMA_HF": "a/B-GGUF"}).server_bin == "/usr/local/bin/llama-server"


def test_nothing_to_serve_is_refused():
    with pytest.raises(ValueError, match="LLAMA_HF"):
        Settings.from_env({"LLAMA_ENV_FILE": "/nonexistent"})
