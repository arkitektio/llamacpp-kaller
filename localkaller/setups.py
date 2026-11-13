"""Named setups: a model and how to serve it, in one YAML (or JSON) file.

A setup file describes the setups a node offers and which one is the default::

    default: deepseek          # the setup used when LLAMA_PROFILE names none

    defaults:                  # shared by every setup; a setup's own value wins
      server_bin: ~/sw/llama.cpp/build/bin/llama-server
      threads: 48

    setups:
      deepseek:
        model: /models/DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00001-of-00008.gguf
        alias: deepseek-v4.1-flash
        ctx: 131072
        args: [--temp, "1.0", --reasoning-budget, "2048", --fit-target, "10240"]
      qwen:
        hf: unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL
        ctx: 65536
        args: -ngl 999 --reasoning-budget 2048

Every key is a ``LLAMA_*`` variable without the prefix, in lower case (see
``settings``): ``model``, ``hf``, ``alias``, ``server_bin``, ``host``, ``port``,
``ctx``, ``threads``, ``args`` (a list, or one shell-style string), ``start_timeout``,
``attach_url``, ``log_dir``, ``cache``, ``agent_id``, ``max_tokens``.

The file is ``setups.yaml`` beside ``app.py``, or what ``LLAMA_SETUPS`` names. The setup
is ``LLAMA_PROFILE``, else the file's ``default``. Environment variables, and ``.env``,
win over the file: a setup is a starting point, not a cage.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

import yaml

SETUPS_FILE = Path(__file__).resolve().parent.parent / "setups.yaml"

#: The keys a setup may have, and the variable each stands for.
KEYS = {
    "model": "LLAMA_MODEL",
    "hf": "LLAMA_HF",
    "alias": "LLAMA_ALIAS",
    "server_bin": "LLAMA_SERVER_BIN",
    "host": "LLAMA_HOST",
    "port": "LLAMA_PORT",
    "ctx": "LLAMA_CTX",
    "threads": "LLAMA_THREADS",
    "args": "LLAMA_ARGS",
    "start_timeout": "LLAMA_START_TIMEOUT",
    "attach_url": "LLAMA_ATTACH_URL",
    "log_dir": "LLAMA_LOG_DIR",
    "cache": "LLAMA_CACHE",
    "agent_id": "LLAMA_AGENT_ID",
    "max_tokens": "LLAMA_MAX_TOKENS",
}

#: Keys that say where the model comes from: a setup's own replace the defaults' together.
SOURCES = ("model", "hf", "attach_url")


class SetupError(ValueError):
    """The setup file, or the setup asked for, is not usable."""


def load(path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """The parsed setup file, or ``None`` when there is none."""
    file = Path(path).expanduser() if path else SETUPS_FILE
    if not file.is_file():
        if path:
            raise SetupError(f"LLAMA_SETUPS names {file}, which does not exist")
        return None
    try:
        data = yaml.safe_load(file.read_text()) or {}
    except yaml.YAMLError as e:
        raise SetupError(f"{file} is not valid YAML or JSON: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("setups", {}), dict):
        raise SetupError(f"{file} needs a mapping with a `setups` mapping in it")
    return data


def names(data: dict[str, Any]) -> list[str]:
    return sorted((data.get("setups") or {}).keys())


def variables(data: dict[str, Any], name: str | None = None) -> dict[str, str]:
    """The ``LLAMA_*`` variables of setup ``name`` (default: the file's ``default``)."""
    setups = data.get("setups") or {}
    chosen = name or data.get("default")
    if not chosen:
        if len(setups) == 1:
            chosen = next(iter(setups))
        else:
            raise SetupError(f"Choose a setup with LLAMA_PROFILE, or set `default:` in the file. Setups: {', '.join(names(data))}")
    if chosen not in setups:
        raise SetupError(f"There is no setup {chosen!r}. Setups: {', '.join(names(data)) or 'none'}")
    own = setups[chosen] or {}
    merged = dict(data.get("defaults") or {})
    if any(key in own for key in SOURCES):
        for key in SOURCES:
            merged.pop(key, None)
    merged.update(own)
    unknown = sorted(set(merged) - set(KEYS))
    if unknown:
        raise SetupError(f"Setup {chosen!r} has unknown keys {unknown}; known: {', '.join(KEYS)}")
    out = {"LLAMA_PROFILE": str(chosen)}
    for key, value in merged.items():
        if value is None:
            continue
        if key == "args" and isinstance(value, (list, tuple)):
            value = shlex.join(str(v) for v in value)
        out[KEYS[key]] = os.path.expanduser(str(value)) if key in ("model", "server_bin", "log_dir", "cache") else str(value)
    return out
