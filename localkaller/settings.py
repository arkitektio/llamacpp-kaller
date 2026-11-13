"""Which local model drives the assistant, and how it is served, from ``LLAMA_*`` variables.

The model is a parameter, not an app: the same app serves DeepSeek today and Qwen
tomorrow by being started with another ``LLAMA_MODEL``.

- ``LLAMA_MODEL``: the GGUF to serve (the first shard of a split model).
- ``LLAMA_HF``: instead of a file, a Hugging Face repository and quant
  (``unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL``) that llama-server downloads on first
  start into ``LLAMA_CACHE``. Raise ``LLAMA_START_TIMEOUT`` for a large first download.
  One of the two is required unless ``LLAMA_ATTACH_URL`` is set.
- ``LLAMA_ALIAS``: the model's name in the API, default derived from the file name.
- ``LLAMA_SERVER_BIN``: the ``llama-server`` binary, default ``~/sw/llama.cpp/build/bin/llama-server``
  when that exists, else ``llama-server`` on ``PATH``.
- ``LLAMA_HOST`` / ``LLAMA_PORT``: where it listens, default ``127.0.0.1:8080``.
- ``LLAMA_CTX``: context length, default 32768. ``LLAMA_THREADS``: CPU threads, default 32.
- ``LLAMA_ARGS``: extra ``llama-server`` flags, one shell-style string. For a model
  larger than the GPU do not pass ``-ngl`` or ``-ot``: that disables llama.cpp's
  automatic fit and it then tries to put everything on the GPU.
- ``LLAMA_START_TIMEOUT``: seconds to wait for the server to be healthy, default 1800
  (a 400 GB model paged in from disk takes minutes).
- ``LLAMA_PROFILE``: the named setup of ``setups.yaml`` to use (see ``setups``); its
  values fill in whatever the environment and ``.env`` do not set.
- ``LLAMA_SETUPS``: another setup file than ``setups.yaml`` beside ``app.py``.
- ``LLAMA_CACHE``: where ``LLAMA_HF`` downloads go (llama-server's own variable).
- ``LLAMA_ATTACH_URL``: use a server already running at this URL instead of starting one.
- ``LLAMA_AGENT_ID``: the name the assistant signs its room messages with, default
  ``local-kaller``. Its earlier replies are recognised by it, so keep it stable.
- ``LLAMA_MAX_TOKENS``: the longest completion of one round, default 8192 (reasoning counts).
"""

from __future__ import annotations

import os
import shlex
import shutil

from localkaller import setups
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BIN = "~/sw/llama.cpp/build/bin/llama-server"


def default_server_bin() -> str:
    """The local build when there is one, else whatever ``llama-server`` is on ``PATH``."""
    built = os.path.expanduser(DEFAULT_BIN)
    if os.path.exists(built):
        return built
    return shutil.which("llama-server") or built
#: The app's own configuration file: `KEY=VALUE` lines beside app.py, read when the
#: environment names no model. Node-specific, so not in git; see .env.example.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """``KEY=VALUE`` lines of ``path`` (default :data:`ENV_FILE`); comments and blanks skipped.

    Values may be quoted with single or double quotes. A missing file is empty.
    """
    file = Path(path).expanduser() if path else ENV_FILE
    if not file.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def alias_of(model_path: str) -> str:
    """``DeepSeek-V4.1-Flash-1017-UD-IQ3_S-00001-of-00008.gguf`` → ``deepseek-v4.1-flash-1017-ud-iq3_s``."""
    name = Path(model_path).name
    if name.endswith(".gguf"):
        name = name[: -len(".gguf")]
    parts = name.split("-")
    if len(parts) >= 3 and parts[-2] == "of" and parts[-3].isdigit() and parts[-1].isdigit():
        parts = parts[:-3]
    return "-".join(parts).lower() or "model"


def alias_of_hf(repo: str) -> str:
    """``unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL`` → ``qwen3.8-flash-next``."""
    name = repo.split(":", 1)[0].rsplit("/", 1)[-1]
    if name.lower().endswith("-gguf"):
        name = name[: -len("-gguf")]
    return name.lower() or "model"


@dataclass(frozen=True)
class Settings:
    """One run's configuration."""

    model: str | None
    hf: str | None
    alias: str
    server_bin: str
    host: str
    port: int
    ctx: int
    threads: int
    extra_args: tuple[str, ...]
    start_timeout: float
    attach_url: str | None
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    agent_id: str = "local-kaller"
    max_tokens: int = 8192

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        """The settings of the process environment, completed from the app's ``.env`` file.

        ``arkitekt run`` is started bare more often than not, so when neither
        ``LLAMA_MODEL`` nor ``LLAMA_ATTACH_URL`` is set, ``.env`` beside ``app.py`` (or
        the file ``LLAMA_ENV_FILE`` names) fills in whatever the environment lacks.
        """
        env: dict[str, str] = dict(os.environ if environ is None else environ)
        if not env.get("LLAMA_MODEL") and not env.get("LLAMA_HF") and not env.get("LLAMA_ATTACH_URL"):
            for key, value in load_env_file(env.get("LLAMA_ENV_FILE")).items():
                env.setdefault(key, value)
        # A named setup from setups.yaml fills in what the environment and .env left open.
        data = setups.load(env.get("LLAMA_SETUPS"))
        if data is not None:
            for key, value in setups.variables(data, env.get("LLAMA_PROFILE")).items():
                env.setdefault(key, value)
        if env.get("LLAMA_CACHE"):
            # llama-server reads it for -hf downloads; it inherits this process's environment.
            os.environ.setdefault("LLAMA_CACHE", os.path.expanduser(env["LLAMA_CACHE"]))
        model = env.get("LLAMA_MODEL") or None
        hf = env.get("LLAMA_HF") or None
        attach = env.get("LLAMA_ATTACH_URL") or None
        if model is None and hf is None and attach is None:
            raise ValueError(
                "No model to serve. Set LLAMA_MODEL to a GGUF, LLAMA_HF to a Hugging Face repo:quant (or LLAMA_ATTACH_URL to a running "
                f"llama-server), put it in {ENV_FILE}, or start the app with ./run.sh deepseek|qwen"
            )
        alias = env.get("LLAMA_ALIAS") or (alias_of(model) if model else alias_of_hf(hf) if hf else "model")
        return cls(
            model=os.path.expanduser(model) if model else None,
            hf=hf,
            alias=alias,
            server_bin=os.path.expanduser(env.get("LLAMA_SERVER_BIN") or default_server_bin()),
            host=env.get("LLAMA_HOST") or "127.0.0.1",
            port=int(env.get("LLAMA_PORT") or 8080),
            ctx=int(env.get("LLAMA_CTX") or 32768),
            threads=int(env.get("LLAMA_THREADS") or 32),
            extra_args=tuple(shlex.split(env.get("LLAMA_ARGS") or "")),
            start_timeout=float(env.get("LLAMA_START_TIMEOUT") or 1800),
            attach_url=attach.rstrip("/") if attach else None,
            log_dir=Path(env.get("LLAMA_LOG_DIR") or "logs"),
            agent_id=env.get("LLAMA_AGENT_ID") or "local-kaller",
            max_tokens=int(env.get("LLAMA_MAX_TOKENS") or 8192),
        )

    @property
    def local_url(self) -> str:
        """Where this app reaches the server."""
        return self.attach_url or f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        """The OpenAI-compatible base, for an SDK client."""
        return f"{self.local_url}/v1"

    def command(self) -> list[str]:
        """The ``llama-server`` command line for these settings."""
        if self.model is not None:
            source = ["-m", self.model]
        elif self.hf is not None:
            source = ["-hf", self.hf]
        else:
            raise ValueError("No model to serve: this run attaches to a running server")
        return [
            self.server_bin,
            *source,
            "--alias", self.alias,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.ctx),
            "-t", str(self.threads),
            "-fa", "on",
            "--jinja",
            *self.extra_args,
        ]
