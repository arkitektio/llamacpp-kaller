"""Starting, watching and stopping ``llama-server``.

A :class:`LlamaServer` owns the process for the length of the app's run. It starts
it, waits until ``/health`` says the model is loaded, answers questions about what
is being served, and stops it on the way out. With ``LLAMA_ATTACH_URL`` it owns
nothing and only talks to a server someone else started.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from localkaller.settings import Settings

log = logging.getLogger(__name__)


class ServerError(RuntimeError):
    """The server did not come up, or went away."""


def _answers(url: str) -> bool:
    """Whether anything answers ``url`` right now."""
    try:
        return httpx.get(url, timeout=2.0).status_code < 500
    except httpx.HTTPError:
        return False


@dataclass(frozen=True)
class ModelInfo:
    """What the running server says it serves."""

    alias: str
    model_path: str | None
    context_length: int | None
    slots: int | None

    def describe(self) -> str:
        parts = [f"model: {self.alias}"]
        if self.model_path:
            parts.append(f"file: {Path(self.model_path).name}")
        if self.context_length:
            parts.append(f"context: {self.context_length}")
        if self.slots:
            parts.append(f"parallel slots: {self.slots}")
        return " · ".join(parts)


class LlamaServer:
    """One ``llama-server``: ours when started here, borrowed when attached."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path: Path | None = None

    @property
    def url(self) -> str:
        return self.settings.local_url

    @property
    def owned(self) -> bool:
        return self.process is not None

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        """Launch the process, unless attaching. Returns at once; see :meth:`wait_ready`."""
        if self.settings.attach_url:
            log.info("Attaching to llama-server at %s", self.settings.attach_url)
            return
        command = self.settings.command()
        if not os.path.exists(command[0]):
            raise ServerError(
                f"llama-server not found at {command[0]}: build llama.cpp, put llama-server on PATH, "
                "or set LLAMA_SERVER_BIN (the app's container image brings its own)"
            )
        if "-m" in command and not os.path.exists(command[1 + command.index("-m")]):
            raise ServerError(f"model file not found: {self.settings.model}")
        if _answers(f"{self.url}/health"):
            raise ServerError(
                f"Something already serves {self.url} (another llama-server, perhaps the "
                f"arkitekt-llamacpp app). Stop it, or set LLAMA_ATTACH_URL={self.url} to use it."
            )
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.settings.log_dir / f"llama-server-{self.settings.alias}.log"
        log.info("Starting %s (log: %s)", " ".join(command), self.log_path)
        with open(self.log_path, "ab") as sink:
            self.process = subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)

    async def wait_ready(self, timeout: float | None = None) -> None:
        """Poll ``/health`` until the model is loaded, the process dies, or time runs out."""
        deadline = time.monotonic() + (self.settings.start_timeout if timeout is None else timeout)
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if self.process is not None and self.process.poll() is not None:
                    raise ServerError(f"llama-server exited with code {self.process.returncode}; see {self.log_path}")
                try:
                    response = await client.get(f"{self.url}/health")
                    if response.status_code == 200:
                        log.info("llama-server at %s is ready", self.url)
                        return
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise ServerError(f"llama-server at {self.url} not healthy after {self.settings.start_timeout:.0f}s")
                await asyncio.sleep(2.0)

    def stop(self, grace: float = 20.0) -> None:
        """Stop our process, gently then not. A borrowed server is left alone."""
        if self.process is None:
            return
        if self.process.poll() is None:
            log.info("Stopping llama-server (pid %s)", self.process.pid)
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        self.process = None

    # -- questions ----------------------------------------------------------------------

    async def healthy(self) -> bool:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                return (await client.get(f"{self.url}/health")).status_code == 200
            except httpx.HTTPError:
                return False

    async def info(self) -> ModelInfo:
        """What is served, from ``/v1/models`` and llama-server's ``/props``."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            models = (await client.get(f"{self.url}/v1/models")).json()
            props: dict[str, Any] = {}
            try:
                response = await client.get(f"{self.url}/props")
                if response.status_code == 200:
                    props = response.json()
            except httpx.HTTPError:
                pass
        data = models.get("data") or []
        alias = data[0]["id"] if data else self.settings.alias
        generation = props.get("default_generation_settings") or {}
        return ModelInfo(
            alias=alias,
            model_path=props.get("model_path"),
            context_length=generation.get("n_ctx"),
            slots=props.get("total_slots"),
        )
