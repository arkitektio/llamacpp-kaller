"""Arkitekt's fixtures, and a stub llama-server in place of the real binary.

Every test runs against ``tests/stub_llama_server.py`` on a free port: the lifecycle
hook starts it like the real one, waits for its ``/health``, and stops it.
"""

import os
import socket
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# The app is a module of the project, not of tests/: a test imports it from there.
sys.path.insert(0, str(ROOT))

pytest_plugins = ["arkitekt.testing"]

STUB = ROOT / "tests" / "stub_llama_server.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def stub_settings(monkeypatch, tmp_path):
    """Point the app at the stub server, on a port of its own, logging under tmp."""
    STUB.chmod(STUB.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "Stub-Model-Q4_K_M-00001-of-00002.gguf"
    model.write_bytes(b"GGUF")
    for key in [k for k in os.environ if k.startswith("LLAMA_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLAMA_ENV_FILE", str(tmp_path / "no.env"))
    # The node's own setups.yaml must not leak into the tests.
    monkeypatch.setattr("localkaller.setups.SETUPS_FILE", tmp_path / "no-setups.yaml")
    monkeypatch.setenv("LLAMA_SERVER_BIN", str(STUB))
    monkeypatch.setenv("LLAMA_MODEL", str(model))
    monkeypatch.setenv("LLAMA_PORT", str(free_port()))
    monkeypatch.setenv("LLAMA_START_TIMEOUT", "20")
    monkeypatch.setenv("LLAMA_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("STUB_LOAD_SECONDS", "0.3")
