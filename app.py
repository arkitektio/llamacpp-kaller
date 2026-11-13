"""kaller with a local model: a room assistant that runs actions, thinking on this node.

The assistant is kaller (``github.com/jhnnsrs/kaller``), used as a library: asked
"segment this", it finds an installed segmenter, calls it as a child of the turn
that the user approves in the room (unless the turn runs in auto-mode), reads
mikro, rekuest, alpaka and kabinet to understand the data, installs what is
missing, remembers the user and asks them questions.

What differs from kaller is only the model. kaller is handed an alpaka model and
reaches it through alpaka's tunnel; this app owns a ``llama-server`` for the GGUF it
is configured with (``LLAMA_*`` variables, or ``.env`` beside this file) and hands
kaller a client pointed at it. So the model needs no provider registration and the
node needs to be reachable by nobody: the app answers rooms over the connection it
already has.

Run:  arkitekt run dev
"""

from __future__ import annotations

import logging
from typing import Any

from alpaka import Alpaka, alpaka_service, stream_into_room
from alpaka.api.schema import Message
from arkitekt import App, Task, run
from kabinet import Kabinet, kabinet_service
from kaller.schema_engine import engines_for
from mikro import Mikro, mikro_service
from rekuest.arkitekt import rekuest_service
from rekuest.client.client import Rekuest

from localkaller.client import LocalClient
from localkaller.server import LlamaServer, ServerError
from localkaller.settings import Settings
from localkaller.turn import run_turn

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "0.1.0"

log = logging.getLogger("localkaller.app")

app = App(
    "llamacpp-kaller",
    __version__,
    author="wilhelmi",
    description="A room assistant that finds and runs the actions a room asks for, driven by a local llama.cpp model",
    services=[alpaka_service, rekuest_service, kabinet_service, mikro_service],
)

# The one server of this run; set by the lifecycle hook, read by the turns.
server: LlamaServer | None = None


@app.lifecycle
async def serve():
    """Start llama-server (or attach to one), wait until the model is loaded, stop it at the end."""
    global server
    server = LlamaServer(Settings.from_env())
    server.start()
    try:
        await server.wait_ready()
        log.info("Serving %s", (await server.info()).describe())
        yield
    finally:
        server.stop()
        server = None


def _server() -> LlamaServer:
    if server is None:
        raise ServerError("The llama-server is not running")
    return server


# It asks the questions the model writes and calls the actions it finds: both made at run
# time, so it declares both.
@app.workflow(dynamic_asks=True, dynamic_calls=True)
async def assign(
    message: Message,
    alpaka: Alpaka,
    rekuest: Rekuest,
    kabinet: Kabinet,
    mikro: Mikro,
    task: Task,
) -> Message:
    """Run the action a room asks for, installing it first if need be, and answer with what it returned.

    The model is the local one this app serves. Each call and install waits for the
    user's approval in the room, unless the turn runs in auto-mode. Structures an
    action returns are attached to the answer.
    """
    srv = _server()
    settings = srv.settings

    def room_stream(room: str) -> Any:
        return stream_into_room(alpaka, room=room, agent_id=settings.agent_id)

    result = await run_turn(
        message,
        client=LocalClient(settings.api_url, max_tokens=settings.max_tokens),
        model=settings.alias,
        agent_id=settings.agent_id,
        alpaka=alpaka,
        rekuest=rekuest,
        kabinet=kabinet,
        task=task,
        open_room_stream=room_stream,
        engines=engines_for({"mikro": mikro, "rekuest": rekuest, "alpaka": alpaka, "kabinet": kabinet}),
    )
    return await alpaka.aget_message(id=result.message_id)


@app.action
async def model_info() -> str:
    """Model Info

    Which local model drives the assistant: name, file, context length and parallel slots.

    Returns
    -------
    str
        A one-line description
    """
    srv = _server()
    if not await srv.healthy():
        return f"llama-server at {srv.url} is not healthy"
    return (await srv.info()).describe()


if __name__ == "__main__":
    run(app)
