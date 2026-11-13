"""The wiring: a room message through kaller's turn and the local server, back into the room.

kaller's own behaviour (searching, approvals, installs, memory) is kaller's tests'.
Here the model is the stub server, which answers without calling tools, so a turn
is one round: the room becomes the prompt, the answer is streamed back.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import app as app_module
from localkaller.client import INSTRUCTION_PREFIX, LocalClient, fold_late_system_messages
from localkaller.turn import LOCAL_NOTE, history_of, run_turn

BOT = "local-kaller"


def msg(text, name="alice", structures=(), streaming=False, before=()):
    return SimpleNamespace(
        id="1", text=text, agent=SimpleNamespace(id="9", name=name), is_streaming=streaming,
        attached_structures=tuple(SimpleNamespace(identifier=i, object=o) for i, o in structures),
        before=tuple(before), room=SimpleNamespace(id="5"),
    )


class FakeTask:
    def __init__(self):
        self.progress = []

    async def aprogress(self, percentage, message=None):
        self.progress.append((percentage, message))

    async def arecord(self, fn, key=None):
        value = fn()
        return await value if asyncio.iscoroutine(value) else value


class FakeAlpaka:
    """Memory answers: who assigned the turn, and nothing remembered yet."""

    async def aassigner(self):
        return SimpleNamespace(name="Alice", preferred_username="alice", sub="1")

    async def amemory_index(self, limit):
        return []


class Room:
    """``stream_into_room`` for one room: every opened message, with what was appended."""

    def __init__(self):
        self.messages = []

    def open(self, room):
        @asynccontextmanager
        async def stream():
            writer = SimpleNamespace(id=f"m{len(self.messages) + 1}", room=room, text="", structures=[])

            async def append(delta):
                writer.text += delta

            writer.append = append
            writer.attach = lambda *s: writer.structures.extend(s)
            self.messages.append(writer)
            yield writer

        return stream()


def test_the_room_becomes_kallers_prompt_with_this_apps_name():
    before = [msg("Segmented: 12 nuclei.", name=BOT), msg("segment this", structures=[("@mikro/image", 42)]), msg("typing…", streaming=True)]

    history = history_of(msg("and measure them", before=before), agent_id=BOT)

    assert history[0]["role"] == "system" and LOCAL_NOTE in history[0]["content"]
    assert [m["role"] for m in history[1:]] == ["user", "assistant", "user"]
    assert history[1]["content"] == "segment this\n[Attached: @mikro/image:42]"
    assert history[3]["content"] == "and measure them"


def test_late_system_messages_become_instructions():
    messages = [{"role": "system", "content": "prompt"}, {"role": "user", "content": "hi"}, {"role": "system", "content": "Stop calling tools."}]

    folded = fold_late_system_messages(messages)

    assert folded[0] == messages[0] and folded[1] == messages[1]
    assert folded[2] == {"role": "user", "content": INSTRUCTION_PREFIX + "Stop calling tools."}


def test_a_turn_streams_the_local_answer_into_the_room(local_runtime):
    srv = app_module.server
    room, task = Room(), FakeTask()
    client = LocalClient(srv.settings.api_url, max_tokens=256)

    result = asyncio.run(
        run_turn(
            msg("hello there"),
            client=client,
            model=srv.settings.alias,
            agent_id=BOT,
            alpaka=FakeAlpaka(),
            rekuest=None,
            task=task,
            open_room_stream=room.open,
            engines={},
        )
    )

    assert len(room.messages) == 1 and result.message_id == "m1"
    reply = room.messages[0]
    assert reply.room == "5"
    assert "last: hello there" in reply.text, reply.text
    assert task.progress[0] == (5, "Reading the room...") and task.progress[-1] == (100, "Done.")


def test_the_app_offers_the_assistant(local_runtime):
    assert {"assign", "model_info"} <= set(app_module.app.registry.implementations)
    assert app_module.server is not None and app_module.server.owned


def test_a_port_that_already_answers_is_not_started_on(local_runtime):
    import pytest

    from localkaller.server import LlamaServer, ServerError

    busy = LlamaServer(app_module.server.settings)
    with pytest.raises(ServerError, match="LLAMA_ATTACH_URL"):
        busy.start()
