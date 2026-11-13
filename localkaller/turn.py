"""One turn of kaller for one room message, with the local model.

This is kaller's own ``assign`` minus the ``model: LLMModel`` argument: the model is
the one this app serves, so the client is the local one and the model name its
alias. kaller's ``history_of`` recognises the assistant's earlier messages by
kaller's agent name; this app signs with its own, so the history is built here, with
kaller's prompt and kaller's way of describing a message.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kaller.assistant import (
    SYSTEM_PROMPT,
    _recorded,
    attach_all,
    context_of,
    converse,
    describe,
    memory_preamble,
)

LOCAL_NOTE = (
    "You run on the lab's own hardware as a local open-weights model. Tools are your "
    "only way to act: call them with exactly the arguments their schemas name, as valid JSON."
)


def history_of(message: Any, *, agent_id: str) -> list[dict[str, str]]:
    """kaller's prompt, then the room's conversation oldest first, then ``message``.

    alpaka hands ``before`` newest first. What this app said earlier (its agent's name
    is ``agent_id``) is the assistant's turn; everyone else is the user.
    """
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{LOCAL_NOTE}"}]
    for previous in reversed(message.before or ()):
        if getattr(previous, "is_streaming", False):
            continue
        speaker = getattr(previous.agent, "name", None)
        messages.append({"role": "assistant" if speaker == agent_id else "user", "content": describe(previous)})
    messages.append({"role": "user", "content": describe(message)})
    return messages


@dataclass(frozen=True)
class TurnResult:
    """The id of the room message that answers, and what the turn attached to it."""

    message_id: str
    attachments: tuple[tuple[str, int], ...]


async def run_turn(
    message: Any,
    *,
    client: Any,
    model: str,
    agent_id: str,
    alpaka: Any,
    rekuest: Any,
    task: Any,
    open_room_stream: Callable[[str], Any],
    kabinet: Any = None,
    engines: dict[str, Any] | None = None,
) -> TurnResult:
    """Read the room, let kaller think and act with the local model, and make sure the room hears back."""
    await task.aprogress(5, "Reading the room...")
    messages = history_of(message, agent_id=agent_id)

    async def recall_the_user() -> str:
        return await memory_preamble(alpaka)

    messages[0]["content"] += "\n\n" + await _recorded(task, recall_the_user, key="memory:index")
    turn = await converse(
        rekuest,
        client,
        open_room_stream,
        room=message.room.id,
        model=model,
        messages=messages,
        context=context_of(message),
        task=task,
        report=task.aprogress,
        kabinet=kabinet,
        alpaka=alpaka,
        engines=engines,
    )
    await task.aprogress(100, "Done.")

    message_id = turn.message_id
    if message_id is None:
        # Nothing was streamed (the answer was empty, or only reasoning): say so, so the
        # room is not left silent after a turn that may have run something.
        async with open_room_stream(message.room.id) as reply:
            await reply.append(turn.answer)
            attach_all(reply, turn.attachments)
            message_id = reply.id
    return TurnResult(message_id=str(message_id), attachments=tuple(turn.attachments))
