"""An agent harness over the Agent Client Protocol, through acpx.

`AcpChat` sends each user message as one ``acpx prompt`` and reads the turn
back from its ACP events. ``run(command) -> (exit_code, stdout)`` runs a shell
command where acpx and its configured target harness live: a Harbor task's
agent container, say, whose acpx config names the target. The harness has
its MCP servers already (Harbor configures them from the task), so
`connect` adds nothing.

Each stretch of the harness's text between tool calls is one message, as the
model baseline does. A message sent while a turn runs is sent once it ends.
Every ACP event is kept in ``events`` for the run's evidence.
"""

from __future__ import annotations

import contextvars
import json
import shlex
import threading

from assistanteval.contracts import EpisodeStepError, Message, ResetEvidence

PROMPT = "acpx --format json --json-strict prompt {text}"


def messages_of(ndjson: str) -> tuple[list[str], list[dict]]:
    """The texts one turn wrote, split at its tool calls, and its events."""
    texts, current, events = [], "", []
    for line in ndjson.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        events.append(event)
        update = (event.get("params") or {}).get("update") or {}
        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk" and update["content"].get("type") == "text":
            current += update["content"]["text"]
        elif kind in ("tool_call", "tool_call_update") and current.strip():
            texts.append(current.strip())
            current = ""
    if current.strip():
        texts.append(current.strip())
    return texts, events


class AcpChat:
    def __init__(self, run):
        self.run = run
        self.events: list[dict] = []
        self.error: str | None = None
        self._messages: list[Message] = []
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self._turn: threading.Thread | None = None
        self._busy = False  # cleared, under the lock, only as the turns end
        self.answering = False  # whether a turn ran when read() took its messages

    def reset(self) -> ResetEvidence:
        # The harness starts in a new environment with a new ACP session.
        return ResetEvidence(not (self._messages or self._turn), "new_acp_session")

    def connect(self, connectors) -> list[str]:
        return []

    def send(self, text: str) -> None:
        with self._lock:
            if self.error:
                raise EpisodeStepError(self.error)
            self._pending.append(text)
            if self._busy:
                return  # sent when the running turn ends
            self._busy = True
            # In the sender's context: a host's per-task state, such as the
            # environment Harbor scopes to the agent, reaches each prompt.
            context = contextvars.copy_context()
            self._turn = threading.Thread(
                target=context.run, args=(self._run,), daemon=True
            )
            self._turn.start()

    def read(self) -> tuple[list[Message], list[dict]]:
        with self._lock:
            if self.error:
                raise EpisodeStepError(self.error)
            self.answering = self._busy
            return list(self._messages), []  # acpx approves the harness's requests

    def close(self) -> None:
        with self._lock:
            self._pending.clear()
            turn = self._turn
        if turn:
            turn.join(timeout=60)

    def _run(self) -> None:
        try:
            error = self._turns()
        except Exception as failure:  # noqa: BLE001 - a failed exec or unreadable output
            # Its type only: the output can hold task text.
            print(f"ACP turn failed: {type(failure).__name__}", flush=True)
            error = "acp_prompt_failed"
        if error:
            with self._lock:
                self.error, self._busy = error, False

    def _turns(self) -> str | None:
        """Sends the pending messages until none is left; the error that
        stopped them, if any."""
        while True:
            with self._lock:
                if not self._pending:
                    self._busy = False
                    return None
                text = self._pending.pop(0)
            code, out = self.run(PROMPT.format(text=shlex.quote(text)))
            texts, events = messages_of(out)
            stops = {(e.get("result") or {}).get("stopReason") for e in events}
            with self._lock:
                self.events += events
                self._messages += [
                    Message(str(len(self._messages) + i), t)
                    for i, t in enumerate(texts)
                ]
            if code != 0:
                return "acp_prompt_failed"
            if "refusal" in stops and not texts:
                # Refused without a word: the session failed (a harness
                # that forgot it, say), not the model declining.
                return "acp_prompt_refused"
