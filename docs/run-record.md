# Run record

A run record is everything saved from one run: one assistant, one task, one conversation.
The runner writes it, the grader reads it, and nothing else is needed to grade the run.

```
runs/<run_id>/
  run.json      the task as the run used it, what ran, every turn, each service's
                requests and start and end state, the simulated user's decisions,
                and how the run ended
  evidence/     the assistant adapter's own evidence (traces, events); not graded
```

Build a `assistanteval.run_record.RunRecord` and write it with `run_record.save(record, folder)`, which writes `run.json` whole or not at all (a temporary file, then a rename). Read it with `run_record.load(folder)`. Never edit a saved record: grades go to `grades/<run_id>/`.

The record holds the task rendered at the run's start (see [tasks/README.md](../tasks/README.md#dates)), so the grader never opens `tasks/`, and editing a task never changes how its old runs grade.

## Rules for a runner

`assistanteval/run_task.py` follows these; a runner of your own must too, or its records grade as invalid.

- **Times** are Unix milliseconds on one clock that the runner and the services share.
  - A user turn is stamped just before it is sent.
  - An assistant message is stamped when it is first read. If the assistant later rewrites it under the same message id, `text` becomes the final text and `edited_at_ms` records when; the grader treats it as shown from `at_ms` and written at `edited_at_ms`.
  - A request is stamped when the service handles it.
- **Services:** connect exactly the task's `environment.connectors`, each from its fixture rendered at the run's anchor. The assistant starts with no earlier chats or memory.
- **Log every request,** including refused and failed ones, and **save each service's full state** at the start (after its fixture loads, before the opening message) and at the end (after the last reply, before cleanup). A change the assistant made is a result, not a cleanup error.
- **What the assistant sees:** only the simulated user's messages, from `opening()`, `reply()` and `nudge()`; never another task field, in a system prompt or anywhere else. The grader checks every user turn against these.
- **Stopping and nudging:** stop at 30 user turns. If the assistant sends nothing for 3 minutes, send a nudge; a nudge counts as a user turn and as user effort. After 3 nudges in a row with no reply, or 45 minutes, end the run with `timeout`.
- **A run is valid only if** the reset is verified, cleanup succeeded, no service failed (a refusal is the assistant's to handle; a service error is the runner's), the conversation started within an hour of the world's anchor on the same local day, the simulated user followed its rules and a judge agrees with how it read each turn, and every user turn came from the task or the simulated user.

## run.json

```json
{
  "run_id": "model_baseline-kingsway-reschedule-crew-20260924T174702Z",
  "task_id": "kingsway-reschedule-crew",
  "product": "model_baseline",
  "manifest": {"...": "..."},
  "task": {"id": "kingsway-reschedule-crew", "...": "..."},
  "turns": [],
  "permission_clicks": [],
  "setup": [],
  "services": [],
  "simulated_user": {"...": "..."},
  "result": {"...": "..."}
}
```

`product` is the assistant's name as given to `--assistant`. `task` is the task file's content with its dates rendered at the run's anchor.

### manifest

What ran, for provenance. The grader reads only `world`.

```json
{
  "task_sha256": "<sha256 of the task file's bytes>",
  "runner_version": "assistanteval.run_task at <git commit>",
  "endpoints": {"judge": {"model": "gpt-5.6-sol", "api": "openai", "base_url": null}},
  "simulated_user": {"model": "z-ai/glm-5.3-flash"},
  "browser_timezone": "America/Vancouver",
  "world": {
    "anchor": "2026-09-24T10:47:00-07:00",
    "fixtures": {"calendar": "<sha256 of the calendar fixture rendered at the anchor>"},
    "fixture_data": {"calendar": {"events": ["..."]}}
  },
  "limits": {"poll_seconds": 1, "settle_seconds": 5, "silence_seconds": 180, "max_nudges": 3, "max_user_turns": 30, "deadline_seconds": 2700}
}
```

- `endpoints` names each model role that was set (never its key); see `assistanteval/endpoints.py`. An assistant factory can add its own entry, as the model baseline adds `model_baseline: {"model": ...}`.
- `browser_timezone` is the task's time zone, which the runner sets as the process's local time zone.
- `world` is the world the run saw: the *anchor* (the minute the run started, in the task's time zone), and each fixture rendered there, with its SHA-256. The grader checks that each service started from its rendered fixture, and that the conversation started while that world was current.

### turns, permission_clicks, setup

```json
{
  "turns": [
    {"role": "user", "text": "doug confirmed thursday. move kingsway over and make sure luis is still on it.", "at_ms": 1790272035208, "nudge": false},
    {"role": "assistant", "text": "Moved Kingsway to Thursday, October 1, 8 to 12, with Luis.", "at_ms": 1790272188200, "nudge": false}
  ],
  "permission_clicks": [],
  "setup": [{"role": "user", "text": "connector setup: calendar", "at_ms": 1790272001000}]
}
```

- **`turns`:** `role` is `user` or `assistant`. Each assistant message is its own turn, so several messages in a row are several turns; an in-progress message such as "one moment" is a message too.
- **`permission_clicks`:** for an assistant that asks permission before using a tool, each prompt its adapter answered for the user, as `{"at_ms", "label", "detail"}`: the button or outcome, and what the prompt asked. The built-in assistants never ask, so theirs is empty.
- **`setup`:** connecting the services, kept apart from the conversation. The runner adds one entry per connected service, and then whatever the assistant posted while it connected them (what its adapter's `connect()` returns). Setup entries may hold only a connector's name, host and API description. The grader checks that the conversation itself starts with the task.

### services

```json
{
  "services": [
    {
      "name": "calendar",
      "start_state": {"events": ["..."]},
      "end_state": {"events": ["..."]},
      "requests": [
        {
          "service": "calendar",
          "operation": "calendar.update",
          "at_ms": 1790272110223,
          "input": {"event_id": "kingsway-2026-09-28", "start": "2026-10-01T08:00:00-07:00"},
          "output": {"id": "kingsway-2026-09-28", "start": "2026-10-01T08:00:00-07:00"},
          "status": "ok",
          "error": null,
          "kind": "private_change",
          "plain": "change or move a calendar event"
        }
      ]
    }
  ]
}
```

- **Every connected service is listed,** even one with no requests. `start_state` and `end_state` are its session's full state.
- **Each request is the service session's log entry:**
  - `status` is `ok`, `refused` when the service rejected it (bad input, an unknown id, a paid change), or `error` when the service failed. For `refused` and `error`, `output` is null, `error` holds the message the assistant received, and the state is unchanged. Requests rejected for their shape (a wrong type, an unknown field) are logged too, over REST and MCP alike.
  - `kind` is what the operation's declaration in `assistanteval/services/` says it does: `read`, `private_new`, `private_change`, `visible_to_others` or `money`. The grader expects the user's yes before a change to an existing record, anything visible to others and anything that spends money, and expects visible and paid actions to be reported.
  - `plain` is the declaration's plain words for the change, empty for a read.

### simulated_user

`SimulatedUser.record()`:

```json
{
  "version": 2,
  "style": null,
  "model": "<writer model>",
  "reader_model": "<reader model>",
  "rules_model": "<judge model>",
  "task_id": "rain-cover-question",
  "decisions": [
    {
      "after_turn": 1,
      "saw": "asks_for:rain_cover",
      "action": "reveal",
      "fact_ids": ["rain_cover"],
      "operation": null,
      "at_ms": 1790272201000,
      "leak_found": false,
      "rewrites": 0,
      "text": "no rain cover, lost it when we moved",
      "answers": [
        {"judge": "llm", "model": "<reader model>", "question": "user_hears_ask:rain_cover", "answer": 0.97}
      ]
    }
  ]
}
```

- **`saw`** is how the simulated user read the assistant's last turn: `asks_for:<fact_id>`, `asks_approval:<operation>`, `finished`, `asks_other`, `no_progress`, or `unsure:<reading>` when neither the reader nor the judge was sure (the user then says it doesn't know rather than end, approve or reveal on a guess).
- **`action`** is `reveal`, `approve`, `refuse`, `dont_know`, `decline` ("no thanks, that's all for now", to an offer of more help), `wait`, `nudge` or `end`. Only `wait` and `end` send nothing.
- **`version`** is the simulated user's behaviour version (absent before 2), and **`style`** the wording style (`terse`, `distracted` or null).
- **`answers`** are the judge answers behind the decision, in order, so a decision can be audited or replayed.
- **`at_ms`** is the time passed to `reply()`; the grader uses the time of the user turn that carried the message.
- **`sent`** is `false` on a reply the turn limit stopped (decided, never sent), and absent otherwise.
- **`leak_found`** is always false with this simulated user: a rewording that adds anything is never sent.

### result

```json
{
  "end_reason": "assistant_finished",
  "reset": {"verified": true, "method": "new_chat_empty_history", "isolated": true},
  "cleanup_errors": {},
  "opening_sent": true,
  "account_ok": true,
  "phase": "close",
  "started_at_ms": 1790272022098,
  "ended_at_ms": 1790272189779
}
```

`end_reason` is one of:
- `assistant_finished`: the simulated user saw the task as finished and ended the run.
- `user_ended`
- `turn_limit`
- `timeout`: 3 nudges in a row with no reply, or the 45-minute deadline.
- `setup_error`: the runner, a service or the assistant's adapter failed. The run is not counted.
- `judge_error`: a judge the simulated user needs failed mid-run. The run is not counted.

The other fields:
- **`error`**: a fixed, credential-free code for what failed, when something did (`reset_not_verified`, `prompt_not_sent`, ...).
- **`phase`**: where the run was when it ended: `world`, `user`, `services`, `sign_in`, `reset`, `connect`, `converse` or `close`.
- **`opening_sent`**: whether the opening message may have reached the assistant. Retry a failed run only when it is false.
- **`reset.isolated`**: `false` when the reset is verified but keeps state an earlier run could have touched, such as a shared workspace. The grader flags such runs and the report counts them apart.
- **`account_ok`**: `false` when the assistant itself refused to open or reset (an `EpisodeStepError` while opening or resetting), `null` when another failure there left that unknown, else `true`.
- **`started_at_ms` / `ended_at_ms`**: from the start of setup to when the assistant's adapter finished closing. Collecting and releasing the services, and grading, come after.

## The conversation loop

1. Open the assistant, reset it and prove the reset. Connect the task's services, and wait until the assistant has stopped posting about the setup.
2. Send `user.opening()` as the first user turn.
3. Wait for the assistant to finish its turn: nothing new for 5 seconds and no answer still coming. If it sends nothing for 3 minutes, send `user.nudge()`.
4. After each assistant turn, call `user.reply(turns, at_ms=...)`. Its `action` says what to do: send `text`, `wait` for the assistant's next message, or `end` the run. The same text may be sent more than once, such as a second "yes, go ahead".
5. Repeat 3 and 4 until the simulated user ends the run, the run reaches 30 user turns, or it times out.
6. Collect each service's requests and its start and end state, close the assistant, release the services, and save the record.

`run_task.run_task()` implements this loop for any assistant adapter (see [developing-episodes.md](developing-episodes.md)); `run_task.simulated_user(task)` builds the simulated user from the READER, JUDGE and WRITER roles.

## Grading

```bash
uv run assistanteval report runs/ --out grades/
```

This grades every run against the task its record holds and prints the pass rate for each task. See [grader/README.md](grader/README.md).
