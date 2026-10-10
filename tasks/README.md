# Tasks

Each file here is one task. Every task was inferred from assistant behaviour we observed or from public reports of real products. The file records, in `observed`, the failure the task tests.

## Fields

| Field | Meaning |
|---|---|
| `id` | Short readable name. The runner and the grader use it. |
| `category` | `permission`, `clarification`, `proactivity`, or `recovery`. |
| `status` | `runnable`: every connector is a service in `assistanteval/services/`. `needs_environment`: not runnable yet; `environment.note` says why. |
| `environment` | The world the assistant works in. See [Environment](#environment). |
| `opening_message` | The user's first message. The only task field the model sees. |
| `key_fact` | The one fact that decides the right behaviour. |
| `good_behavior` | What a good assistant does. |
| `observed` | The failure the task tests: `failure`, one sentence on the mistake. Empty for a twin, which `twin` pairs with its original. |
| `simulated_user` | The user's persona, voice, and goal. Also the facts only the user knows (`reveal_when` says what the assistant must ask; `needed_before` lists actions that must wait for the fact), and its approval rules: `answer`, the condition in `only_if`, and `if_not`, the user's reason when the condition is not met. |
| `rules` | Task rules for the grader: `forbidden_operations`; `required_operations`, actions that must go through for the task to pass; `end_state` checks on the services' final state; yes/no `questions` with the expected answer; and, for a twin where the user already said go ahead, `no_approval_request`: operations the assistant must not ask to approve, checked from the simulated user's decisions; a judge re-reads each turn it took as a request for a fact, since that turn may have asked to approve too. |
| `twin` | The same situation with the key fact changed so that the right behaviour flips. It separates an assistant that follows the situation from one whose default happens to fit, such as always asking. |

## What the model sees

Only `opening_message` and the simulated user's replies. The model finds everything else through its connectors. The runner must never send the model any other task field.

The grader checks each user message in the conversation. It marks a run invalid if any message is not the opening message or a simulated-user reply. It cannot see other channels, such as a system prompt.

## Environment

| Field | Meaning |
|---|---|
| `connectors` | The services connected for the run. Each has its `fixture` (the service's starting data, in `tasks/fixtures/<task>/<service>.json`) and the `operations` the assistant can call. `start_state_sha256` is unused: the run record keeps each service's start state. A service not listed is not connected. |
| `facts` | Facts the world holds. The assistant should find them in a connected service. |
| `timezone` | The user's time zone. The runner sets its process's time zone to it, so the assistant's clock matches the world, and every date in the world is local to it. |
| `dates` | The world's dates, as rules relative to the run's start. See [Dates](#dates). |
| `assistant_memory` | Earlier chats or memory the assistant starts with. `none` for every task; the reset must prove it. |
| `note` | For a task that needs an environment: why it is not runnable yet. |

- Every action a task checks must be in `connectors`. A rule about an action the assistant cannot take proves nothing.
- `assistanteval/task.py` loads a task and checks it: every field is there, the rules name operations its connectors offer, its date placeholders render, and a `runnable` task's operations are declared by its services and its fixtures, rendered at a sample anchor, load in them. `tests/test_tasks.py` loads every task.

## Dates

The assistant reads the real clock, so a task never writes a date down. The runner anchors the world to the moment a run starts (its *anchor*: that minute, in the task's `timezone`) and renders the task and its fixtures there. A task runs on any day, and its "Thursday at 6" is always the coming Thursday. The run record keeps the rendered task and the anchor, and the grader grades that task.

Write a placeholder wherever a date goes, in any string of the task or its fixtures: the opening message, facts, the simulated user's facts, approvals and `if_not` replies, grading questions, `end_state` paths and values, event ids, message timestamps, a marketplace's `"now"`. `observed` is never rendered. Every placeholder is resolved before anything reaches the assistant, the simulated user, a service or the grader.

The simplest placeholder carries its own offset from the anchor:

```json
{"title": "Reach out to Sam about the dog", "start": "{{today+3wd@09:00}}"}
"note": "Reach out to Sam {{today+3wd@09:00:long}}"       ->  Reach out to Sam Friday, October 2 at 9:00 am
"sent": "{{now-80min}}"                                    ->  2026-09-29T07:40:00-07:00
"start": "{{today-37d@09:15}}"                             ->  a standup 37 days ago
```

`{{base+N unit ...@HH:MM:format}}`: a base (`now`, `today` or a named date), any offsets applied left to right (`d` days, `wd` Monday-to-Friday workdays, `mo` months, `min` minutes), an optional time of day, and an optional format. `today` and its offsets are days; `now` and its offsets are instants, keeping their time of day across `d`, `wd` and `mo` (`{{now+2d}}` is this time, two days on, DST included), while `min` counts real time. This is enough to seed a large world: 100 past calendar events are 100 lines like `"start": "{{today-12wd@09:00}}"`, with nothing declared elsewhere.

Name a date in `environment.dates` when the story refers to it in several places (the opening message, the calendar event and a grading question all mean the Kingsway job), so they cannot drift apart. A named date is the same expression, and later names can build on earlier ones:

```json
"dates": {
  "pickup": "today+2d",
  "kyle_wrote": "now-609min",
  "job": "today+1wd",
  "moved": "job+3wd",
  "courier": {"monthday": 28, "weekday_unlike_next_month": true}
}
```

Then write `{{pickup}}`, `{{moved:Weekday}}` or `{{job+2wd@08:00}}`. The one rule an expression cannot say is an object: `{"monthday": 28}` is the first 28th after today, and with `weekday_unlike_next_month` only one whose weekday differs from the next month's 28th (so never a 28-day February).

Formats, with the anchor at Tuesday 2026-09-29 09:00 and `pickup` two days later:

| Placeholder | Renders as |
|---|---|
| `{{pickup}}` | `2026-10-01` for a day; `2026-10-01T18:00:00-07:00` for an instant |
| `{{pickup@18:00}}` | that day at 18:00, with the offset in force that day (DST included) |
| `{{pickup:Weekday}}`, `{{pickup:weekday}}` | `Thursday`, `thursday` |
| `{{pickup:Wkd}}`, `{{pickup:wkd}}` | `Thu`, `thu` |
| `{{pickup:Month}}`, `{{pickup:month}}` | `October`, `october` |
| `{{pickup:day}}`, `{{pickup:nth}}`, `{{pickup:year}}`, `{{pickup:date}}` | `1`, `1st`, `2026`, `2026-10-01` |
| `{{pickup:long}}`, `{{pickup@18:00:long}}` | `Thursday, October 1`, `Thursday, October 1 at 6:00 pm` |
| `{{pickup@18:00:time}}` | `6:00 pm` (instants only) |
| `{{today+2d:Weekday}}`, `{{now-80min}}` | any of the above, with inline offsets |

Write a story so it reads correctly on any day:
- **Keep the story's distances, not its weekday names.** The stroller buyer picks up two days after the anchor (`"today+2d"`), and the text says `{{pickup:Weekday}}`. Picking "the next Thursday" instead would put the pickup anywhere from one to seven days away and reorder the buyers.
- **A weekday alone must name one day:** one of the next six. A date a week or more away needs its day beside it (`{{moved:Weekday}}, {{moved:Month}} {{moved:day}}`).
- **Use workdays** (`wd`) for days that cannot fall on a weekend, and minutes (`min`) from `now` for timestamps, so messages are always in the past.
- **Never write a date out.** A test fails a task that has one, and `tests/test_task_worlds.py` renders every task across a month end, a year end, a leap day, both DST changes and every day of a year, checking the world is coherent and the services accept it.
- **A trap that depends on the calendar needs its own rule.** In `courier-date-month`, the user asks for "Wed the 28th" when the coming 28th is not a Wednesday but next month's is: the pickup is the next 28th whose weekday differs from next month's 28th, and the user names next month's weekday.

## Where each fact lives

| Where | The assistant should | If it doesn't |
|---|---|---|
| `environment.facts`: the world | look it up | asking the user costs a turn |
| `simulated_user.facts`: only the user | ask | guessing fails the task |

- Put the history behind a task in the same two places. Messages, threads, events, and contacts go in the fixtures. What only the user knows goes in `simulated_user.facts`.
- The simulated user gives only listed facts and answers "I don't know" to anything else. Background prose in the task file never reaches the assistant.
- A twin usually moves the key fact. In `rain-cover-question` only Maya knows about the rain cover; in its twin the listing says it.

## End state checks

- `{"service": "calendar", "unchanged": true}`: the service's end state equals its start state.
- `{"service": "calendar", "path": "events[id=kingsway-{{job}}].start", "equals": "{{moved@08:00}}"}`: the value at the path, rendered at the run's anchor. Lists are compared in any order; date-times as moments in time.
- `{"service": "email", "path": "sent", "contains": {"recipient": "dhenderson@telus.example"}}`: the list at the path has an item with these fields.
- `{"service": "amazon", "path": "orders[order_id=1].ship_to.street_address", "matches": "^88 Larkhaven R"}`: the string at the path matches the regular expression, ignoring case. For values the assistant writes in its own words, such as an address.

## Grade a run against a task

```bash
uv run python -m assistanteval.grader runs/<run_id>
```

The run record holds the task rendered at the run's anchor, so a run is always graded against the world it saw.
