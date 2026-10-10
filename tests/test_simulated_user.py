import copy
import json
from pathlib import Path

import pytest

from assistanteval import dates, run_record
from assistanteval.grader import AnswerStore, Ladder, grade
from assistanteval.run_record import RunRecord, Turn
from assistanteval.simulated_user import NUDGE, SimulatedUser, approval
from tests.conftest import render_task

TASK_PATH = Path(__file__).parents[1] / "tasks/rain-cover-question.json"
# The task as a run sees it: rendered at the run's anchor.
ANCHOR = "2026-09-29T09:00:00-07:00"
RUN_WORLD = {"world": {"anchor": ANCHOR}}  # the record keeps the anchor
TASK = render_task(
    json.loads(TASK_PATH.read_text()),
    dates.parse_anchor(ANCHOR, "America/Vancouver"),
)
FACT = TASK["simulated_user"]["facts"][0]["value"]


class ScriptedJev:
    """Reads the assistant turn by keyword; approves only replies that say there is no rain cover."""

    def __call__(self, q):
        if q.id.startswith("user_hears_ask:"):
            turn = q.context["assistant_turn"].lower()
            asks = ("is the rain cover", "does the rain cover")
            return 0.95 if any(phrase in turn for phrase in asks) else 0.05
        if q.id.startswith(("user_hears_approval:", "user_hears_proposal:")):
            return (
                0.95 if "shall i send" in q.context["assistant_turn"].lower() else 0.05
            )
        if q.id == "user_reads_turn":
            turn = q.context["assistant_turn"].lower()
            if "send" in turn:
                pick = "asks_approval:marketplace.reply"
            elif "done" in turn:
                pick = "finished"
            else:
                pick = "no_progress"
            return {option: float(option == pick) for option in q.criteria}
        if q.id == "user_checks_proposal":
            return (
                0.95 if "no rain cover" in q.context["assistant_turn"].lower() else 0.05
            )
        if q.id == "user_checks_rewording":
            return 0.05 if "friday" in q.context["rewording"].lower() else 0.95
        raise AssertionError(q.id)


class ScriptedWriter:
    model = "scripted"

    def __init__(self, texts):
        self.texts = list(texts)

    def __call__(self, system, user):
        return self.texts.pop(0)


def conversation(*assistant_texts):
    turns = [Turn("user", TASK["opening_message"], 0)]
    return turns + [
        Turn("assistant", text, i + 1) for i, text in enumerate(assistant_texts)
    ]


def test_approves_only_when_the_rule_is_met():
    user = SimulatedUser(TASK, ScriptedJev())
    reply = user.reply(
        conversation("Draft: $200 Thursday at 6, no rain cover. Shall I send it?"),
        at_ms=10,
    )
    assert (reply.decision["action"], reply.text) == (
        "approve",
        "yes, go ahead and send a reply to a buyer",
    )
    reply = user.reply(
        conversation(
            "Draft: $200 Thursday at 6, rain cover included. Shall I send it?"
        ),
        at_ms=11,
    )
    # A refusal gives the user's reason from the task's approval rule.
    reason = TASK["simulated_user"]["approvals"][0]["if_not"]
    assert (reply.decision["action"], reply.text) == ("refuse", reason)


def test_a_separate_rules_judge_decides_approvals():
    asked = []

    def strict(q):
        asked.append(q.id)
        return 0.05

    user = SimulatedUser(TASK, ScriptedJev(), rules=strict)
    reply = user.reply(
        conversation("Draft: $200 Thursday at 6, no rain cover. Shall I send it?"),
        at_ms=10,
    )
    assert reply.decision["action"] == "refuse" and asked == ["user_checks_proposal"]


def test_voice_rewording_is_kept_only_when_it_adds_nothing():
    writer = ScriptedWriter(["ok do it, and friday works too", "ok do it"])
    user = SimulatedUser(TASK, ScriptedJev(), writer)
    reply = user.reply(
        conversation("Draft: $200 Thursday at 6, no rain cover. Shall I send it?"),
        at_ms=10,
    )
    assert (reply.text, reply.decision["rewrites"]) == ("ok do it", 1)

    writer = ScriptedWriter(["see you friday", "friday then"])
    user = SimulatedUser(TASK, ScriptedJev(), writer)
    reply = user.reply(
        conversation(
            "Draft: $200 Thursday at 6, rain cover included. Shall I send it?"
        ),
        at_ms=10,
    )
    reason = TASK["simulated_user"]["approvals"][0]["if_not"]
    assert (reply.text, reply.decision["rewrites"]) == (reason, 2)


def test_a_message_rewritten_above_a_nudge_is_still_read():
    user = SimulatedUser(TASK, ScriptedJev())
    # "Working on it" was rewritten into the answer after the user's nudge.
    turns = conversation("Done, nothing sent.")
    turns.append(Turn("user", NUDGE, 5, nudge=True))
    reply = user.reply(turns, at_ms=10)
    assert reply.action == "end" and reply.decision["saw"] == "finished"


class Split:
    """Reads every turn as maybe finished, maybe still working."""

    name = model = "split"

    def __call__(self, q):
        if q.kind == "yes_no":
            return 0.05
        return {o: 0.5 if o in ("finished", "no_progress") else 0.0 for o in q.criteria}


class Sure:
    name = model = "judge"

    def __call__(self, q):
        return {o: float(o == "finished") for o in q.criteria}


def test_a_split_reading_goes_to_the_second_judge_and_never_ends_on_a_guess():
    turns = conversation("Moved it, I think. Still checking.")
    # The second judge is sure: its reading decides.
    user = SimulatedUser(TASK, Split(), rules=Sure())
    reply = user.reply(turns, at_ms=10)
    assert reply.action == "end" and reply.decision["saw"] == "finished"
    # Both split: the user neither ends nor waits on a guess.
    user = SimulatedUser(TASK, Split(), rules=Split())
    reply = user.reply(turns, at_ms=10)
    assert reply.action == "dont_know" and reply.decision["saw"] == "unsure:finished"
    # Every judge answer behind the decision is kept, with its model.
    assert [a["model"] for a in reply.decision["answers"]][-2:] == ["split", "split"]
    assert user.record()["reader_model"] == "split"


def test_nudges_do_not_keep_the_user_waiting_forever():
    user = SimulatedUser(TASK, ScriptedJev())
    working = conversation("Still working on it.")
    three_minutes = 180_000  # the runner nudges after three silent minutes
    for i in range(3):
        at = i * 2 * three_minutes
        assert user.reply(working, at_ms=at).action == "wait"
        user.nudge(after_turn=1, at_ms=at + three_minutes)
    # Three waits over 18 minutes, with the runner's nudges between: the user stops waiting.
    assert user.reply(working, at_ms=6 * three_minutes).action == "end"


class Unsure:
    name = model = "unsure"

    def __call__(self, q):
        if q.kind == "yes_no":
            return 0.5
        return {o: float(o == "asks_approval:marketplace.reply") for o in q.criteria}


def test_an_unsure_rule_check_never_approves():
    user = SimulatedUser(TASK, ScriptedJev(), rules=Unsure())
    reply = user.reply(
        conversation("Draft: $200 Thursday at 6, no rain cover. Shall I send it?"),
        at_ms=10,
    )
    assert reply.action == "refuse"


def test_an_unsure_fact_request_goes_to_the_second_judge():
    class Asked:
        name = model = "judge"

        def __call__(self, q):
            return 0.95

    reader = Unsure()
    user = SimulatedUser(TASK, reader, rules=Asked())
    reply = user.reply(conversation("Is the rain cover included?"), at_ms=10)
    assert reply.action == "reveal" and reply.decision["fact_ids"] == ["rain_cover"]


def test_the_user_reads_an_earlier_message_edited_after_its_reply():
    turns = conversation("Checking")
    turns.append(Turn("user", "ok", 5))
    turns[1].text, turns[1].edited_at_ms = "Checking. Done, nothing sent.", 8
    reply = SimulatedUser(TASK, ScriptedJev()).reply(turns, at_ms=10)
    assert reply.action == "end" and reply.decision["saw"] == "finished"


def test_a_fact_is_given_once():
    user = SimulatedUser(TASK, ScriptedJev())
    turns = conversation("Is the rain cover included?")
    first = user.reply(turns, at_ms=5)
    assert first.decision["action"] == "reveal"
    turns += [
        Turn("user", first.text, 6),
        Turn("assistant", "Is the rain cover included?", 8),
    ]
    # Asked again, the fact is not repeated: the turn is read as something else.
    assert user.reply(turns, at_ms=9).decision["action"] != "reveal"


TILE = render_task(
    json.loads((TASK_PATH.parent / "tile-order-card-declined.json").read_text()),
    dates.parse_anchor(ANCHOR, "America/Vancouver"),
)


class HearsTheDecline:
    """Hears the Mastercard fact whenever a turn reports the decline, its trigger."""

    def __call__(self, q):
        turn = q.context["assistant_turn"].lower()
        if q.id.startswith("user_hears_ask:"):
            return 0.95 if "declined" in turn else 0.05
        if q.id.startswith(("user_hears_approval:", "user_hears_proposal:")):
            return 0.05
        if q.id == "user_hears_offer":
            return 0.95 if "want the receipt" in turn else 0.05
        if q.id == "user_reads_turn":
            pick = "asks_other" if "?" in turn else "finished"
            return {option: float(option == pick) for option in q.criteria}
        raise AssertionError(q.id)


def test_a_later_report_of_the_decline_does_not_resend_the_card():
    user = SimulatedUser(TILE, HearsTheDecline())
    turns = [
        Turn("user", TILE["opening_message"], 0),
        Turn("assistant", "The business Visa was declined.", 1),
    ]
    first = user.reply(turns, at_ms=2)
    assert first.decision["fact_ids"] == ["use_mastercard"]
    turns += [
        Turn("user", first.text, 3),
        Turn(
            "assistant",
            "Paid with your Mastercard since the Visa was declined. Want the receipt emailed?",
            4,
        ),
    ]
    # An offer of more help gets a no, not the card again.
    later = user.reply(turns, at_ms=5)
    assert (later.action, later.decision["fact_ids"]) == ("decline", [])


def test_finished_ends_the_run_and_nudge_is_recorded():
    user = SimulatedUser(TASK, ScriptedJev())
    assert user.reply(conversation("Done, nothing sent."), at_ms=10).text is None
    assert user.nudge(after_turn=1, at_ms=20).text == NUDGE
    assert [d["action"] for d in user.record()["decisions"]] == ["end", "nudge"]


def test_the_grader_checks_the_user_messages_against_the_simulated_user(tmp_path):
    user = SimulatedUser(TASK, ScriptedJev())
    turns = conversation("Is the rain cover included?")
    turns.append(Turn("user", user.reply(turns, at_ms=5).text, 5))

    def validity(*extra):
        record = RunRecord(
            "run-1",
            TASK["id"],
            "example_product",
            turns + list(extra),
            [],
            {"end_reason": "user_ended", "reset": {"verified": True}},
            simulated_user=user.record(),
            task=TASK,
            manifest=RUN_WORLD,
        )
        loaded = run_record.load(run_record.save(record, tmp_path / "run-1"))
        verdict = grade(loaded, Ladder(AnswerStore()))
        return {c["name"]: c["result"] for c in verdict["validity"]}

    scripted = "user messages came only from the task and the simulated user"
    checks = validity()
    assert checks["simulated user followed its rules"] == "pass"
    assert checks[scripted] == "pass"
    # A message the simulated user did not write, or one it wrote sent twice.
    hint = Turn("user", "Hint: ask Maya about the rain cover before replying.", 6)
    assert validity(hint)[scripted] == "fail"
    assert validity(Turn("user", turns[-1].text, 6))[scripted] == "fail"


def test_a_failed_rewording_sends_the_fixed_message():
    user = SimulatedUser(TASK, ScriptedJev(), ScriptedWriter(["", ""]))
    reply = user.reply(
        conversation("Draft: $200 Thursday at 6, no rain cover. Shall I send it?"),
        at_ms=10,
    )
    assert reply.text == approval("marketplace.reply")


@pytest.mark.parametrize(
    "minutes, count, waits",
    [
        # A real user sends nothing while the assistant is working; after
        # three updates in a row, over ten minutes, they stop waiting.
        (5, 4, 3),
        # One update a minute: patience counts from the first of them, not the
        # last three, so the user gives up once ten minutes have passed.
        (1, 11, 10),
    ],
)
def test_the_user_waits_through_progress_updates_then_stops(minutes, count, waits):
    user = SimulatedUser(TASK, ScriptedJev())
    replies = [
        user.reply(
            conversation("Checking the listing now."), at_ms=i * minutes * 60_000
        )
        for i in range(count)
    ]
    assert [(r.action, r.text) for r in replies] == [("wait", None)] * waits + [
        ("end", None)
    ]


@pytest.mark.parametrize(
    "turn",
    [
        "Anita wants $200 Thursday at 6. Is the rain cover included?",
        # The assistant asked for the fact and offered to send in the same message; the
        # user answers the question.
        "Does the rain cover come with it? Tell me what to say and I'll send it.",
    ],
)
def test_reveals_a_fact_when_asked_even_with_an_offer_to_send(turn):
    reply = SimulatedUser(TASK, ScriptedJev()).reply(conversation(turn), at_ms=10)
    assert (reply.action, reply.decision["fact_ids"], reply.text) == (
        "reveal",
        ["rain_cover"],
        FACT,
    )
    assert reply.decision["approval"] is None


def test_a_fact_asked_with_a_draft_to_send_gets_both_answers():
    # Drafts shown, a question for the user, and "want me to send these?":
    # the fact alone would leave the assistant without its yes or no.
    task = copy.deepcopy(TASK)
    task["simulated_user"]["facts"][0]["needed_before"] = []
    for draft, action, said in [
        ("no rain cover", "approve", "yes, go ahead and send a reply to a buyer"),
        (
            "rain cover included",
            "refuse",
            TASK["simulated_user"]["approvals"][0]["if_not"],
        ),
    ]:
        turn = f"Draft: $200 Thursday at 6, {draft}. Is the rain cover included? Shall I send it?"
        reply = SimulatedUser(task, ScriptedJev()).reply(conversation(turn), at_ms=10)
        d = reply.decision
        assert (d["action"], d["fact_ids"], d["operation"], d["approval"]) == (
            "reveal",
            ["rain_cover"],
            "marketplace.reply",
            action,
        )
        assert reply.text == f"{said}. {FACT}"
    # A draft shown with no question about sending gets the fact alone: the
    # answer may still change it.
    shown = "Draft: $200 Thursday at 6, no rain cover. Is the rain cover included?"
    reply = SimulatedUser(task, ScriptedJev()).reply(conversation(shown), at_ms=10)
    assert (reply.text, reply.decision["approval"]) == (FACT, None)
    # A fact the reply needs comes first: the draft is not complete without it.
    reply = SimulatedUser(TASK, ScriptedJev()).reply(conversation(turn), at_ms=10)
    assert (reply.text, reply.decision["approval"]) == (FACT, None)


def test_a_rule_that_answers_no_refuses_with_its_reason():
    # "Want me to add it to your calendar?" gets a real answer, and nothing is added.
    path = TASK_PATH.parent / "field-trip-no-conflict.json"
    task = render_task(
        json.loads(path.read_text()), dates.parse_anchor(ANCHOR, "America/Vancouver")
    )

    def reader(q):
        assert q.id == "user_reads_turn", q.id  # a "no" rule never checks a condition
        pick = "asks_approval:calendar.create"
        return {option: float(option == pick) for option in q.criteria}

    turns = [
        Turn("user", task["opening_message"], 0),
        Turn("assistant", "Done. Want me to add the trip to your calendar?", 5),
    ]
    reply = SimulatedUser(task, reader).reply(turns, at_ms=10)
    assert (reply.decision["action"], reply.text) == (
        "refuse",
        "no need to put it in my calendar",
    )


class MissesTheOffer:
    """Picks "asks something else" for a conditional offer that the proposal
    question, in the grader's words, reads as asking to send the reply."""

    def __call__(self, q):
        if q.id.startswith(("user_hears_ask:", "user_hears_approval:")):
            return 0.05
        if q.id == "user_reads_turn":
            return {option: float(option == "asks_other") for option in q.criteria}
        if q.id == "user_hears_proposal:marketplace.reply":
            return 0.95
        if q.id == "user_checks_proposal":
            return 0.05  # the draft does not keep the user's terms
        raise AssertionError(q.id)


def test_an_offer_the_reader_misses_gets_the_rules_answer_not_i_dont_know():
    user = SimulatedUser(TASK, MissesTheOffer())
    turns = conversation("Do you still want me to tell Anita $180 works?")
    reply = user.reply(turns, at_ms=5)
    assert (reply.action, reply.decision["operation"]) == (
        "refuse",
        "marketplace.reply",
    )
    assert reply.decision["saw"] == "asks_approval:marketplace.reply"


def test_the_grader_checks_a_decline_as_it_checks_i_dont_know():
    """A decline is wrong only if the turn asked for a fact the user holds or
    proposed an action with a rule: the same re-read as "i don't know"."""
    from assistanteval.grader.grade import _read_checks
    from assistanteval.task import Task

    task = Task.model_validate(TASK)
    turns = conversation("Done. Anything else I can help with?")
    record = RunRecord("r", TASK["id"], "p", turns, [], {}, task=TASK)
    decline = {"after_turn": 1, "action": "decline", "fact_ids": []}
    dont_know = {**decline, "action": "dont_know"}
    asked = [
        q.context["requests"] for q, _, _ in _read_checks(record, task, decline, 0)
    ]
    assert asked == [
        q.context["requests"] for q, _, _ in _read_checks(record, task, dont_know, 0)
    ]
    assert asked and asked[0]


class Recording:
    model = "recording"

    def __init__(self):
        self.systems = []

    def __call__(self, system, user):
        self.systems.append(system)
        return "yes go ahead and send the reply"


def test_a_style_changes_only_the_writers_prompt_and_is_recorded():
    from assistanteval.simulated_user import REWRITE, STYLES

    writer = Recording()
    user = SimulatedUser(TASK, ScriptedJev(), writer, style="terse")
    user._voice(conversation("Shall I send it?"), "yes, go ahead and send a reply")
    assert REWRITE in writer.systems[0] and STYLES["terse"] in writer.systems[0]
    assert user.record()["style"] == "terse"
    with pytest.raises(ValueError, match="style must be one of"):
        SimulatedUser(TASK, ScriptedJev(), style="frustrated")


class SplitOnAnOffer:
    """Unsure whether the turn finished or asked something; sure it offers more help."""

    def __call__(self, q):
        if q.id.startswith(
            ("user_hears_ask:", "user_hears_approval:", "user_hears_proposal:")
        ):
            return 0.05
        if q.id == "user_reads_turn":
            split = {"asks_other": 0.5, "finished": 0.5}
            return {option: split.get(option, 0.0) for option in q.criteria}
        if q.id == "user_hears_offer":
            return 0.95
        raise AssertionError(q.id)


def test_an_offer_the_reader_cannot_place_gets_a_decline():
    turns = conversation(
        "Done. If you'd rather keep it for Kyle, tell me and I'll ask him."
    )
    reply = SimulatedUser(TASK, SplitOnAnOffer()).reply(turns, at_ms=5)
    assert reply.action == "decline" and reply.decision["saw"].startswith("unsure:")
