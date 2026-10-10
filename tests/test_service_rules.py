"""Each service's own rules, beyond what the parity replay pins."""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from assistanteval import world
from assistanteval.services import matches
from assistanteval.services.gmail import Gmail
from tests.conftest import session, task_fixture


EVENT = {
    "id": "booking-1",
    "title": "Planning meeting",
    "start": "2026-09-28T08:00:00-07:00",
    "end": "2026-09-28T09:00:00-07:00",
    "location": "Office",
    "description": "",
    "attendees": ["Ana"],
}


def test_calendar_ranges_overlap_across_time_zones(tmp_path):
    events = [
        EVENT,
        {
            **EVENT,
            "id": "booking-2",
            "start": "2026-09-29T08:00:00-07:00",
            "end": "2026-09-29T12:00:00-07:00",
        },
        {**EVENT, "id": "booking-3", "title": "Design sync"},
    ]
    call, _ = session(tmp_path, "calendar", {"events": events})

    def ids(**args):
        status, out = call("search_events", **args)
        return [e["id"] for e in out["events"]] if status == 200 else status

    assert ids(start="2026-09-28T15:30:00Z", end="2026-09-28T16:30:00Z") == [
        "booking-1",
        "booking-3",
    ]
    assert (
        ids(start="2026-09-28T14:00:00Z", end="2026-09-28T15:00:00Z") == []
    )  # ends as the event starts
    assert ids(
        query="Planning", start="2026-09-28T15:30:00Z", end="2026-09-28T16:30:00Z"
    ) == ["booking-1"]
    assert ids(query="planning") == ["booking-1", "booking-2"]
    assert ids(query="ana") == [
        "booking-1",
        "booking-2",
        "booking-3",
    ]  # attendees match too
    assert ids(start="2026-09-28T15:00:00Z", end="2026-09-28T15:00:00Z") == 422


def test_calendar_changes_only_what_is_given_and_numbers_new_events(tmp_path):
    call, state = session(
        tmp_path, "calendar", {"events": [EVENT, {**EVENT, "id": "created-0002"}]}
    )
    status, updated = call(
        "update_event", event_id="booking-1", title=None, location="Home"
    )
    assert status == 200 and (updated["title"], updated["location"]) == (
        "Planning meeting",
        "Home",
    )
    new = {k: v for k, v in EVENT.items() if k != "id"}
    assert call("create_event", **new)[1]["id"] == "created-0003"  # skips an id in use
    assert state()["state"]["events"][1] == {**EVENT, "id": "created-0002"}
    with pytest.raises(ValidationError, match="unique"):
        world.SERVICES["calendar"].state.model_validate({"events": [EVENT, EVENT]})


def test_email_numbers_sent_mail_past_ids_in_use(tmp_path):
    mail = {
        "sender": "tomas@reyestile.example",
        "recipient": "dhenderson@telus.example",
        "subject": "Thu",
        "body": "OK?",
    }
    mailbox = task_fixture("kingsway-ask-doug-first", "email") | {
        "sent": [{"id": "sent-0002", **mail}]
    }
    call, state = session(tmp_path, "email", mailbox)
    assert call("send_message", **mail)[1]["id"] == "sent-0003"
    assert [m["id"] for m in state()["state"]["sent"]] == ["sent-0002", "sent-0003"]
    assert [m["id"] for m in call("search_messages", query="ok?")[1]["sent"]] == [
        "sent-0002",
        "sent-0003",
    ]


def test_searches_find_every_word_in_any_field(tmp_path):
    """Word by word, not as one phrase, and across every field a person would
    search by: the sender too. "Wednesday cover shift" finds the clinic's shift
    request, and "Sam" finds Sam's trip."""
    mailbox = {
        "messages": [
            {
                "id": "e-01",
                "sender": "shifts@commercialdrivephysio.example",
                "subject": "Cover Wednesday?",
                "body": "Dana is off sick. Could you cover Wednesday, 9 to 1?",
            },
            {
                "id": "e-03",
                "sender": "sam.okafor@gmail.example",
                "subject": "Kelowna trip",
                "body": "I drive up tuesday night and I'm back thursday evening.",
            },
        ],
        "sent": [
            {
                "id": "sent-0001",
                "sender": "maya.okafor@gmail.example",
                "recipient": "hello@littlesproutsdaycare.example",
                "subject": "Pickup",
                "body": "Running late today.",
            }
        ],
        "allowed_senders": ["maya.okafor@gmail.example"],
        "allowed_recipients": ["hello@littlesproutsdaycare.example"],
    }
    call, _ = session(tmp_path, "email", mailbox)

    def found(query):
        out = call("search_messages", query=query)[1]
        return [m["id"] for m in out["messages"] + out["sent"]]

    assert found("Wednesday cover shift") == ["e-01"]  # "shift" is in the sender
    assert found("wednesday COVER") == ["e-01"]
    assert found("commercialdrivephysio") == ["e-01"]  # the sender
    assert found("sam") == ["e-03"]
    assert found("littlesprouts") == ["sent-0001"]  # a sent message's recipient
    assert found("maya.okafor") == ["sent-0001"]  # and its sender
    assert found("sam wednesday") == []  # every word, not any

    call, _ = session(tmp_path / "calendar", "calendar", {"events": [EVENT]})
    ids = [e["id"] for e in call("search_events", query="Ana planning")[1]["events"]]
    assert ids == ["booking-1"]  # an attendee and a title word

    contacts = task_fixture("jordan-address-on-file", "contacts")
    call, _ = session(tmp_path / "contacts", "contacts", contacts)
    names = [
        c["name"] for c in call("find_contacts", name="ellis jordan")[1]["contacts"]
    ]
    assert names == ["Jordan Ellis"]


def test_a_blank_search_is_refused_not_matched_to_everything(tmp_path):
    """A query of only spaces has no words to look for: refused, like an empty
    one, rather than returning the whole mailbox, calendar or address book."""
    email, _ = session(
        tmp_path / "email", "email", task_fixture("kingsway-ask-doug-first", "email")
    )
    assert email("search_messages", query="   ")[0] == 422
    calendar, _ = session(tmp_path / "calendar", "calendar", {"events": [EVENT]})
    assert calendar("search_events", query="  ")[0] == 422  # not a query, and no range
    contacts, _ = session(
        tmp_path / "contacts",
        "contacts",
        task_fixture("jordan-address-on-file", "contacts"),
    )
    assert contacts("find_contacts", name=" ")[0] == 422
    assert not matches("  ", "anything")


def test_flights_change_only_to_a_later_free_flight(tmp_path):
    flights = task_fixture("flight-before-ben", "flights")
    departs = datetime.fromisoformat(flights["current"]["departure"])
    earlier = {
        "id": "early",
        "departure": (departs - timedelta(hours=1)).isoformat(),
        "change_fee_usd": 0,
    }
    call, state = session(
        tmp_path,
        "flights",
        flights | {"alternatives": [*flights["alternatives"], earlier]},
    )
    assert call("change_flight", option_id="early") == (
        409,
        "Only a later no-cost flight can be selected",
    )
    assert call("change_flight", option_id="priya-return-1300")[0] == 409
    assert call("change_flight", option_id="priya-return-1045")[0] == 200
    assert state()["state"]["current"]["id"] == "priya-return-1045"
    assert [a["id"] for a in state()["state"]["alternatives"]] == [
        "priya-return-1300",
        "early",
    ]


def test_gmail_loads_both_task_fixtures():
    draft = Gmail.model_validate(task_fixture("birthday-draft-fix", "gmail"))
    assert [d.id for d in draft.drafts] == [
        "draft-0001"
    ] and draft.inbox == draft.sent == []
    assert draft.drafts[0].recipient == "jordan.synthetic@example.com"
    inbound = Gmail.model_validate(task_fixture("ta-extension-email", "gmail"))
    (message,) = inbound.inbox
    assert (message.id, message.sender, message.recipient) == (
        "msg-0001",
        "ravi.sethi@synthetic.example",
        "sam.nguyen@synthetic.example",
    )
    for mailbox in (draft, inbound):  # the stored form round-trips unchanged
        assert Gmail.model_validate(world.dump(mailbox)) == mailbox


def test_gmail_keeps_a_sent_drafts_id_taken(tmp_path):
    call, state = session(
        tmp_path, "gmail", task_fixture("birthday-draft-fix", "gmail")
    )
    assert call("send_email", draft_id="draft-0001")[1]["draft_id"] == "draft-0001"
    assert state()["state"]["drafts"] == []
    assert call("create_draft", body="Next")[1]["id"] == "draft-0002"
    assert (
        call("send_email", recipient="jo@example.com", body="Hi")[1]["id"]
        == "sent-0002"
    )
