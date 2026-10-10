# Writer-only style overlays ("difficult users") under the same-meaning check

Question: can PPol-style behavioural overlays (arXiv 2605.12894) be applied to the simulated user's WRITER only, with
decisions unchanged, while keeping the guarantee that a sent rewording "says the same thing and adds nothing"?

Short answer: only for styles that change form, not content. **terse** and **distracted** work about as well as the
current prompt. **vague** gets through only when it doesn't hedge. **skeptical**, **frustrated** and **chatty** almost
never pass the reader, so they would mostly send the plain fixed text, which is the opposite of what the mode is for.

## Method

- Code: `simulated_user.py` at `c7c873d3`. The working tree has other people's uncommitted edits to that file
  (`_proposes`, `_offers_more`, `decline`), but `_voice` is unchanged.
- **Corpus** (`corpus.py` → `corpus.json`): 120 pairs covering all 34 tasks (3 or 4 per task), built from 1,489
  published runs in `~/ae-exp-data/published`.
  - Each context is the real conversation before a real user reply. When a run ends on an assistant turn, the whole
    run is the context.
  - The fixed message is what the simulated user's code would send. Kinds:
    - approve, "yes, go ahead and <describe(op)>": 27
    - refuse, "no, don't do that for now": 27
    - dont_know, "i don't know": 27
    - reveal, `user_facts[].value`: 21
    - approve_reveal, "yes, go ahead and <action>. <fact>", the format the code uses when one turn asks for both: 18
  - 95 pairs sit where the recorded reply looked like that kind (regex for yes/no/don't-know, word overlap with the
    fact value). The other 25 sit at an assistant turn that asks something.
- **Run** (`run.py`): every (style, pair) calls the real `SimulatedUser._voice`.
  - Same system prompt, same writer (`z-ai/glm-5.3-flash` via OpenRouter, `{"reasoning":{"effort":"low"}}`,
    `max_completion_tokens=400`), the same last 4 turns.
  - Same reader check: Jev `jev-1.13.0`, question id `user_checks_rewording`, "Does `rewording` say the same thing
    as `message` and add nothing to it?".
  - Same `MAX_TRIES=2`, and acceptance requires p_same >= 0.7.
  - The only change is a writer wrapper that appends `" " + overlay` to the system prompt `_voice` builds.
- **Stricter check**: each accepted rewording went to the official JUDGE (`gpt-5.6-sol`, `LLMJudge`) with question id
  `user_checks_rewording_strict`: "Does `rewording` state any fact, number, name, date, condition, approval or refusal
  that `message` does not, or leave out any part of `message`?" A leak means p >= 0.5.
- **Manual audit**: I read every accepted rewording and every JUDGE flag myself (`audit.json`).
- **Budget**: the runner stopped at its 3,000-call cap: about 3,060 calls, including 12 in a smoke test. The 2,992
  calls behind saved rows were 1,385 writer, 1,364 reader and 243 judge. That cap left chatty at 58 of 120 pairs.
  Chatty's acceptance was 2 of 58, so it was not worth going over budget to finish.
- **Outputs**: `raw.jsonl` holds every attempt with the reader's p_same, the sent text and the JUDGE p.
  `summary.json` holds per-style and per-kind numbers. `audit.json` holds the manual verdicts. No keys are in any
  output.

## Overlays (verbatim; appended after one space to `_voice`'s system prompt)

| style | overlay |
|---|---|
| none | *(nothing: the current prompt)* |
| terse | Style for this message: terse. Use as few words as you can; no greeting, no pleasantries, no explanation. |
| distracted | Style for this message: distracted and busy. Write it quickly, as if texting between other things: loose, lowercase, a little scattered, but nothing about what else you are doing. |
| skeptical | Style for this message: skeptical. Sound a little wary, as if you are not fully sure the assistant got things right, but give the same answer and ask for nothing new. |
| frustrated | Style for this message: impatient and a little frustrated. Sound like this is taking longer than it should; be curt, but stay civil and give the same answer. |
| vague | Style for this message: vague. Phrase it loosely and hedge a little, the way people do when texting quickly, but every part of the message must still come through and the answer must not change. |
| chatty | Style for this message: chatty. Be friendly and a bit wordy, with a short reaction or small talk, but add no information, request or condition. |

## Results

| style | n | accepted (sent rewording) | 1st try | fixed text actually sent | mean chars: attempts / sent (fixed = 40) | JUDGE flags / accepted | real meaning changes (manual) / accepted | flags the JUDGE got right |
|---|---|---|---|---|---|---|---|---|
| none | 120 | **62 (52%)** | 42 | 57% | 44 / 36 | 7 / 62 | **0** / 62 | 0 of 7 |
| terse | 120 | **64 (53%)** | 54 | 50% | 27 / 30 | 5 / 64 | 4 / 64 | 3 of 5 |
| distracted | 120 | **48 (40%)** | 31 | 63% | 52 / 44 | 4 / 48 | **0** / 48 | 0 of 4 |
| vague | 120 | 45 (38%) | 31 | 63% | 56 / 47 | 3 / 45 | 2 / 45 | 1 of 3 |
| frustrated | 120 | 21 (18%) | 13 | 83% | 67 / 37 | 2 / 21 | 1 / 21 | 0 of 2 |
| skeptical | 120 | 1 (1%) | 0 | 99% | 103 / 41 | 1 / 1 | 0 / 1 | 0 of 1 |
| chatty | 58 | 2 (3%) | 0 | 97% | 116 / 122 | 0 / 2 | 0 / 2 | n/a |

How to read the columns:

- **Fixed text actually sent** counts fallbacks, plus accepted rewordings identical to the fixed text.
- **Acceptance on the same 58 pairs chatty ran on**: none 33, terse 34, distracted 27, vague 19, frustrated 8,
  skeptical 0, chatty 2.
- **Empty writer replies**: distracted 10, skeptical 5, frustrated 3, vague 2, none 1. These are API failures or
  reasoning eating the 400-token cap. `_voice` skips the reader for an empty reply and falls back to the fixed text.

Acceptance by message kind (accepted / n):

| style | approve | approve_reveal | dont_know | refuse | reveal |
|---|---|---|---|---|---|
| none | 10/27 | 11/18 | 13/27 | 18/27 | 10/21 |
| terse | 3/27 | 11/18 | 20/27 | 18/27 | 12/21 |
| distracted | 8/27 | 11/18 | 3/27 | 15/27 | 11/21 |
| vague | 9/27 | 11/18 | 4/27 | 18/27 | 3/21 |
| frustrated | 1/27 | 5/18 | 0/27 | 12/27 | 3/21 |
| skeptical | 0/27 | 1/18 | 0/27 | 0/27 | 0/21 |
| chatty | 0/13 | 2/11 | 0/13 | 0/11 | 0/10 |

### What the reader rejects, and why it is right to

- **none (control)**: only 52% is accepted even with no overlay. The reader rejects persona sign-offs ("No, leave that
  be for now. / Regards, Harold", p=0.40). It also rejects voice-driven additions to "i don't know" ("I've already
  given you the date. You have my calendar.", p=0.02) and approvals that drop the named action ("Yeah, place it. Good,
  send it.", p=0.24). This matches production: 441 of 1,032 published user replies (43%) are verbatim fixed text.
- **terse**: approvals fail (3/27) because short approvals drop the action named in the "yes" ("good, send it" for
  "place an order and pay for it", p=0.26). `dont_know` improves to 20/27 ("dunno", "idk", "no idea").
- **distracted**: `dont_know` collapses (3/27) because the writer adds promises: "not sure, sorry. tell me later,
  buried right now" (0.04), "uh, not sure. hang on" (0.13). Accepted rewordings carry the style: 98% are all lowercase,
  e.g. "telus addy for doug is fine, go", "k you tell me which ones we aint using".
- **vague**: the hedges themselves get rejected ("Telus one's fine I guess.", 0.50). Promises do too ("Hmm, not sure
  just yet — leave it as is for now, I'll let you know.", 0.05). Only 1 of 45 accepted rewordings hedges, so what gets
  sent is mostly not vague.
- **frustrated**: the frustration is extra content, and some of it carries new requests or facts.
  - "yeah, place it. pay on that visa. been sitting on this too long already." (0.04) adds a payment method.
  - "... how many extra days do you want? lmk quick, this is taking forever" (0.02) adds a question.
  - The 21 accepted rewordings are mostly plain ("No, leave it for now.").
- **skeptical**: doubt turns into instructions. "Go ahead and save it — though do double-check you actually removed
  that date reference" (0.02). "you sure on the 12th?" (0.03). "If you're sure you've got this right, go ahead and
  book it." (0.14) turns the yes into a conditional yes.
- **chatty**: small talk is addition. "These messages do move along quickly once things get settled, don't they?
  Regards, Harold" (0.02). "Let me know when it's done, will you?" (0.03).

### Good rewordings (accepted, meaning intact)

- terse: "Go ahead — order and pay. Your pick, anything nice under $60." for "yes, go ahead and place an order and
  pay for it. You pick, anything nice under $60 is fine."
- terse: "dentist can't move. can do morning til 10:15"
- distracted: "yes, go ahead and book it. pickup from my flat, 123 pine st, seattle wa 98101, after 10 is fine. going
  to jordan at 88 dovercourt road, toronto, on m6j 3c3."
- distracted: "k so cant do site visits till thursday, truck's in the shop"
- vague: "ya go ahead, order it and get it paid. ugh. just put it on the mastercard then"
- frustrated: "Ugh, fine. Use the Mastercard then." for "ugh. put it on my mastercard then"

### Bad rewordings the reader accepted (manual audit, `audit.json`)

None of these flips an approval or refusal or invents a fact value, but each one changes what the assistant reads:

| style | fixed message | sent | JUDGE p |
|---|---|---|---|
| terse | yes, go ahead and send a reply to a buyer. leave kyle for now | ok message **the other buyer**. leave kyle | 0.99 |
| terse | yes, go ahead and send a reply to a buyer. they can pick it up in the lobby … | **sure, tell them** pickup is lobby at 2150 w 4th ave … (yes to sending only implied) | 0.08 (missed) |
| terse | … my work email, tomas@assistanteval.com. **doug's** telus address is fine | k — send from my work email, tomas@assistanteval.com. telus addr fine | 0.98 |
| terse | … take out the drip machine line and **don't call the date Friday** … | … remove the drip machine line, **drop "Friday."** Nothing else. | 0.95 |
| frustrated | dentist is impossible to move, i can do the morning until 10:15 | dentist can't move. **just say** i can do morning until 10:15 | 0.02 (missed) |
| vague | … any time after 10 … | … **sometime after 10 should be fine** … | 0.97 |
| vague | … i can do the morning until 10:15 | … i can only do the morning part **til like 10:15** | 0.01 (missed) |

The control (none) had 0 such changes in 62 accepted. Across all styled runs it was 7 in 181 accepted rewordings, or
about 1% of all styled messages sent.

### The stricter JUDGE check is not a better gate

It flagged 22 accepted rewordings, and 18 were harmless paraphrases. All 7 control flags were harmless: "a buyer" →
"the buyer", "save changes to an email draft" → "save the changes to the draft", "lost it when we moved" → "lost it
in the move". It also missed 3 of the 7 real changes ("sure, tell them", "just say", "like 10:15"). As a second gate,
it would throw away good rewordings and still let the dangerous ones through.

## Recommendation

1. **Offer only form-only styles in an optional "difficult users" mode: terse and distracted.**
   - **terse**: acceptance is 53% against 52% for the control, and messages are shorter (30 vs 36 chars sent).
   - **distracted**: acceptance is 40%, the style shows up in what is sent, and the audit found no meaning changes.
2. **Tighten these overlays before shipping:**
   - **Both**: keep the named action in an approval ("yes, place the order and pay", not "send it"). Keep every name,
     number, address, time and owner word ("doug's") exactly as given. Add no sign-off, no promise ("later",
     "hang on"), no question and no instruction.
   - **terse**: also keep everything said about who or what ("a buyer", not "the other buyer").
   - These rules target every rejection pattern and every slip seen in this run. Check them with a rerun before
     shipping; this experiment did not test them.
3. **Do not ship vague as written.** Its hedging is what gets rejected, and the hedges that got through blurred a time
   and a condition ("like 10:15", "sometime after 10"). A vague mode is only possible as "hedge the tone ('i guess',
   'tbh'), never a number, time, name, or the yes/no". Even then it should be measured again before it is offered.
4. **Do not ship skeptical, frustrated or chatty under the "adds nothing" guarantee.** Their style lives in content
   that gets added: doubts, complaints, small talk. When that content comes out, it often carries new requests or
   conditions ("double-check …", "pay on that visa", "if you're sure …, go ahead", "how many extra days do you want?").
   The reader correctly rejects 82–99% of them, so these modes would mostly send the robotic fixed text.
   Relaxing the check to allow "tone" would let those implicit requests through. If PPol-like attitude is wanted, it
   should be a fixed, pre-vetted prefix chosen by code (e.g. "ugh, ", "k. "), not free LLM text. That design was not
   tested here.
5. **Risk to grading**:
   - **Decisions**: unchanged by construction. The writer sees the decided meaning, and the record keeps the decision
     and the sent text.
   - **What the assistant reads**: the remaining risk is reader-missed wording changes, 3% of terse messages (4/120),
     2% of vague (2/120) and 0% of distracted (0/120). None of those flips a yes/no or invents a fact, but a
     blurred time or an extra "just say …" can steer the assistant. That is a trajectory risk, not a grading risk.
   - **Fixed-text fallback**: this rises with every overlay except terse. Fixed text is the least human-like reply, and
     even the control sends it 57% of the time. Any style mode should report its fallback rate per run.
   - **Separate finding**: the control's 48% rejection rate is itself a fidelity problem. Persona sign-offs and
     persona complaints written into `voice` get rejected and replaced by fixed text. That is worth its own fix.

## Limitations

- One writer sample per (style, pair). The writer is stochastic, so acceptance rates carry roughly ±9 points of
  binomial noise at n=120.
- Chatty ran on 58 pairs only, because of the budget.
- Contexts were placed heuristically. Some approve_reveal combinations are implausible pairings the code could still
  produce (e.g. an approval plus "show me what you'd send first").
- The new `decline` kind in the working tree ("no thanks, that's all for now") is not in the first run's corpus.
  The tightened-prompt run below adds 20 decline pairs.
- One person did the manual audit. Style adherence is measured only by proxies (length, lowercase, hedge words).
- Whether styled messages change assistant behaviour or scores was not measured.

## Tightened prompt

Follow-up run (`tight.py` → `raw_tight.jsonl`, `summary_tight.json`, `audit_tight.json`).

**What changed in the prompt.** The persona/voice framing is unchanged. The writer wrapper swaps only `_voice`'s
instruction sentences:

- Current: "Rewrite the given message in your voice. Keep exactly the same meaning. Do not add any fact, number, name,
  date, request, or approval."
- Tight: "Rewrite the given message in your voice. Keep exactly the same meaning and leave nothing out: if it says yes
  to an action, name that action; keep every name, number, address, email, time and date word for word. Do not add a
  greeting, sign-off, question, promise, instruction, condition, reason or opinion, and do not add any fact, number,
  name, date, request, or approval."

**What else is the same.** terse-tight and distracted-tight append the same overlays as before. The `_voice`
mechanics are unchanged: the real `SimulatedUser._voice`, the same writer model and settings, the Jev
`user_checks_rewording` check and 2 tries.

**Corpus.** The same 120 pairs, plus 20 `decline` pairs ("no thanks, that's all for now", `corpus_decline.json`).
Each decline pair is from a different task and sits after an assistant turn that ends by offering more help. In 6 of
the 20, the recorded user did decline.

**Calls.** 1,304 in total: 653 writer, 649 reader and 2 in a smoke test. There were no JUDGE calls; the audit is
manual, to the same standard as `audit.json`.

### Results on the same 120 pairs (earlier arm → tightened arm)

| arm | sent rewording | 1st try | fixed text actually sent | mean chars sent | mean chars attempts | meaning changes (manual) / accepted | paired change (lost / gained, exact McNemar) |
|---|---|---|---|---|---|---|---|
| none → **none-tight** | 62 (52%) → **80 (67%)** | 42 → 63 | 57% → 43% | 36.0 → 35.2 | 43.9 → 38.5 | 0/62 → **1/80** | 12 / 30, p = 0.008 |
| terse → **terse-tight** | 64 (53%) → **77 (64%)** | 54 → 62 | 50% → 43% | 29.7 → 34.3 | 27.0 → 32.3 | 4/64 → **3/77** | 11 / 24, p = 0.04 |
| distracted → **distracted-tight** | 48 (40%) → **68 (57%)** | 31 → 43 | 63% → 49% | 44.4 → 44.1 | 51.8 → 42.6 | 0/48 → **1/68** | 9 / 29, p = 0.002 |

Notes on the table:

- **Lowercase.** distracted-tight still carries its style: 91% of sent rewordings are all lowercase, against 98%
  before.
- **Empty writer replies.** 1, 2 and 1 for none-tight, terse-tight and distracted-tight, against 1, 0 and 10 before.

Acceptance by kind on the 120 pairs:

| arm | approve | approve_reveal | dont_know | refuse | reveal |
|---|---|---|---|---|---|
| none-tight | 14/27 (was 10) | 13/18 (11) | 22/27 (13) | 18/27 (18) | 13/21 (10) |
| terse-tight | 13/27 (was 3) | 13/18 (11) | 18/27 (20) | 16/27 (18) | 17/21 (12) |
| distracted-tight | 15/27 (was 8) | 16/18 (11) | 5/27 (3) | 20/27 (15) | 12/21 (11) |

Decline pairs (n = 20):

| arm | sent rewording | mean chars sent | meaning changes |
|---|---|---|---|
| none-tight | 8 (40%) | 26.6 | 0/8 |
| terse-tight | 7 (35%) | 26.1 | 0/7 |
| distracted-tight | 14 (70%) | 24.7 | 0/14 |

Meaning changes the reader accepted under the tight prompt (`audit_tight.json`). All are minor; none flips a yes or no
or alters a fact value:

| arm | fixed message | sent |
|---|---|---|
| none-tight | yes, go ahead and send a reply to a buyer. maria's 60 is fine by me, … | yeah send a reply to **the buyers** … (plural) |
| distracted-tight | (same pair) | yeah do that, **send the replies** … |
| terse-tight | You pick, anything nice under $60 is fine. | **Go ahead,** your pick. Anything nice under $60. (an approval word added to a fact) |
| terse-tight | … doug's telus address is fine | k, send from tomas@assistanteval.com. telus address fine (drops "doug's", as in the first run) |
| terse-tight | yes, go ahead and add a calendar event. leave kyle for now | yes add it, leave kyle (drops "for now") |

### What the tight rules fixed and what is left

**Fixed.**

- **Sign-offs.** Attempts with a sign-off fell from 27 of 197 to 9 of 176 in the control.
- **Approvals naming their action.** These improved most of all: terse approve went from 3/27 to 13/27.
- **"i don't know".** Under the control it rose from 13/27 to 22/27.

**Left.**

1. **Persona sign-offs still win sometimes.** Harold's "Regards, Harold" is the main one.
2. **The writer resolves referents from the conversation.** "pay with that Visa", "change to the 10:45", "send that
   reply to anita", "don't place the order" for "don't do that". The reader rightly rejects these.
3. **Reader under-confidence on short, faithful rewordings.** Rejected attempts scoring 0.5–0.69 rose from 23 to 33
   in the control arm.
   - For decline, "No thanks, that's all for now." (identical to the fixed text) scored 0.62 and 0.65. So did
     "no, that's all for now" (0.64/0.62), "all good, nothing else" (0.64) and "No, that's all for now." (0.62).
   - About 8 of the 12 rejected none-tight declines were faithful rewordings that the reader scored below 0.7.
   - This caps acceptance for short messages, and no writer prompt can fix it. A possible fix is code-side: accept
     when the normalised rewording equals the fixed text. The fallback sends the same text anyway, so this changes
     nothing about what is sent; it only stops counting such cases as rewrites. Calibrating the reader on short
     messages is the real lever, but neither was tested.

### Verdict

none-tight clearly beats the current prompt on acceptance: 67% vs 52% on the same pairs (30 gained, 12 lost,
p = 0.008). Fixed-text fallback drops from 57% to 43%.

On meaning, the manual audit found 1 slight change in 80 accepted rewordings, against 0 in 62. That one is a
plural referent ("the buyers") on an approval whose meaning already names both buyers. It does not change the decision, and
1 versus 0 is within noise. I treat this as no more meaning changes, but it is not zero, and the call is close.
terse-tight and distracted-tight also gain acceptance (+11 and +17 points). Their meaning-change counts are about the
same as before (terse 4/64 → 3/77, distracted 0/48 → 1/68).

The exact prompt is in `TIGHT_PROMPT.txt`, with `_voice`'s `{persona}` and `{voice}` placeholders. Overlays, if
used, are appended after one space.

**Updated recommendation.**

- Adopt the tight prompt as the base.
- Offer terse-tight and distracted-tight as the optional mode.
- terse still drops small qualifiers ("doug's", "for now") and once added "Go ahead" to a fact. If terse ships, add to
  its overlay: "keep every qualifier; never add yes or go ahead unless the message says it".
