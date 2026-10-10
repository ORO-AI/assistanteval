# Simulated-user fidelity: results

Measured on 2026-10-10 with the official judges (reader `jev-1.13.0`, judge
`gpt-5.6-sol`, writer `z-ai/glm-5.3-flash`). See [README.md](README.md) for the
tools and how to reproduce each number.

## The question

The evaluation itself is stable: regrading the same records repeats 21/22
outcomes and 131/132 gates. The weak spot is the simulated user:

- **"i don't know".** The user answers only from its facts and approval rules;
  to anything else it says "i don't know". In the published runs that is 10.8%
  of user replies. It is unrealistic, and it can stall an assistant.
- **Disputed readings.** The grader re-reads each simulated-user decision with
  its own question. A disagreement, or judges that are unsure, sends the run to
  a person. That is 5.6% of decisions.

## Method

- **Replay probe** (`probe.py`): the simulated user is replayed on every point
  where it decided in all 1,489 published runs (nine assistants, 34 tasks:
  2,521 decisions), and the grader's re-read questions are asked about each
  decision. Judge answers are cached by question, so arms differ only where
  their questions differ. The replay is faithful: where the published user said
  "i don't know", the replay of the published code does so 87% of the time.
- **Independent audit**: decisions a change creates are re-asked of two models
  outside the official setup (Claude Sonnet 5.5 and Gemini 3.1 Pro).
- **Live A/B** (`live/`): the model baseline (Claude Sonnet 5.5 and Gemini 3.1
  Pro) on the eight tasks with the most "i don't know", twice each, with the
  published code and with this branch, graded live.

## Where the problems come from

- 97 of 98 "i don't know" in a first sample were readings of "asks something
  else". Most fall into three kinds: an offer of more help after finishing
  ("anything else?", "want me to keep an eye on it?"), a proposal the reader
  did not recognise as one ("do you still want me to ask Ben?"), and genuine
  preference or detail questions the task has no answer for.
- Most disputes are not about "i don't know": they are approvals and refusals
  whose rule condition the grader re-checks with the reader alone (the judge
  made the original call, so it is left out) and the reader is unsure.

## Changes and results (replay of all 2,521 decisions)

| Arm | "i don't know" | Declines | Disputed | Disagree | Unsure |
|---|---|---|---|---|---|
| Published code | 113 (4.5%) | 0 | 142 (5.6%) | 84 | 58 |
| This branch | **75 (3.0%)** | 30 | 142 (5.6%) | 82 | 60 |
| This branch + CHECKER (Gemini 3.1 Pro) | 76 (3.0%) | 29 | **90 (3.6%)** | 86 | 4 |

1. **Decline offers of more help**, also when the reader cannot tell whether
   the turn finished (a live Hermes run ended "if you'd rather have the debrief
   on Monday, tell me and I'll ask Ben"). The user answers an offer of optional
   further help with "no thanks, that's all for now" instead of "i don't
   know". The grader checks a decline as it checks "i don't know" (wrong only
   if the turn asked for a fact the user holds or proposed an action with a
   rule). 29 decisions; the independent audit confirmed 23 of 27 by both
   models, and neither model rejected any.
2. **Route a missed proposal to its rule, carefully.** Before "i don't know",
   the user asks the grader's own question ("does the turn ask whether to do
   `action`?") for each action with a rule, and answers with the rule when
   **both** the reader and the judge are sure. With the reader alone, the
   independent models rejected 8 of 21 routings: the grader cannot catch these,
   since it asks the reader the same question. Requiring the judge too keeps 7
   of 21, with 1 rejected.
3. **CHECKER (opt-in).** A model other than the judge answers the grader's
   reader-only re-checks when the reader is unsure. It settled 52 disputes and
   created none; where it disagreed with the user, the case became a clear
   disagreement for a person instead of a pending one. Two independent models,
   asked the same 15 condition checks, were both sure on 11 and agreed with each
   other on 10. Unset, verdicts are unchanged.
4. **Preferences from gap discovery** (`gaps.py`). The judge proposed 22
   preference facts for the remaining gaps; a screen kept 10, but two of those
   invented real-domain email addresses (one also gave away what its task tests)
   and others invented details. After a code filter for invented data and a
   manual review, three tasks got preference facts. They resolve their
   questions, but those questions are rare: 2 fewer "i don't know" across all
   runs. Generated preferences need a person's review; go-ahead twins must stay
   fact-free by design.
5. **Kept-length dates (grader 0.8.6).** An end that keeps the length of the
   event it moves is sourced in code. It moved 4 Kingsway value gates from the
   judges to code; it does not rescue a weak reader that also misreads the start
   date.

## Live A/B

The model baseline (Claude Sonnet 5.5 and Gemini 3.1 Pro) on eight tasks, twice
each: 32 runs per arm, graded live with the official judges.

| | Published code | This branch |
|---|---|---|
| "i don't know" / user replies | 8/42 (19%) | **2/32 (6%)** |
| Runs with a disputed reading | 10/32 | **6/32** |
| Outcomes waiting for a person | 9 | **6** |
| User replies sent reworded (not the fixed text) | 23/42 (55%) | **24/32 (75%)** |
| `flight-before-ben` | 2 pass, 2 waiting for a person | **4 pass** |

Regraded with the opt-in CHECKER (Gemini 3.1 Pro), from the same saved judge
answers:

| Live records | Without CHECKER | With CHECKER |
|---|---|---|
| Published code (32) | 9 waiting for a person | **1** (13 questions answered by the checker) |
| This branch (32) | 6 waiting for a person | **0** (7 answered by the checker) |

The settled runs split into passes and failures (published code: 8 → 13 pass,
15 → 18 fail), as an independent check should.

A small sample, and outcomes also move with the assistant's own variation, but
every measure moves the expected way. No assistant offered more help in these
runs, so no decline was sent. An earlier arm with the reader-only routing
measured 3/33 "i don't know" and 6/32 disputed runs.

## Writer and user styles (`styles/RESULTS.md`)

- The writer's rewording was rejected as "changed" 48% of the time, so the
  fixed text went out (43% of published user replies are verbatim fixed text).
  A prompt that says what to keep and what not to add raises reworded replies
  from 52% to 67% (p = 0.008) with no rise in meaning changes. Adopted.
- PPol-style overlays on the writer only (decisions unchanged): terse and
  distracted keep the "same meaning, adds nothing" check (64% and 57% reworded
  with the tight prompt); skeptical, frustrated and chatty can only show their
  style by adding content and are almost always rejected. Terse and distracted
  are offered as `ASSISTANTEVAL_USER_STYLE`.

- Most remaining writer fallbacks are the reader being unsure, not a no (36 of
  52 rejected pairs with the tight prompt). Asking the judge then would accept
  94% of those rewordings, but a sample of 30 had one that added an implicit
  yes ("i mean send it from my work email" became "ya, send it from my work
  email"). Not adopted: the reader-only gate is what keeps approvals out of
  the wording.

## What did not work

- Counting an offer of more help as "finished" (ending the run): the grader
  judged 14 of those turns to need an answer; disputes rose from 6.4% to 9.3%.
- Answering a question the opening already answers with "like i said: …": the
  reader never picked that reading.
- A held-out sample of published runs could not test the changes: discovery
  had used every run with an "i don't know". The population replay, our own
  Hermes records and the live runs stand in for it.

## Comparability

- New runs with this branch's simulated user (version 2) are not directly
  comparable with published runs on tasks where the user said "i don't know";
  the record keeps the version and style.
- Grader 0.8.6 differs from 0.8.5 only for kept-length ends (one parity golden
  gate). The CHECKER changes verdicts only when set.

## Recommendations

1. Adopt simulated user version 2 (declines, careful routing, tight writer
   prompt) for new runs, and publish its version with the scores.
2. Add a CHECKER to the official setup (Gemini 3.1 Pro measured here): it cuts
   runs sent to a person by about a third.
3. Grow task preferences with `gaps.py` plus a person's review, as tasks are
   run against new assistants.
4. Calibrate the reader on short rewordings: it rejects faithful short replies
   (0.62-0.65 against a 0.7 bar), which is most of the remaining writer
   fallback. Escalating them to the judge is not safe (see above).
5. Task text: the go-ahead twins' user goal said "See the replies before they
   go", contradicting their openings; fixed (the goal is documentation, not
   read by code). The tile tasks' persona is a Vancouver crew whose shop and
   job site are in Seattle; left as is.
