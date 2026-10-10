# Simulated-user fidelity experiments

Tools for measuring how often the simulated user cannot answer, how often the
grader disputes its reading of an assistant turn, and whether a change helps.
They need the model roles set (see the README's "Two setups"); every
measurement here used the official setup.

## Data

The published runs on [assistanteval.com](https://assistanteval.com) are the
probe material: real conversations with nine assistants. They are downloaded
to `~/ae-exp-data/published/`, outside the repository: the data terms allow
reading and verifying them, not redistributing them as a dataset.

```bash
mkdir -p ~/ae-exp-data/published && cd ~/ae-exp-data
curl -s https://assistanteval.com/data/index.json > index.json
python3 -c "import json; print('\n'.join(r['id'] for r in json.load(open('index.json'))['runs']))" \
  | xargs -P 12 -I{} curl -s -f -o published/{}.json https://assistanteval.com/data/runs/{}.json
```

## Tools

- `probe.py`: replays the current simulated user on every point of a sampled
  conversation where the user replied or the assistant stopped, then asks the
  grader's re-read questions about each new decision. `sample` picks runs
  stratified by task (oversampling runs whose user said it did not know);
  `run` replays them. Judge answers are cached by question, so two arms
  differ only where their questions differ. `AE_CODE=<checkout>` runs another
  checkout's code (an earlier arm); `--overlay tasks/` adds user facts the
  published tasks lack.
- `gaps.py`: groups the "i don't know" decisions the grader agreed with by
  task, has the JUDGE propose preference facts, screens them, and rejects any
  that invent an email, URL or number. Proposals still need a person's review
  before they go into a task (see the results).
- `live/run.sh`: one live model-baseline run with a checkout's code, then live
  grading.
- `derived/`: the measurement of the grader's kept-length check.
- `styles/`: the writer style-overlay experiment.
