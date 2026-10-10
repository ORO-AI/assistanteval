#!/bin/bash
# run.sh TREE ARM MODEL TASK REP: one live model-baseline run with TREE's code, then live grading.
# Writes ~/ae-exp-data/live/<ARM>/<model>/<task>-<rep>/{run,grade}. Needs roles-official.sh sourced.
set -u
TREE=$1; ARM=$2; MODEL=$3; TASK=$4; REP=$5
OUT=~/ae-exp-data/live/$ARM/${MODEL//\//_}/$TASK-$REP
mkdir -p "$OUT"
cd "$TREE" || exit 1
ASSISTANTEVAL_BASELINE_MODEL=$MODEL timeout 1500 uv run --frozen assistanteval run "tasks/$TASK.json" --assistant model_baseline --out "$OUT/run" > "$OUT/run.log" 2>&1
timeout 900 uv run --frozen assistanteval grade "$OUT/run" --answers "$OUT/answers.json" --out "$OUT/grade" > "$OUT/grade.log" 2>&1
echo "$ARM $MODEL $TASK $REP done"
