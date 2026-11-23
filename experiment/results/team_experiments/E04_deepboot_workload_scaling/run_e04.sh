#!/usr/bin/env bash
set +e

OUT_DIR="experiment/results/team_experiments/E04_deepboot_workload_scaling"
if [[ -e "$OUT_DIR/simulator-output" ]]; then
  echo "Refusing to overwrite existing simulator output: $OUT_DIR/simulator-output" >&2
  exit 2
fi
date --iso-8601=seconds > "$OUT_DIR/start_time.txt"
/usr/bin/time -v -o "$OUT_DIR/runtime.txt" \
  conda run --no-capture-output -n pollux python simulator.py \
    --workload ./experiment/workload/middle.csv \
    --AFE 0 --AISS 1 --log_file 0 \
    --output "$OUT_DIR/simulator-output" \
    --min-nodes 16 --num-gpus 4 --policy dp \
    > "$OUT_DIR/execution.log" 2>&1
STATUS=$?
date --iso-8601=seconds > "$OUT_DIR/end_time.txt"
printf 'exit_status=%s\n' "$STATUS" > "$OUT_DIR/exit_status.txt"
exit "$STATUS"
