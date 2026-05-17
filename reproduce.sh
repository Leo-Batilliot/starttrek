#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then PY="python3"; fi

SEEDS="${SEEDS:-0 1 2 3 4}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
export OMP_NUM_THREADS="${JOBS_THREADS:-3}"
export MKL_NUM_THREADS="${JOBS_THREADS:-3}"

CONFIGS=(
  "configs/baseline/dqn.yaml:dqn_baseline"
  "configs/ablations/dqn_buffer10k.yaml:dqn_buffer10k"
  "configs/ablations/dqn_depth3.yaml:dqn_depth3"
)

echo "======================================================================"
echo "  reproduce.sh (T3.5)"
echo "  python : $PY"
echo "  seeds  : $SEEDS    skip_existing=$SKIP_EXISTING    threads=$OMP_NUM_THREADS"
echo "======================================================================"

for entry in "${CONFIGS[@]}"; do
  cfg="${entry%%:*}"
  run="${entry##*:}"
  echo ""
  echo "--- training '$run' ($cfg) — seeds: $SEEDS ---"
  mkdir -p "runs/$run"
  pids=()
  for s in $SEEDS; do
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "runs/$run/$s/summary.json" ]; then
      echo "  seed $s — summary.json exists, skipping (SKIP_EXISTING=1)"
      continue
    fi
    echo "  seed $s — launching (log: runs/$run/$s.repro.log)"
    "$PY" train.py --config "$cfg" --seed "$s" \
      > "runs/$run/$s.repro.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]:-}"; do
    [ -n "$p" ] && wait "$p"
  done
  # Non-fatal: a failed solved-criterion must not abort artifact regen.
  "$PY" scripts/aggregate_runs.py --base "runs/$run" --seeds $SEEDS \
    || echo "  WARN: '$run' did not meet the >=4/5 solved criterion (continuing)"
done

echo ""
echo "--- regenerating plots ---"
"$PY" scripts/make_plots.py --run runs/dqn_baseline/0 --out-dir assets/plots
"$PY" scripts/aggregated_plots.py --seeds $SEEDS

echo ""
echo "======================================================================"
echo "  reproduce.sh — done."
echo "  runs/{dqn_baseline,dqn_buffer10k,dqn_depth3}/  + summary_aggregated"
echo "  assets/plots/  (5 single-run + 3 aggregated + 2 paired)"
echo "======================================================================"
