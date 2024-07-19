#!/usr/bin/env bash
set -xe
cmd=(
  python3.11 ./plot_generators/characteristic_curve.py
  --N=10
  --num-initial-types=1
  --mutation-rate=0.01
  --max-steps=10000
  --graph-generators='complete'
  --num-simulations=100
  --num-workers=32
  --chunksize=1
  --sample-rate=1.1
  --characteristic-curve-data-file 'data/characteristic-curve.pkl'
  # --characteristic-curve-data-file 'data/characteristic-curve-2024-07-16::16:50:41.647394.pkl'
  --timestamp-str "$(date +'%Y-%m-%d::%H:%M:%S')"
  --use-timestamp
  --stat-to-calculate='num_types_left'
  --overwrite
  # --use-existing-data
  # --draw
)
"${cmd[@]}" 