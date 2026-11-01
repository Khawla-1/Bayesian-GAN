#!/usr/bin/env bash
# Regenerates every table and figure of the manuscript from the raw data.
# Approximate CPU time on a 2-core machine: rolling evaluation ~2 h, final stage ~30 min,
# foundation models ~30 min (needs huggingface.co / hf.co).
set -euo pipefail
cd "$(dirname "$0")"
CAMS=../data/RjimMaatoug.csv
MET=../data/dataset1000.xlsx
python run_experiments.py --cams $CAMS --met $MET --out results            # 8 origins x 3 seeds
python run_foundation.py  --cams $CAMS --met $MET --out results || echo "foundation models skipped"
python run_final.py       --cams $CAMS --met $MET --out results            # MC size, Sobol, projection
python analysis.py        --cams $CAMS --met $MET --out results            # tables/figures/summary.json
python make_numbers.py results ../manuscript/numbers.tex                    # numbers quoted in the text
