#!/usr/bin/env bash
# Reproduce orbit 01-sin-plus-quad from the repo root.
# Hypothesis: f(x) = sin(x) + 0.1 * x**2 (eyeballed — no fitting).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# Run the deterministic evaluator over 3 seeds (all identical by design).
for SEED in 1 2 3; do
  uv run --with numpy python3 research/eval/evaluator.py \
    --solution orbits/01-sin-plus-quad/solution.py --seed "$SEED"
done

# Regenerate figures.
uv run --with numpy --with matplotlib python3 \
  orbits/01-sin-plus-quad/make_figures.py
