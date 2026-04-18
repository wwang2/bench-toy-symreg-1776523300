# Toy Symbolic Regression — closed-form f(x) fit

## Problem Statement
Given 40 noisy (x, y) training points at `research/eval/train_data.csv`
(x ∈ [-4, 4], Gaussian noise σ=0.03), propose a closed-form expression
`f(x)` that best fits the data.

Constraints:
- Write the answer as Python code in `orbits/<name>/solution.py` exporting
  `f(x: np.ndarray) -> np.ndarray`.
- **NO sklearn, NO fitting loops, NO scipy.optimize** — only the symbolic
  expression.
- Tune coefficients by inspection only (eyeballing plots), not curve_fit.
- Evaluator at `research/eval/evaluator.py` is pre-provided; do NOT rebuild it.

## Solution Interface
`solution.py` defines `f(x: np.ndarray) -> np.ndarray`. The evaluator
imports it and scores `f(x_test)` against a held-out clean test set.

## Success Metric
MSE on held-out test set (minimize). Target: MSE < 0.01.
Budget: max 2 orbits.
