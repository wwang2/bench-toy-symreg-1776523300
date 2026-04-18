#!/usr/bin/env python3
"""
Generate data for the toy symbolic-regression benchmark.

Target function (HIDDEN from agents):
  f(x) = sin(x) + 0.1 * x^2

Chosen to be solvable by inspection on a pencil-and-paper scan:
  - sinusoidal + mild upward quadratic
  - obvious parity (symmetric parabolic envelope, odd sinusoid)
  - no fitting required — an agent can propose the closed form after
    eyeballing the data and hit target MSE on the first try.
"""

import numpy as np


def target_function(x):
    return np.sin(x) + 0.1 * x**2


def generate_train_data(n_points=40, noise_sigma=0.03, seed=42):
    rng = np.random.RandomState(seed)
    x = np.linspace(-4, 4, n_points)
    y = target_function(x) + rng.normal(0, noise_sigma, n_points)
    return x, y


def generate_test_data(n_points=400, seed=99):
    x = np.linspace(-4, 4, n_points)
    y = target_function(x)
    return x, y


if __name__ == "__main__":
    x_train, y_train = generate_train_data()
    np.savetxt(
        "train_data.csv",
        np.column_stack([x_train, y_train]),
        delimiter=",",
        header="x,y",
        comments="",
    )
    print(f"Training data: {len(x_train)} points, noise=0.03")
    print(f"  x range: [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f"  y range: [{y_train.min():.3f}, {y_train.max():.3f}]")

    x_test, y_test = generate_test_data()
    np.savetxt(
        "test_data.csv",
        np.column_stack([x_test, y_test]),
        delimiter=",",
        header="x,y",
        comments="",
    )
    print(f"Test data: {len(x_test)} points (clean, for evaluation)")
