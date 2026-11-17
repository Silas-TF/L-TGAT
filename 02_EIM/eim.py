"""Exponential-decay Bass model adapted to observed topic interaction windows."""

import numpy as np
from scipy.optimize import least_squares

from cascade_inputs import hourly_actions
from protocol import HORIZON


def prepare(topic, windows, options):
    return np.asarray([
        ((w.time - topic.start) / HORIZON, hourly_actions(topic, w.time).sum() / 3)
        for w in windows
    ], dtype=float)


def forecast(parameters, samples):
    external, internal, decay, capacity = parameters
    age, recent = samples.T
    # Effective capacity and initial state are expressed on the paper's label scale.
    state = np.minimum(recent, capacity)
    initial = state.copy()
    steps = 24
    for step in range(steps):
        time = age + (step + 0.5) / steps
        hazard = external * np.exp(-decay * time) + internal * state / capacity
        state += (capacity - state) * -np.expm1(-hazard / steps)
    return state - initial


def fit(train, targets, validation, validation_targets, seeds):
    """Fit on train only and compare two starting points per seed on validation."""
    scale = max(1.0, float(train[:, 1].max()), float(np.max(targets)))
    bounds = ([0, 0, 0, scale * 1.001], [20, 20, 20, scale * 1000])
    for seed in seeds:
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(2):
            initial = [rng.uniform(.005, .3), rng.uniform(.005, .3),
                       rng.uniform(.001, .15), scale * rng.uniform(1.5, 6)]
            fit_result = least_squares(
                lambda params: (forecast(params, train) - targets) / scale,
                initial, bounds=bounds, max_nfev=200,
            )
            score = np.mean((forecast(fit_result.x, validation) - validation_targets) ** 2)
            if best is None or score < best[0]:
                best = score, fit_result
        yield seed, best[1].x, {"validation_mse": float(best[0]),
                                "optimizer_converged": bool(best[1].success)}
