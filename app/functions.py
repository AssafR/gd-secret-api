import math
from collections.abc import Sequence

import numpy as np

from app.datasets import RIDGE_100D_X, RIDGE_100D_Y


def _to_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def parabola_1d(x: Sequence[float]) -> float:
    return float((x[0] - 3.0) ** 2)


def parabola_1d_grad(x: Sequence[float]) -> list[float]:
    return [float(2.0 * (x[0] - 3.0))]


def wavy_1d(x: Sequence[float]) -> float:
    return float((x[0] - 3.0) ** 2 + 2.0 * math.sin(3.0 * x[0]))


def wavy_1d_grad(x: Sequence[float]) -> list[float]:
    return [float(2.0 * (x[0] - 3.0) + 6.0 * math.cos(3.0 * x[0]))]


def bowl_2d(x: Sequence[float]) -> float:
    x_arr = _to_array(x)
    return float((x_arr[0] - 2.0) ** 2 + (x_arr[1] + 1.0) ** 2)


def bowl_2d_grad(x: Sequence[float]) -> list[float]:
    x_arr = _to_array(x)
    return [float(2.0 * (x_arr[0] - 2.0)), float(2.0 * (x_arr[1] + 1.0))]


def stretched_bowl_2d(x: Sequence[float]) -> float:
    x_arr = _to_array(x)
    return float(10.0 * (x_arr[0] - 2.0) ** 2 + (x_arr[1] + 1.0) ** 2)


def stretched_bowl_2d_grad(x: Sequence[float]) -> list[float]:
    x_arr = _to_array(x)
    return [float(20.0 * (x_arr[0] - 2.0)), float(2.0 * (x_arr[1] + 1.0))]


def rosenbrock_2d(x: Sequence[float]) -> float:
    x_arr = _to_array(x)
    return float((1.0 - x_arr[0]) ** 2 + 100.0 * (x_arr[1] - x_arr[0] ** 2) ** 2)


def rosenbrock_2d_grad(x: Sequence[float]) -> list[float]:
    x_arr = _to_array(x)
    dx = -2.0 * (1.0 - x_arr[0]) - 400.0 * x_arr[0] * (x_arr[1] - x_arr[0] ** 2)
    dy = 200.0 * (x_arr[1] - x_arr[0] ** 2)
    return [float(dx), float(dy)]


def ridge_loss_100d(w: Sequence[float], alpha: float = 0.1) -> float:
    w_arr = _to_array(w)
    errors = RIDGE_100D_X @ w_arr - RIDGE_100D_Y
    mse = np.mean(errors ** 2)
    penalty = alpha * np.sum(w_arr ** 2)
    return float(mse + penalty)


def ridge_loss_100d_grad(w: Sequence[float], alpha: float = 0.1) -> list[float]:
    w_arr = _to_array(w)
    errors = RIDGE_100D_X @ w_arr - RIDGE_100D_Y
    grad_mse = (2.0 / len(RIDGE_100D_Y)) * (RIDGE_100D_X.T @ errors)
    grad_penalty = 2.0 * alpha * w_arr
    return (grad_mse + grad_penalty).tolist()