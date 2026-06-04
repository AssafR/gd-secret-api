import math
import numpy as np
from app.datasets import RIDGE_100D_X, RIDGE_100D_Y


def parabola_1d(x: list[float]) -> float:
    return (x[0] - 3) ** 2


def parabola_1d_grad(x: list[float]) -> list[float]:
    return [2 * (x[0] - 3)]


def wavy_1d(x: list[float]) -> float:
    return (x[0] - 3) ** 2 + 2 * math.sin(3 * x[0])


def wavy_1d_grad(x: list[float]) -> list[float]:
    return [2 * (x[0] - 3) + 6 * math.cos(3 * x[0])]


def bowl_2d(x: list[float]) -> float:
    return (x[0] - 2) ** 2 + (x[1] + 1) ** 2


def bowl_2d_grad(x: list[float]) -> list[float]:
    return [
        2 * (x[0] - 2),
        2 * (x[1] + 1),
    ]



def ridge_loss_100d(w: list[float], alpha: float = 0.1) -> float:
    w_arr = np.array(w)

    errors = RIDGE_100D_X @ w_arr - RIDGE_100D_Y

    mse = np.mean(errors ** 2)
    penalty = alpha * np.sum(w_arr ** 2)

    return float(mse + penalty)


def ridge_loss_100d_grad(w: list[float], alpha: float = 0.1) -> list[float]:
    w_arr = np.array(w)

    errors = RIDGE_100D_X @ w_arr - RIDGE_100D_Y

    grad_mse = (2 / len(RIDGE_100D_Y)) * (RIDGE_100D_X.T @ errors)
    grad_penalty = 2 * alpha * w_arr

    return (grad_mse + grad_penalty).tolist()