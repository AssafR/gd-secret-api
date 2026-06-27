import math
from collections.abc import Sequence

import numpy as np
from app.datasets import RIDGE_100D_X, RIDGE_100D_Y, QUAD_X, QUAD_Y


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


# ---- Quadratic binary classification problem ---------------------------------
def _quadratic_design_matrix(X: np.ndarray) -> np.ndarray:
    """
    Build design matrix for quadratic basis:
    features: [1, x, y, x^2, y^2, x*y]
    X: (n_samples, 2)
    returns: (n_samples, 6)
    """
    x = X[:, 0]
    y = X[:, 1]
    ones = np.ones_like(x)
    return np.column_stack([ones, x, y, x ** 2, y ** 2, x * y])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def quadratic_binary_loss(params: Sequence[float], alpha: float = 0.0) -> float:
    """
    Binary cross-entropy loss for quadratic classifier.
    params: [b, w1, w2, w11, w22, w12]
    """
    p = _to_array(params)
    if p.size != 6:
        raise ValueError("Expected 6 parameters for quadratic classifier.")
    Phi = _quadratic_design_matrix(QUAD_X)  # (n,6)
    logits = Phi @ p
    preds = _sigmoid(logits)
    eps = 1e-12
    preds = np.clip(preds, eps, 1.0 - eps)
    y = QUAD_Y
    loss = -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))
    if alpha and alpha > 0.0:
        # do not regularize bias term
        loss += 0.5 * alpha * float(np.sum(p[1:] ** 2))
    return float(loss)


def quadratic_binary_grad(params: Sequence[float], alpha: float = 0.0) -> list[float]:
    """
    Analytical gradient of the binary cross-entropy loss w.r.t params.
    Returns a list of 6 floats.
    """
    p = _to_array(params)
    if p.size != 6:
        raise ValueError("Expected 6 parameters for quadratic classifier.")
    Phi = _quadratic_design_matrix(QUAD_X)  # (n,6)
    logits = Phi @ p
    preds = _sigmoid(logits)
    errors = preds - QUAD_Y  # (n,)
    grad = (Phi.T @ errors) / len(QUAD_Y)
    if alpha and alpha > 0.0:
        reg = np.concatenate([[0.0], p[1:]]) * alpha
        grad = grad + reg
    return grad.tolist()


def quadratic_predict_proba(params: Sequence[float], X: np.ndarray | None = None) -> np.ndarray:
    """
    Return predicted probabilities for given params on provided X.
    If X is None uses the hidden dataset QUAD_X.
    """
    p = _to_array(params)
    if p.size != 6:
        raise ValueError("Expected 6 parameters for quadratic classifier.")
    if X is None:
        X = QUAD_X
    Phi = _quadratic_design_matrix(np.asarray(X))
    logits = Phi @ p
    return _sigmoid(logits)