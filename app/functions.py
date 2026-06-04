import math


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