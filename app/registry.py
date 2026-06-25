from fastapi import HTTPException

from app.functions import (
    bowl_2d,
    bowl_2d_grad,
    parabola_1d,
    parabola_1d_grad,
    ridge_loss_100d,
    ridge_loss_100d_grad,
    rosenbrock_2d,
    rosenbrock_2d_grad,
    stretched_bowl_2d,
    stretched_bowl_2d_grad,
    wavy_1d,
    wavy_1d_grad,
)
from app.problem import Problem


PROBLEMS: dict[str, Problem] = {
    "parabola_1d": Problem(
        name="Simple 1D parabola",
        dimension=1,
        function=parabola_1d,
        gradient=parabola_1d_grad,
        description="Easy convex function. Good first gradient descent example.",
    ),
    "wavy_1d": Problem(
        name="Wavy 1D function",
        dimension=1,
        function=wavy_1d,
        gradient=wavy_1d_grad,
        description="Has local wiggles. Good for showing local minima and learning-rate issues.",
    ),
    "bowl_2d": Problem(
        name="Simple 2D bowl",
        dimension=2,
        function=bowl_2d,
        gradient=bowl_2d_grad,
        description="Easy 2D convex function. Good for visualizing movement on a surface.",
    ),
    "stretched_bowl_2d": Problem(
        name="Stretched 2D bowl",
        dimension=2,
        function=stretched_bowl_2d,
        gradient=stretched_bowl_2d_grad,
        description="A narrow, ill-conditioned bowl. Good for showing scaling and step size sensitivity.",
    ),
    "rosenbrock_2d": Problem(
        name="Rosenbrock function",
        dimension=2,
        function=rosenbrock_2d,
        gradient=rosenbrock_2d_grad,
        description="Classic optimization benchmark with a shallow valley and difficult curvature.",
    ),
    "ridge_100d": Problem(
        name="Hidden 100D ridge regression loss",
        dimension=100,
        function=ridge_loss_100d,
        gradient=ridge_loss_100d_grad,
        description="A higher-dimensional hidden loss showing why analytical gradients scale better.",
    ),
}


def get_problem(problem_id: str) -> Problem:
    if problem_id not in PROBLEMS:
        raise HTTPException(status_code=404, detail=f"Unknown problem_id: {problem_id}")
    return PROBLEMS[problem_id]


def validate_dimension(problem: Problem, x: list[float]) -> None:
    if len(x) != problem.dimension:
        raise HTTPException(
            status_code=400,
            detail=f"Expected x of length {problem.dimension}, got {len(x)}.",
        )


def validate_gradient_available(problem: Problem) -> None:
    if problem.gradient is None:
        raise HTTPException(
            status_code=400,
            detail=f"Gradient is not available for problem_id: {problem.name}",
        )


def list_problem_metadata():
    return {
        problem_id: {
            "name": problem.name,
            "dimension": problem.dimension,
            "description": problem.description,
        }
        for problem_id, problem in PROBLEMS.items()
    }