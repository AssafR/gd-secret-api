from fastapi import APIRouter
import numpy as np

from app.registry import (
    get_problem,
    list_problem_metadata,
    validate_dimension,
    validate_gradient_available,
)
from app.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    EvaluateWithGradientResponse,
    EvaluateVisualizationResponse,
    ProblemMetadata,
)
from app.datasets import get_dataset
from app.functions import quadratic_predict_proba

router = APIRouter()


@router.get("/", tags=["health"])
def root():
    return {"message": "Gradient Descent API", "try": "/docs"}


@router.get("/problems", response_model=dict[str, ProblemMetadata], tags=["discovery"])
def problems():
    return list_problem_metadata()


@router.post("/evaluate", response_model=EvaluateResponse, tags=["evaluation"])
def evaluate(request: EvaluateRequest):
    problem = get_problem(request.problem_id)
    validate_dimension(problem, request.x)
    return {
        "problem_id": request.problem_id,
        "x": request.x,
        "y": problem.function(request.x),
    }


@router.post(
    "/evaluate-with-gradient",
    response_model=EvaluateWithGradientResponse,
    tags=["evaluation"],
)
def evaluate_with_gradient(request: EvaluateRequest):
    problem = get_problem(request.problem_id)
    validate_dimension(problem, request.x)
    validate_gradient_available(problem)
    return {
        "problem_id": request.problem_id,
        "x": request.x,
        "y": problem.function(request.x),
        "gradient": problem.gradient(request.x),
    }


@router.post(
    "/evaluate-with-visualization",
    response_model=EvaluateVisualizationResponse,
    tags=["evaluation"],
)
def evaluate_with_visualization(request: EvaluateRequest):
    """
    Return loss, gradient, and dataset/accuracy information for visualization clients.
    Works for dataset-backed problems (e.g. 'quadratic_binary').
    """
    problem = get_problem(request.problem_id)
    validate_dimension(problem, request.x)
    validate_gradient_available(problem)

    y = problem.function(request.x)
    grad = problem.gradient(request.x)

    points = None
    accuracy = None
    try:
        X, y_true = get_dataset(request.problem_id)
        # compute probabilities using the same internal helper
        probs = quadratic_predict_proba(request.x, X)
        preds = (probs >= 0.5).astype(int)
        accuracy = float(np.mean(preds == y_true))
        points = X.tolist()
    except KeyError:
        # dataset not available for this problem_id; that's fine
        points = None
        accuracy = None

    return {
        "problem_id": request.problem_id,
        "x": request.x,
        "y": y,
        "gradient": grad,
        "accuracy": accuracy,
        "points": points,
        "coefficients": request.x,
    }
