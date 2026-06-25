from fastapi import APIRouter

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
    ProblemMetadata,
)

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
