from fastapi import FastAPI
from app.schemas import EvaluateRequest
from app.registry import list_problem_metadata, get_problem, validate_dimension

app = FastAPI(title="Gradient Descent Secret Function API")


@app.get("/")
def root():
    return {"message": "Gradient Descent API", "try": "/docs"}


@app.get("/problems")
def problems():
    return list_problem_metadata()


@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    problem = get_problem(request.problem_id)
    validate_dimension(problem, request.x)

    return {
        "problem_id": request.problem_id,
        "x": request.x,
        "y": problem.function(request.x),
    }


@app.post("/evaluate-with-gradient")
def evaluate_with_gradient(request: EvaluateRequest):
    problem = get_problem(request.problem_id)
    validate_dimension(problem, request.x)

    return {
        "problem_id": request.problem_id,
        "x": request.x,
        "y": problem.function(request.x),
        "gradient": problem.gradient(request.x),
    }