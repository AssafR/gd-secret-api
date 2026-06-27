from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    problem_id: str
    x: list[float]


class EvaluateResponse(BaseModel):
    problem_id: str
    x: list[float]
    y: float


class EvaluateWithGradientResponse(EvaluateResponse):
    gradient: list[float]

class EvaluateVisualizationResponse(EvaluateWithGradientResponse):
    accuracy: float | None = None
    # list of points [[x,y], ...] for plotting
    points: list[list[float]] | None = None
    # the coefficients that were evaluated (same as request.x)
    coefficients: list[float] | None = None
class ProblemMetadata(BaseModel):
    name: str
    dimension: int
    description: str
    