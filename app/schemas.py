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


class ProblemMetadata(BaseModel):
    name: str
    dimension: int
    description: str
    