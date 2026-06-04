from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    problem_id: str
    x: list[float]
    