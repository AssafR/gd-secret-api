from fastapi import FastAPI
from pydantic import BaseModel
import math

app = FastAPI(title="Gradient Descent Secret Function API")


class XInput(BaseModel):
    x: float


def secret_function(x: float) -> float:
    return (x - 3) ** 2 + 2 * math.sin(3 * x)


def secret_gradient(x: float) -> float:
    return 2 * (x - 3) + 6 * math.cos(3 * x)


@app.post("/evaluate")
def evaluate(data: XInput):
    return {
        "x": data.x,
        "y": secret_function(data.x),
    }


@app.post("/evaluate-with-gradient")
def evaluate_with_gradient(data: XInput):
    return {
        "x": data.x,
        "y": secret_function(data.x),
        "gradient": secret_gradient(data.x),
    }