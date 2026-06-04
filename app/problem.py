from dataclasses import dataclass
from typing import Callable


@dataclass
class Problem:
    name: str
    dimension: int
    function: Callable[[list[float]], float]
    gradient: Callable[[list[float]], list[float]]
    description: str