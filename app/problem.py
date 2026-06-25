from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Optional


@dataclass(frozen=True)
class Problem:
    name: str
    dimension: int
    function: Callable[[Sequence[float]], float]
    gradient: Optional[Callable[[Sequence[float]], list[float]]]
    description: str

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "description": self.description,
        }