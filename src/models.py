from typing import List
from pydantic import BaseModel, Field


class ColorVector(BaseModel):
    id: int
    title: str
    values: List[float] = Field(..., min_items=1)

    @classmethod
    def from_dict(cls, data: dict) -> "ColorVector":
        return cls(
            id=data["id"],
            title=data["title"],
            values=data["values"],
        )