from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any


class Table(BaseModel):
    name: str
    columns: List[str]

    @validator("name")
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Table name is required")
        return v


class Relation(BaseModel):
    source: str = Field(..., alias="from")
    target: str = Field(..., alias="to")
    type: Optional[str] = ""

    @validator("source", "target")
    def endpoints_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Relation endpoint is required")
        return v

    class Config:
        allow_population_by_field_name = True


class UsagePair(BaseModel):
    tables: List[str]
    count: int = 0

    @validator("tables")
    def two_tables(cls, v: List[str]) -> List[str]:
        if len(v) != 2:
            raise ValueError("Usage entry must contain exactly two tables")
        if any(not t or not t.strip() for t in v):
            raise ValueError("Usage tables must be non-empty")
        return v


class SchemaPayload(BaseModel):
    tables: List[Table] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    usage: List[UsagePair] = Field(default_factory=list)

    @validator("tables")
    def no_duplicate_tables(cls, v: List[Table]) -> List[Table]:
        names = [t.name for t in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate table names are not allowed")
        return v


class SuggestionsRequest(BaseModel):
    analysis: Dict[str, Any]
