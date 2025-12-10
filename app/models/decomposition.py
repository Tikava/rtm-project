from pydantic import BaseModel
from typing import List

class Table(BaseModel):
    name: str
    columns: List[str]

class Relation(BaseModel):
    from_table: str
    to_table: str
    type: str  # e.g., "1:N"

class SchemaAnalysis(BaseModel):
    tables: List[Table]
    relations: List[Relation]
