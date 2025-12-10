from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from app.core.ai_engine import analyze_schema
from app.utils.db_tools import load_schema_from_json
from app.models.decomposition import SchemaPayload

router = APIRouter()

@router.post("/")
def analyze_database_schema(payload: dict):
    # payload: { "tables": [...], "relations": [...] }
    normalized = load_schema_from_json(payload)
    try:
        schema = SchemaPayload.parse_obj(normalized)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    result = analyze_schema(schema.dict(by_alias=True))
    return {"analysis": result}
