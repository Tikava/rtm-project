from fastapi import APIRouter
from app.core.ai_engine import analyze_schema
from app.utils.db_tools import load_schema_from_json

router = APIRouter()

@router.post("/")
def analyze_database_schema(payload: dict):
    # payload: { "tables": [...], "relations": [...] }
    normalized = load_schema_from_json(payload)
    result = analyze_schema(normalized)
    return {"analysis": result}
