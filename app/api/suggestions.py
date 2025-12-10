from fastapi import APIRouter
from pydantic import ValidationError
from app.services.migration_service import generate_migration_plan
from app.models.decomposition import SuggestionsRequest

router = APIRouter()

@router.post("/")
def generate_suggestions(payload: dict):
    # payload: results from analysis
    try:
        req = SuggestionsRequest.parse_obj(payload)
    except ValidationError as ve:
        return {"errors": ve.errors()}
    plan = generate_migration_plan(req.dict())
    return {"migration_plan": plan}
