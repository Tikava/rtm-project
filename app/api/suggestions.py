from fastapi import APIRouter
from app.services.migration_service import generate_migration_plan

router = APIRouter()

@router.post("/")
def generate_suggestions(payload: dict):
    # payload: results from analysis
    plan = generate_migration_plan(payload)
    return {"migration_plan": plan}
