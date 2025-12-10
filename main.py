from fastapi import FastAPI
from app.core.env import load_env
from app.api import health, analysis, suggestions

load_env()

app = FastAPI(
    title="AI‑Assisted Migration Prototype",
    version="0.1",
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(suggestions.router, prefix="/suggestions", tags=["Suggestions"])
