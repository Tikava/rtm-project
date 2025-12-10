from fastapi import FastAPI
from app.api import health, analysis, suggestions

app = FastAPI(
    title="AI‑Assisted Migration Prototype",
    version="0.1",
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(suggestions.router, prefix="/suggestions", tags=["Suggestions"])
