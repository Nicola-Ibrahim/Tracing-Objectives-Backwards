from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .middleware import (
    pydantic_validation_exception_handler,
    validation_exception_handler,
)
from .v1.datasets.routes import router as datasets_router
from .v1.evaluation.routes import router as evaluation_router
from .v1.inverse.routes import router as inverse_router

app = FastAPI(
    title="Tracing Objectives Backwards API",
    description="API for dataset exploration, multi-engine inverse mapping, and evaluation.",
    version="1.0.0",
)

# Exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ValidationError, pydantic_validation_exception_handler)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Include routers
app.include_router(datasets_router, prefix="/api/v1/datasets", tags=["Datasets"])
app.include_router(inverse_router, prefix="/api/v1/inverse", tags=["Inverse Mapping"])
app.include_router(evaluation_router, prefix="/api/v1/evaluation", tags=["Evaluation"])

# Mount static files
# Note: Keep this if the backend serves the frontend.
# If using next.js dev server, this can be commented out or kept as fallback.
try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="ui")
except Exception:
    pass
