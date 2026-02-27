from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .middleware import (
    pydantic_validation_exception_handler,
    validation_exception_handler,
)
from .v1.dataset_routes import router as dataset_router
from .v1.generation_routes import router as generation_router

app = FastAPI(
    title="Tracing Objectives Backwards API",
    description="API for dataset exploration and coherent candidate generation.",
    version="0.1.0",
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
app.include_router(dataset_router, prefix="/api/v1/datasets", tags=["Datasets"])
app.include_router(generation_router, prefix="/api/v1/generation", tags=["Generation"])

# Mount static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="ui")
