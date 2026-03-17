from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from ..startup import ModulesContainer, start_backend, stop_backend
from .middleware import (
    pydantic_validation_exception_handler,
    validation_exception_handler,
)
from .settings import settings
from .v1.api import api_v1_router


class ApiApplication:
    """
    Encapsulates the FastAPI application creation and configuration.
    Using environment-based settings for production-level robustness.
    """

    def __init__(self):
        self.app = FastAPI(
            title=settings.PROJECT_NAME,
            description=(
                "API for dataset exploration, multi-engine inverse mapping, "
                "and evaluation."
            ),
            version=settings.VERSION,
            debug=settings.DEBUG,
            docs_url="/api/docs" if settings.DEBUG else None,
            redoc_url="/api/redoc" if settings.DEBUG else None,
            openapi_url="/api/openapi.json" if settings.DEBUG else None,
            lifespan=self._lifespan,
        )
        self._configure_middleware()
        self._configure_exception_handlers()
        self._configure_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # Explicitly wire the container to the application modules
        # We target specific route modules to avoid issues with broken package imports
        ModulesContainer.wire(
            modules=[
                "src.api.v1.datasets.routes",
                "src.api.v1.inverse.routes",
                "src.api.v1.modeling.routes",
                "src.api.v1.evaluation.routes",
            ]
        )

        # Startup
        await start_backend(ModulesContainer)
        yield
        # Shutdown
        await stop_backend(ModulesContainer)

    def _configure_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.BACKEND_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _configure_exception_handlers(self):
        self.app.add_exception_handler(
            RequestValidationError, validation_exception_handler
        )
        self.app.add_exception_handler(
            ValidationError, pydantic_validation_exception_handler
        )

    def _configure_routes(self):
        # Health check (Native /api/health)
        @self.app.get("/api/health", tags=["Health"])
        async def health_check():
            return {
                "status": "healthy",
                "env": settings.ENV,
                "version": settings.VERSION,
            }

        # Include aggregated routers for each version
        self.app.include_router(api_v1_router, prefix=settings.API_V1_STR)

    def get_app(self) -> FastAPI:
        return self.app


api_application = ApiApplication()
app = api_application.get_app()
