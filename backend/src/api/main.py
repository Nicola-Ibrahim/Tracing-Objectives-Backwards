from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from ..startup import BackendConfig, BackendStartUp
from .core.middlewares.pydantic import pydantic_validation_exception_handler
from .core.middlewares.validation import validation_exception_handler
from .core.settings import settings
from .routers.v1 import router as api_v1_router


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
            docs_url="/docs" if settings.DEBUG else None,
            redoc_url="/redoc" if settings.DEBUG else None,
            openapi_url="/openapi.json" if settings.DEBUG else None,
            lifespan=self._lifespan,
        )
        self._configure_middleware()
        self._configure_exception_handlers()
        self._configure_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # Startup
        config = BackendConfig(
            redis_url=settings.REDIS_URL, data_storage_path=settings.DATA_STORAGE_PATH
        )
        backend_startup = BackendStartUp(config=config)
        await backend_startup.start()
        yield
        # Shutdown
        await backend_startup.stop()

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
        # Health check (Native /health)
        @self.app.get("/health", tags=["Health"])
        async def health_check():
            return {
                "status": "healthy",
                "env": settings.ENV,
                "version": settings.VERSION,
            }

        # Include aggregated routers for each version
        self.app.include_router(api_v1_router)

    def get_app(self) -> FastAPI:
        return self.app


api_application = ApiApplication()
app = api_application.get_app()
