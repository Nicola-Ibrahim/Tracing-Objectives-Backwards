from fastapi import APIRouter

from .datasets.routes import router as datasets_router
from .evaluation.routes import router as evaluation_router
from .inverse.routes import router as inverse_router
from .modeling.routes import router as modeling_router

api_v1_router = APIRouter()

api_v1_router.include_router(datasets_router)
api_v1_router.include_router(inverse_router)
api_v1_router.include_router(evaluation_router)
api_v1_router.include_router(modeling_router)
