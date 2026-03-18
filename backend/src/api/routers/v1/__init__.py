from fastapi import APIRouter

from .datasets.routes import router as datasets_router
from .evaluation.routes import router as evaluation_router
from .inverse.routes import router as inverse_router
from .modeling.routes import router as modeling_router

router = APIRouter(prefix="/v1")

router.include_router(datasets_router)
router.include_router(inverse_router)
router.include_router(evaluation_router)
router.include_router(modeling_router)
