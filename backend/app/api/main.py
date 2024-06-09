from fastapi import APIRouter

from app.api.routes import inference

api_router = APIRouter()

api_router.include_router(inference.router, prefix="/answer", tags=["inference"])
