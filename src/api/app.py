"""FastAPI application entry point for DataForge Engine.

This module creates the top-level ``FastAPI`` application instance and
wires up the versioned API router.  It is intentionally thin — route
definitions and business logic live in separate modules so that this
file only handles application-level configuration.

Design principles:
  - SRP (Single Responsibility Principle): This module is responsible
    *only* for assembling the application object.  Route handlers,
    request schemas, and business logic each live in their own modules.
  - OCP (Open/Closed Principle): New API versions or feature routers can
    be added via ``app.include_router(...)`` without modifying existing
    route code.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from ..logging_config import get_logger

# Initialize structured logging
logger = get_logger(__name__)

# Create the FastAPI application with metadata used by the auto-generated
# OpenAPI / Swagger docs (available at /docs when the server is running).
app = FastAPI(
    title="DataForge Engine",
    description="Automated Q&A dataset generation pipeline",
    version="0.1.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all dataset-related routes under /api/v1.  The prefix keeps the
# URL namespace organised and makes it straightforward to introduce a v2
# router later without breaking existing clients (OCP).
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Log application startup."""
    logger.info(
        "DataForge Engine API started",
        extra={"version": "0.1.0", "environment": os.getenv("ENV", "development")},
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown."""
    logger.info("DataForge Engine API shutting down")


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "healthy", "service": "dataforge-engine"}
