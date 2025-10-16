from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.routes import chat, websocket
import logging

settings = get_settings()

# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG if you want more details
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS
# Note: For development, allow all origins to enable WebSocket connections
# In production, set specific origins in .env via BACKEND_CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}", tags=["chat"])
app.include_router(websocket.router, prefix=f"{settings.API_V1_STR}", tags=["websocket"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Teacher API",
        "course": settings.COURSE_TOPIC,
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
