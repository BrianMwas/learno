from pydantic import BaseModel
from typing import Literal


class ChatMessage(BaseModel):
    """Chat message model."""
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint - just the message."""
    message: str


class ResumeRequest(BaseModel):
    """Request model for resume endpoint - answer to interrupt."""
    answer: str


class SlideContent(BaseModel):
    """Slide content model."""
    title: str
    content: str
    code_example: str | None = None
    visual_description: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    message: str
    slide: SlideContent
    thread_id: str
    current_stage: str
    current_topic: str | None = None
    topics_covered: list[str] = []
    current_slide_index: int = 0
    total_slides: int = 0


class SessionInfoResponse(BaseModel):
    """Response model for session info endpoint."""
    current_stage: str | None
    current_topic: str | None
    topics_covered: list[str]
    topics_remaining: list[str]
    questions_asked: int
    understanding_level: str
    current_slide_index: int
    total_slides: int


class SlideNavigationResponse(BaseModel):
    """Response model for slide navigation."""
    slide: SlideContent
    slide_index: int
    total_slides: int
