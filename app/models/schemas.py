from pydantic import BaseModel, Field
from typing import Literal, Optional


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
    
    

class NameExtraction(BaseModel):
    """Extract user's name from their message."""
    name: Optional[str] = Field(None, description="The user's name if mentioned")
    confidence: str = Field(description="Confidence: 'high', 'medium', 'low', 'none'")
    reasoning: str = Field(description="How the name was determined")


class GoalExtraction(BaseModel):
    """Extract user's learning goal."""
    goal: Optional[str] = Field(None, description="Learning goal if mentioned")
    wants_to_skip: bool = Field(description="True if user wants to skip goal")
    reasoning: str = Field(description="Explanation of extraction")


class ConversationAnalysis(BaseModel):
    """Analyze conversation for routing."""
    is_question: bool = Field(description="Is this a question?")
    is_assessment_answer: bool = Field(description="Is this an assessment answer?")
    suggested_route: Literal[
        "question_answering",
        "assessment_evaluation", 
        "teaching",
        "introduction",
        "continue"
    ] = Field(description="Suggested routing destination")


class AssessmentEvaluation(BaseModel):
    """Evaluate student's assessment answer."""
    judgment: Literal["correct", "partial", "incorrect"]
    what_was_correct: str
    what_was_missing: str
    feedback: str
    should_pass: bool
    needs_review: bool

