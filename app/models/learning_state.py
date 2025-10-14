from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class LearningState(TypedDict):
    """
    State for the learning workflow.

    Tracks the progression through the course, user understanding,
    and conversation history.
    """
    # Conversation messages
    messages: Annotated[list[AnyMessage], add_messages]

    # Learning progression
    current_stage: Literal[
        "introduction",      # Welcome and course overview
        "teaching",          # Active teaching mode
        "assessment",        # Check understanding
        "question_answering", # Answer student questions
        "completed"          # Course completed
    ]

    # Course progress tracking
    topics_covered: list[str]  # Topics already taught
    current_topic: str | None   # Topic being taught now
    topics_remaining: list[str] # Topics to cover

    # Understanding metrics
    understanding_level: Literal["beginner", "intermediate", "advanced"]
    questions_asked: int        # Number of questions student asked
    assessments_passed: int     # Number of successful assessments

    # Slide management - each topic gets its own slide
    slides: list[dict]          # All slides generated for the course
    current_slide_index: int    # Index of currently active slide (0-based)

    # User context
    user_name: str | None
    learning_goal: str | None
