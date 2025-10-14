from langchain_core.messages import HumanMessage
from app.models.schemas import SlideContent
from app.services.learning_workflow import LearningWorkflow
from app.models.learning_state import LearningState
import uuid


class AITeacherService:
    """AI Teacher service using LangGraph workflow."""

    def __init__(self):
        self.workflow = LearningWorkflow()
        # Store active learning sessions
        self.sessions: dict[str, LearningState] = {}

    async def chat(self, user_message: str, thread_id: str | None = None, user_name: str | None = None) -> tuple[str, SlideContent, str, str]:
        """
        Process a chat message through the learning workflow.

        Args:
            user_message: The user's message
            thread_id: Optional thread ID for conversation continuity
            user_name: Optional user name

        Returns:
            Tuple of (ai_message, slide_content, thread_id, current_stage)
        """
        if not thread_id:
            thread_id = str(uuid.uuid4())

        # Get or initialize state
        if thread_id not in self.sessions:
            state = self.workflow.initialize_state(user_name)
            self.sessions[thread_id] = state
        else:
            state = self.sessions[thread_id]

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_message))

        # Run the workflow
        result = self.workflow.graph.invoke(state)

        # Update session
        self.sessions[thread_id] = result

        # Extract response
        ai_message = result["messages"][-1].content

        # Get current slide from slides collection
        current_slide_index = result.get("current_slide_index", 0)
        slides = result.get("slides", [])

        if slides and current_slide_index < len(slides):
            slide_data = slides[current_slide_index]
        else:
            slide_data = {}

        # Convert slide data to SlideContent
        slide = SlideContent(
            title=slide_data.get("title", "Learning Slide"),
            content=slide_data.get("content", ai_message[:200]),
            code_example=slide_data.get("code_example"),
            visual_description=slide_data.get("visual_description", "Illustration of the concept")
        )

        current_stage = result.get("current_stage", "teaching")

        return ai_message, slide, thread_id, current_stage

    def get_session_info(self, thread_id: str) -> dict:
        """Get information about a learning session."""
        if thread_id not in self.sessions:
            return {}

        state = self.sessions[thread_id]
        return {
            "current_stage": state.get("current_stage"),
            "current_topic": state.get("current_topic"),
            "topics_covered": state.get("topics_covered", []),
            "topics_remaining": state.get("topics_remaining", []),
            "questions_asked": state.get("questions_asked", 0),
            "understanding_level": state.get("understanding_level", "beginner"),
            "current_slide_index": state.get("current_slide_index", 0),
            "total_slides": len(state.get("slides", []))
        }

    def get_all_slides(self, thread_id: str) -> list[dict]:
        """Get all slides for a session."""
        if thread_id not in self.sessions:
            return []

        state = self.sessions[thread_id]
        return state.get("slides", [])

    def navigate_slide(self, thread_id: str, direction: str) -> dict | None:
        """Navigate to next or previous slide."""
        if thread_id not in self.sessions:
            return None

        state = self.sessions[thread_id]
        current_index = state.get("current_slide_index", 0)
        slides = state.get("slides", [])

        if direction == "next" and current_index < len(slides) - 1:
            state["current_slide_index"] = current_index + 1
            return slides[current_index + 1]
        elif direction == "previous" and current_index > 0:
            state["current_slide_index"] = current_index - 1
            return slides[current_index - 1]

        return None


# Singleton instance
_teacher_service = None


def get_teacher_service() -> AITeacherService:
    """Get or create the AI teacher service instance."""
    global _teacher_service
    if _teacher_service is None:
        _teacher_service = AITeacherService()
    return _teacher_service
