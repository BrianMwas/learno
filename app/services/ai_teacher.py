from langchain_core.messages import HumanMessage
from app.models.schemas import SlideContent
from app.services.learning_workflow import LearningWorkflow
from app.models.learning_state import LearningState
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
import logging

logger = logging.getLogger(__name__)


class AITeacherService:
    """AI Teacher service using LangGraph workflow."""

    def __init__(self):
        self.workflow = LearningWorkflow()
        # Store active learning sessions
        self.sessions: dict[str, LearningState] = {}

    def resume_with_answer(self, thread_id: str, answer: str) -> tuple[str, SlideContent, str, str]:
        """
        Resume a paused workflow with user's answer to an interrupt.
        This is called after an interrupt requesting user input.

        Args:
            thread_id: The session thread ID
            answer: User's answer (e.g., their name or learning goal)

        Returns:
            Tuple of (ai_message, slide_content, thread_id, current_stage)

        Raises:
            ValueError: If thread_id or answer is invalid
            Exception: For other processing errors
        """
        try:
            if not thread_id or not thread_id.strip():
                raise ValueError("Thread ID cannot be empty")

            if answer is None:
                raise ValueError("Answer cannot be None")

            logger.info(f"Resuming workflow with answer for thread_id: {thread_id}")

            config = {"configurable": {"thread_id": thread_id}}

            try:
                result = self.workflow.graph.invoke(Command(resume=answer), config=config)
            except GraphInterrupt as interrupt_exception:
                logger.info("Another interrupt encountered during resume")

                interrupt_value = interrupt_exception.interrupts[0].value if interrupt_exception.interrupts else "Please provide your information"

                slide = SlideContent(
                    title="Getting Started",
                    content=interrupt_value,
                    code_example=None,
                    visual_description="User information prompt"
                )

                return interrupt_value, slide, thread_id, "awaiting_user_input"

            self.sessions[thread_id] = result

            if not result.get("messages"):
                raise ValueError("No messages in workflow result after resume")

            ai_message = result["messages"][-1].content

            current_slide_index = result.get("current_slide_index", 0)
            slides = result.get("slides", [])

            if slides and current_slide_index < len(slides):
                slide_data = slides[current_slide_index]
                slide = SlideContent(
                    title=slide_data.get("title", "Learning Slide"),
                    content=slide_data.get("content", ai_message[:200] if ai_message else "Content unavailable"),
                    code_example=slide_data.get("code_example"),
                    visual_description=slide_data.get("visual_description", "Illustration of the concept")
                )
            else:
                slide = SlideContent(
                    title="Learning Session",
                    content=ai_message[:200] if ai_message else "Content unavailable",
                    code_example=None,
                    visual_description="Illustration of the concept"
                )

            current_stage = result.get("current_stage", "teaching")

            logger.info(f"Workflow resumed successfully for thread_id: {thread_id}")
            return ai_message, slide, thread_id, current_stage

        except ValueError as e:
            logger.error(f"Validation error in resume_with_answer: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error resuming workflow for thread_id {thread_id}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to resume workflow: {str(e)}")

    async def chat(self, user_message: str, thread_id: str) -> tuple[str, SlideContent, str, str]:
        """
        Process a chat message through the learning workflow.

        Args:
            user_message: The user's message
            thread_id: Thread ID for conversation continuity

        Returns:
            Tuple of (ai_message, slide_content, thread_id, current_stage)

        Raises:
            ValueError: If user_message or thread_id is invalid
            Exception: For other processing errors
        """
        try:
            if not user_message or not user_message.strip():
                raise ValueError("User message cannot be empty")

            if not thread_id or not thread_id.strip():
                raise ValueError("Thread ID cannot be empty")

            # Get or initialize state
            if thread_id not in self.sessions:
                logger.info(f"Initializing new session for thread_id: {thread_id}")
                state = self.workflow.initialize_state()
                self.sessions[thread_id] = state
            else:
                logger.info(f"Initializing new session for thread_id already set: {thread_id}")
                state = self.sessions[thread_id]

            # Add user message to state
            state["messages"].append(HumanMessage(content=user_message))

            # Run the workflow with config for checkpointer
            config = {"configurable": {"thread_id": thread_id, }}

            try:
                result = self.workflow.graph.invoke(state, config=config)
            except GraphInterrupt as interrupt_exception:
                # Handle interrupt - graph is waiting for user input
                logger.info(f"Graph interrupted with prompt")

                # Extract interrupt value (the prompt string)
                interrupt_value = interrupt_exception.interrupts[0].value if interrupt_exception.interrupts else "Please provide your information"

                logger.info(f"Interrupt prompt: {interrupt_value}")

                # Create a response indicating we need user input
                slide = SlideContent(
                    title="Getting Started",
                    content=interrupt_value,
                    code_example=None,
                    visual_description="User information prompt"
                )

                # Return with special stage so frontend knows to collect input
                return interrupt_value, slide, thread_id, "awaiting_user_input"

            # Update session
            self.sessions[thread_id] = result

            # Extract response
            if not result.get("messages"):
                raise ValueError("No messages in workflow result")

            ai_message = result["messages"][-1].content

            # Get current slide from slides collection
            current_slide_index = result.get("current_slide_index", 0)
            slides = result.get("slides", [])
            print("slides ", slides)
            logger.info("slide data", slides)

            # Create slide with proper defaults
            if slides and current_slide_index < len(slides):
                slide_data = slides[current_slide_index]
                print("slide data", slide_data)
                slide = SlideContent(
                    title=slide_data.get("title", "Learning Slide"),
                    content=slide_data.get("content", ai_message[:200] if ai_message else "Content unavailable"),
                    code_example=slide_data.get("code_example"),
                    visual_description=slide_data.get("visual_description", "Illustration of the concept")
                )
                print("slides data", slide)
            else:
                # Fallback slide when no slides exist yet
                slide = SlideContent(
                    title="Learning Session",
                    content=ai_message[:200] if ai_message else "Content unavailable",
                    code_example=None,
                    visual_description="Illustration of the concept"
                )
                logger.info(f"Chat slide data else {thread_id}, stage: {slide}")

            current_stage = result.get("current_stage", "teaching")

            logger.info(f"Chat processed successfully for thread_id: {thread_id}, stage: {current_stage}")
            return ai_message, slide, thread_id, current_stage

        except ValueError as e:
            logger.error(f"Validation error in chat: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing chat for thread_id {thread_id}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to process chat message: {str(e)}")

    def get_session_info(self, thread_id: str) -> dict:
        """
        Get information about a learning session.

        Args:
            thread_id: The session thread ID

        Returns:
            Dictionary containing session information, empty dict if session not found
        """
        try:
            if not thread_id or not thread_id.strip():
                logger.warning("Empty thread_id provided to get_session_info")
                return {}

            if thread_id not in self.sessions:
                logger.info(f"Session not found for thread_id: {thread_id}")
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
        except Exception as e:
            logger.error(f"Error getting session info for thread_id {thread_id}: {str(e)}", exc_info=True)
            return {}

    def get_all_slides(self, thread_id: str) -> list[dict]:
        """
        Get all slides for a session.

        Args:
            thread_id: The session thread ID

        Returns:
            List of slide dictionaries, empty list if session not found
        """
        try:
            if not thread_id or not thread_id.strip():
                logger.warning("Empty thread_id provided to get_all_slides")
                return []

            if thread_id not in self.sessions:
                logger.info(f"Session not found for thread_id: {thread_id}")
                return []

            state = self.sessions[thread_id]
            return state.get("slides", [])
        except Exception as e:
            logger.error(f"Error getting slides for thread_id {thread_id}: {str(e)}", exc_info=True)
            return []

    
    def navigate_slide(self, thread_id: str, direction: str) -> dict | None:
        """
        Navigate to next or previous slide.

        Args:
            thread_id: The session thread ID
            direction: Either "next" or "previous"

        Returns:
            Slide dictionary if navigation successful, None otherwise

        Raises:
            ValueError: If direction is invalid
        """
        try:
            if not thread_id or not thread_id.strip():
                logger.warning("Empty thread_id provided to navigate_slide")
                return None

            if direction not in ["next", "previous"]:
                raise ValueError(f"Invalid direction: {direction}. Must be 'next' or 'previous'")

            if thread_id not in self.sessions:
                logger.info(f"Session not found for thread_id: {thread_id}")
                return None

            state = self.sessions[thread_id]
            current_index = state.get("current_slide_index", 0)
            slides = state.get("slides", [])

            if not slides:
                logger.warning(f"No slides available for thread_id: {thread_id}")
                return None

            if direction == "next" and current_index < len(slides) - 1:
                state["current_slide_index"] = current_index + 1
                logger.info(f"Navigated to slide {current_index + 1} for thread_id: {thread_id}")
                return slides[current_index + 1]
            elif direction == "previous" and current_index > 0:
                state["current_slide_index"] = current_index - 1
                logger.info(f"Navigated to slide {current_index - 1} for thread_id: {thread_id}")
                return slides[current_index - 1]

            logger.info(f"Cannot navigate {direction} from slide {current_index} for thread_id: {thread_id}")
            return None

        except ValueError as e:
            logger.error(f"Validation error in navigate_slide: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error navigating slide for thread_id {thread_id}: {str(e)}", exc_info=True)
            return None


# Singleton instance
_teacher_service = None


def get_teacher_service() -> AITeacherService:
    """
    Get or create the AI teacher service instance.

    Returns:
        Singleton instance of AITeacherService

    Raises:
        Exception: If service initialization fails
    """
    global _teacher_service
    try:
        if _teacher_service is None:
            logger.info("Initializing AI Teacher Service")
            _teacher_service = AITeacherService()
        return _teacher_service
    except Exception as e:
        logger.error(f"Failed to initialize AI Teacher Service: {str(e)}", exc_info=True)
        raise Exception(f"Service initialization failed: {str(e)}")
