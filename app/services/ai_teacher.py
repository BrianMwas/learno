from langchain_core.messages import HumanMessage
from app.models.schemas import SlideContent
from app.services.learning_workflow import LearningWorkflow
from langgraph.types import Command
import logging

logger = logging.getLogger(__name__)


class AITeacherService:
    """AI Teacher service using LangGraph workflow."""

    def __init__(self):
        self.workflow = LearningWorkflow()
        # Track which sessions have been created (just for existence check)
        self.sessions = set()

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

            # Resume the graph with Command(resume=answer)
            # The interrupt() call in the node will return this value
            result = self.workflow.graph.invoke(Command(resume=answer), config=config)
            logger.info(f"Graph resumed successfully")

            # Check if another interrupt occurred
            if "__interrupt__" in result:
                logger.info("Another interrupt encountered during resume")

                interrupts = result["__interrupt__"]
                interrupt_value = interrupts[0].value if interrupts else "Please provide your information"

                slide = SlideContent(
                    title="Getting Started",
                    content=interrupt_value,
                    code_example=None,
                    visual_description="User information prompt"
                )

                return interrupt_value, slide, thread_id, "awaiting_user_input"

            # Mark session as active
            self.sessions.add(thread_id)

            if not result.get("messages"):
                raise ValueError("No messages in workflow result after resume")

            ai_message = result["messages"][-1].content

            current_slide_index = result.get("current_slide_index", 0)
            slides = result.get("slides", [])

            logger.info(f"Resume result - slides: {len(slides)}, index: {current_slide_index}")

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

            logger.info(f"Workflow resumed successfully for thread_id: {thread_id}, stage: {current_stage}")
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

            logger.info(f"Processing chat for thread_id: {thread_id}, message: {user_message[:50]}")

            # Configuration for checkpointer - let LangGraph manage state
            config = {"configurable": {"thread_id": thread_id}}

            # For new conversations, we need to pass initial state
            # For existing conversations, LangGraph loads from checkpoint
            if thread_id not in self.sessions:
                logger.info(f"New session - initializing state for thread_id: {thread_id}")
                # Initialize state and add user message
                initial_state = self.workflow.initialize_state()
                initial_state["messages"].append(HumanMessage(content=user_message))
                input_data = initial_state
            else:
                logger.info(f"Existing session for thread_id: {thread_id}")
                # Just pass the new message - LangGraph will load the rest from checkpoint
                input_data = {"messages": [HumanMessage(content=user_message)]}

            # Invoke the graph
            logger.info(f"Invoking graph for thread_id: {thread_id}")
            result = self.workflow.graph.invoke(input_data, config=config)
            logger.info(f"Graph execution completed for thread_id: {thread_id}")

            # Check if graph paused due to interrupt
            if "__interrupt__" in result:
                logger.info(f"Graph interrupted - waiting for user input")

                # Extract interrupt value (the prompt string)
                interrupts = result["__interrupt__"]
                interrupt_value = interrupts[0].value if interrupts else "Please provide your information"

                logger.info(f"Interrupt prompt: {interrupt_value}")

                # Mark that we have a session (even though it's paused)
                self.sessions.add(thread_id)

                # Create a response indicating we need user input
                slide = SlideContent(
                    title="Getting Started",
                    content=interrupt_value,
                    code_example=None,
                    visual_description="User information prompt"
                )

                # Return with special stage so frontend knows to collect input
                return interrupt_value, slide, thread_id, "awaiting_user_input"

            # Update session marker
            self.sessions.add(thread_id)

            # Extract response
            if not result.get("messages"):
                raise ValueError("No messages in workflow result")

            ai_message = result["messages"][-1].content
            logger.info(f"AI response generated: {ai_message[:100]}")

            # Get current slide from slides collection
            current_slide_index = result.get("current_slide_index", 0)
            slides = result.get("slides", [])

            logger.info(f"Slides count: {len(slides)}, current_index: {current_slide_index}")

            # Create slide with proper defaults
            if slides and current_slide_index < len(slides):
                slide_data = slides[current_slide_index]
                logger.info(f"Using slide at index {current_slide_index}: {slide_data.get('title')}")

                slide = SlideContent(
                    title=slide_data.get("title", "Learning Slide"),
                    content=slide_data.get("content", ai_message[:200] if ai_message else "Content unavailable"),
                    code_example=slide_data.get("code_example"),
                    visual_description=slide_data.get("visual_description", "Illustration of the concept")
                )
            else:
                # Fallback slide when no slides exist yet
                logger.warning(f"No slides available - using fallback")
                slide = SlideContent(
                    title="Learning Session",
                    content=ai_message[:200] if ai_message else "Content unavailable",
                    code_example=None,
                    visual_description="Illustration of the concept"
                )

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

            # Get state from checkpointer
            config = {"configurable": {"thread_id": thread_id}}
            try:
                state = self.workflow.graph.get_state(config)
                if not state or not state.values:
                    logger.info(f"No state found in checkpointer for thread_id: {thread_id}")
                    return {}

                state_values = state.values

                return {
                    "current_stage": state_values.get("current_stage"),
                    "current_topic": state_values.get("current_topic"),
                    "topics_covered": state_values.get("topics_covered", []),
                    "topics_remaining": state_values.get("topics_remaining", []),
                    "questions_asked": state_values.get("questions_asked", 0),
                    "understanding_level": state_values.get("understanding_level", "beginner"),
                    "current_slide_index": state_values.get("current_slide_index", 0),
                    "total_slides": len(state_values.get("slides", []))
                }
            except Exception as e:
                logger.error(f"Error reading from checkpointer: {str(e)}")
                return {}

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

            # Get state from checkpointer
            config = {"configurable": {"thread_id": thread_id}}
            try:
                state = self.workflow.graph.get_state(config)
                if not state or not state.values:
                    return []

                return state.values.get("slides", [])
            except Exception as e:
                logger.error(f"Error reading from checkpointer: {str(e)}")
                return []

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

            # Get and update state from checkpointer
            config = {"configurable": {"thread_id": thread_id}}
            try:
                checkpoint = self.workflow.graph.get_state(config)
                if not checkpoint or not checkpoint.values:
                    return None

                state = checkpoint.values
                current_index = state.get("current_slide_index", 0)
                slides = state.get("slides", [])

                if not slides:
                    logger.warning(f"No slides available for thread_id: {thread_id}")
                    return None

                new_index = None
                if direction == "next" and current_index < len(slides) - 1:
                    new_index = current_index + 1
                elif direction == "previous" and current_index > 0:
                    new_index = current_index - 1

                if new_index is not None:
                    # Update the state in checkpointer
                    state["current_slide_index"] = new_index
                    self.workflow.graph.update_state(config, state)
                    logger.info(f"Navigated to slide {new_index} for thread_id: {thread_id}")
                    return slides[new_index]
                else:
                    logger.info(f"Cannot navigate {direction} from slide {current_index} for thread_id: {thread_id}")
                    return None

            except Exception as e:
                logger.error(f"Error reading/updating checkpointer: {str(e)}")
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
