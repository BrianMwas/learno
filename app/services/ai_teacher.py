from langchain_core.messages import HumanMessage
from app.models.schemas import SlideContent
from app.services.learning_workflow import LearningWorkflow
from typing import AsyncGenerator, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)


class AITeacherService:
    """AI Teacher service using LangGraph workflow with streaming support."""

    def __init__(self):
        self.workflow = LearningWorkflow()
        self.sessions = set()

    async def chat_stream(
        self, 
        user_message: str, 
        thread_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat responses through the learning workflow.

        Args:
            user_message: The user's message
            thread_id: Thread ID for conversation continuity

        Yields:
            Dictionary chunks with types: 'token', 'slide', 'stage', 'complete', 'error'

        Raises:
            ValueError: If user_message or thread_id is invalid
        """
        try:
            if not user_message or not user_message.strip():
                raise ValueError("User message cannot be empty")

            if not thread_id or not thread_id.strip():
                raise ValueError("Thread ID cannot be empty")

            logger.info(f"Streaming chat for thread_id: {thread_id}, message: {user_message[:50]}")

            config = {"configurable": {"thread_id": thread_id}}

            # Prepare input data
            if thread_id not in self.sessions:
                logger.info(f"New session - initializing state for thread_id: {thread_id}")
                initial_state = self.workflow.initialize_state()
                initial_state["messages"].append(HumanMessage(content=user_message))
                input_data = initial_state
            else:
                logger.info(f"Existing session for thread_id: {thread_id}")
                input_data = {"messages": [HumanMessage(content=user_message)]}

            # Stream the graph execution
            full_message = ""
            current_slide = None
            current_stage = None

            # Use astream for async streaming
            async for chunk in self.workflow.graph.astream(input_data, config=config, stream_mode="updates"):
                logger.debug(f"Stream chunk: {chunk.keys() if isinstance(chunk, dict) else type(chunk)}")

                # Extract node updates
                for node_name, node_output in chunk.items():
                    if node_name == "__end__":
                        continue

                    logger.info(f"Node '{node_name}' produced output")

                    # Extract stage information
                    if "current_stage" in node_output:
                        current_stage = node_output["current_stage"]
                        yield {
                            "type": "stage",
                            "stage": current_stage,
                            "node": node_name
                        }

                    # Extract messages and stream tokens
                    if "messages" in node_output and node_output["messages"]:
                        last_message = node_output["messages"][-1]
                        
                        if hasattr(last_message, "content"):
                            message_content = last_message.content
                            
                            # Stream the new content (delta from previous)
                            if len(message_content) > len(full_message):
                                delta = message_content[len(full_message):]
                                full_message = message_content
                                
                                yield {
                                    "type": "token",
                                    "content": delta,
                                    "node": node_name
                                }

                    # Extract slide information
                    if "slides" in node_output and node_output["slides"]:
                        current_slide_index = node_output.get("current_slide_index", len(node_output["slides"]) - 1)
                        if current_slide_index < len(node_output["slides"]):
                            slide_data = node_output["slides"][current_slide_index]
                            current_slide = SlideContent(
                                title=slide_data.get("title", "Learning Slide"),
                                content=slide_data.get("content", "Content unavailable"),
                                code_example=slide_data.get("code_example"),
                                visual_description=slide_data.get("visual_description", "Illustration")
                            )
                            
                            yield {
                                "type": "slide",
                                "slide": current_slide.dict(),
                                "slide_index": current_slide_index,
                                "total_slides": len(node_output["slides"])
                            }

            # Mark session as active
            self.sessions.add(thread_id)

            # Get final state for completion info
            final_state = self.workflow.graph.get_state(config)
            if final_state and final_state.values:
                state_values = final_state.values
                
                # Send completion event with full state
                yield {
                    "type": "complete",
                    "message": full_message,
                    "thread_id": thread_id,
                    "stage": state_values.get("current_stage", "teaching"),
                    "slide": current_slide.dict() if current_slide else None,
                    "session_info": {
                        "current_topic": state_values.get("current_topic"),
                        "topics_covered": state_values.get("topics_covered", []),
                        "topics_remaining": state_values.get("topics_remaining", []),
                        "understanding_level": state_values.get("understanding_level", "beginner"),
                        "assessments_passed": state_values.get("assessments_passed", 0),
                        "questions_asked": state_values.get("questions_asked", 0)
                    }
                }

            logger.info(f"Stream completed for thread_id: {thread_id}")

        except ValueError as e:
            logger.error(f"Validation error in chat_stream: {str(e)}")
            yield {"type": "error", "error": str(e), "error_type": "validation"}
        except Exception as e:
            logger.error(f"Error streaming chat for thread_id {thread_id}: {str(e)}", exc_info=True)
            yield {"type": "error", "error": f"Failed to process message: {str(e)}", "error_type": "processing"}

    async def chat(self, user_message: str, thread_id: str) -> tuple[str, SlideContent, str, str]:
        """
        Non-streaming chat (for backwards compatibility).
        Internally uses streaming but collects the full response.

        Args:
            user_message: The user's message
            thread_id: Thread ID for conversation continuity

        Returns:
            Tuple of (ai_message, slide_content, thread_id, current_stage)
        """
        try:
            full_message = ""
            slide = None
            stage = "teaching"

            # Collect from stream
            async for chunk in self.chat_stream(user_message, thread_id):
                if chunk["type"] == "token":
                    full_message += chunk["content"]
                elif chunk["type"] == "slide":
                    slide = SlideContent(**chunk["slide"])
                elif chunk["type"] == "stage":
                    stage = chunk["stage"]
                elif chunk["type"] == "complete":
                    full_message = chunk["message"]
                    if chunk["slide"]:
                        slide = SlideContent(**chunk["slide"])
                    stage = chunk["stage"]
                elif chunk["type"] == "error":
                    raise Exception(chunk["error"])

            # Fallback slide if none created
            if not slide:
                slide = SlideContent(
                    title="Learning Session",
                    content=full_message[:200] if full_message else "Content unavailable",
                    code_example=None,
                    visual_description="Illustration of the concept"
                )

            return full_message, slide, thread_id, stage

        except Exception as e:
            logger.error(f"Error in non-streaming chat: {str(e)}")
            raise

    def get_session_info(self, thread_id: str) -> dict:
        """Get information about a learning session."""
        try:
            if not thread_id or not thread_id.strip():
                logger.warning("Empty thread_id provided to get_session_info")
                return {}

            if thread_id not in self.sessions:
                logger.info(f"Session not found for thread_id: {thread_id}")
                return {}

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
                    "total_slides": len(state_values.get("slides", [])),
                    "user_name": state_values.get("user_name"),
                    "learning_goal": state_values.get("learning_goal"),
                    "assessments_passed": state_values.get("assessments_passed", 0)
                }
            except Exception as e:
                logger.error(f"Error reading from checkpointer: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error getting session info for thread_id {thread_id}: {str(e)}", exc_info=True)
            return {}

    def get_all_slides(self, thread_id: str) -> list[dict]:
        """Get all slides for a session."""
        try:
            if not thread_id or not thread_id.strip():
                logger.warning("Empty thread_id provided to get_all_slides")
                return []

            if thread_id not in self.sessions:
                logger.info(f"Session not found for thread_id: {thread_id}")
                return []

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
        """Navigate to next or previous slide."""
        try:
            if not thread_id or not thread_id.strip():
                logger.warning("Empty thread_id provided to navigate_slide")
                return None

            if direction not in ["next", "previous"]:
                raise ValueError(f"Invalid direction: {direction}. Must be 'next' or 'previous'")

            if thread_id not in self.sessions:
                logger.info(f"Session not found for thread_id: {thread_id}")
                return None

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
                    state["current_slide_index"] = new_index
                    self.workflow.graph.update_state(config, state)
                    logger.info(f"Navigated to slide {new_index} for thread_id: {thread_id}")
                    return slides[new_index]
                else:
                    logger.info(f"Cannot navigate {direction} from slide {current_index}")
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
    """Get or create the AI teacher service instance."""
    global _teacher_service
    try:
        if _teacher_service is None:
            logger.info("Initializing AI Teacher Service")
            _teacher_service = AITeacherService()
        return _teacher_service
    except Exception as e:
        logger.error(f"Failed to initialize AI Teacher Service: {str(e)}", exc_info=True)
        raise Exception(f"Service initialization failed: {str(e)}")