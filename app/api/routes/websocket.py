"""
Simplified WebSocket route with token streaming.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.ai_teacher import get_teacher_service
from app.utils.error_messages import format_learner_error
from langchain_core.messages import HumanMessage
import json
import uuid
import logging
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)


def generate_id() -> str:
    """Generate unique message ID."""
    return f"{datetime.utcnow().timestamp()}_{uuid.uuid4().hex[:6]}"


async def send_json(ws: WebSocket, data: dict):
    """Send JSON with auto-generated message ID."""
    try:
        if "message_id" not in data:
            data["message_id"] = generate_id()
        await ws.send_json(data)
        logger.debug(f"Sent: {data.get('type', 'unknown')}")
    except WebSocketDisconnect:
        raise
    except Exception as e:
        logger.warning(f"Send failed: {e}")


@router.websocket("/ws/chat/{thread_id}")
async def websocket_chat(ws: WebSocket, thread_id: str):
    """WebSocket endpoint for streaming chat."""
    await ws.accept()
    connection_id = generate_id()
    logger.info(f"Connected: {connection_id} (thread: {thread_id})")

    try:
        teacher_service = get_teacher_service()
        config = {"configurable": {"thread_id": thread_id}}

        # Check if this is a new session
        is_new_session = thread_id not in teacher_service.sessions
        
        if is_new_session:
            logger.info(f"New session: {thread_id}")
            # Check if state exists in checkpointer
            existing_state = teacher_service.workflow.graph.get_state(config)
            
            if not existing_state or not existing_state.values or not existing_state.values.get("slides"):
                # Truly new session - trigger introduction
                logger.info("Fresh session - starting introduction")
                await handle_stream(ws, teacher_service, thread_id, "(start)", is_new=True)
            
            teacher_service.sessions.add(thread_id)

        # Main message loop
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "ping":
                await send_json(ws, {"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                
            elif msg_type == "message":
                content = data.get("content", "").strip()
                if not content:
                    await send_json(ws, {"type": "error", "message": "Empty message"})
                    continue
                    
                await handle_stream(ws, teacher_service, thread_id, content, is_new=False)
                
            else:
                await send_json(ws, {"type": "error", "message": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"Disconnected: {connection_id}")
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON from {connection_id}")
        await send_json(ws, {"type": "error", "message": "Invalid JSON"})
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await send_json(ws, {"type": "error", "message": format_learner_error(e)})
        except:
            pass
    finally:
        try:
            await ws.close()
        except:
            pass
        logger.info(f"Cleaned up: {connection_id}")


async def handle_stream(
    ws: WebSocket, 
    teacher_service, 
    thread_id: str, 
    message: str,
    is_new: bool = False
):
    """
    Handle streaming for a message.
    
    Args:
        ws: WebSocket connection
        teacher_service: AI teacher service instance
        thread_id: Thread ID for state persistence
        message: User's message content
        is_new: Whether this is a new session (needs state initialization)
    """
    try:
        await send_json(ws, {"type": "stream_start", "message": "Meemo is thinking..."})

        config = {"configurable": {"thread_id": thread_id}}

        # Prepare input data
        if is_new:
            # Initialize new session state
            logger.info("Initializing new session state")
            initial_state = teacher_service.workflow.initialize_state()
            initial_state["messages"].append(HumanMessage(content=message))
            input_data = initial_state
        else:
            # Existing session - just add new message
            input_data = {"messages": [HumanMessage(content=message)]}

        # Track streaming state
        accumulated = ""
        current_stage = None
        current_slide = None
        seen_nodes = set()

        # Stream with updates mode for incremental token delivery
        async for chunk in teacher_service.workflow.graph.astream(
            input_data, 
            config=config, 
            stream_mode="updates"
        ):
            for node_name, output in chunk.items():
                if node_name == "__end__":
                    continue

                # Send node notification (only once per node)
                if node_name not in seen_nodes:
                    seen_nodes.add(node_name)
                    await send_json(ws, {
                        "type": "node_start",
                        "node": node_name
                    })

                # Extract and stream tokens
                if "messages" in output and output["messages"]:
                    last_msg = output["messages"][-1]
                    
                    if hasattr(last_msg, "type") and last_msg.type == "ai":
                        content = last_msg.content
                        
                        # Calculate delta (new tokens)
                        if len(content) > len(accumulated):
                            delta = content[len(accumulated):]
                            accumulated = content
                            
                            # Stream the delta
                            await send_json(ws, {
                                "type": "token",
                                "content": delta,
                                "node": node_name
                            })

                # Track stage changes
                if "current_stage" in output:
                    new_stage = output["current_stage"]
                    if new_stage != current_stage:
                        current_stage = new_stage
                        await send_json(ws, {
                            "type": "stage_change",
                            "stage": current_stage,
                            "node": node_name
                        })

                # Track slide updates
                if "slides" in output:
                    slides = output["slides"]
                    slide_idx = output.get("current_slide_index", len(slides) - 1)
                    
                    if slides and slide_idx < len(slides):
                        slide_data = slides[slide_idx]
                        
                        # Only send if it's a new slide
                        if current_slide is None or current_slide.get("slide_number") != slide_data.get("slide_number"):
                            current_slide = slide_data
                            await send_json(ws, {
                                "type": "slide",
                                "slide": slide_data,
                                "slide_index": slide_idx,
                                "total_slides": len(slides)
                            })

        # Get final state for completion info
        final_state = teacher_service.workflow.graph.get_state(config)
        result = final_state.values if final_state else {}

        # Send final response
        await send_final(ws, result, accumulated)
        await send_json(ws, {"type": "stream_end"})

        logger.info(f"Stream completed: {len(accumulated)} characters")

    except Exception as e:
        logger.error(f"Error in handle_stream: {e}", exc_info=True)
        await send_json(ws, {
            "type": "error",
            "message": format_learner_error(e),
            "technical_details": str(e) if logger.level <= logging.DEBUG else None
        })


async def send_final(ws: WebSocket, result: dict, message: str):
    """Send final completion message with full state."""
    slides = result.get("slides", [])
    idx = result.get("current_slide_index", 0)
    slide = slides[idx] if slides and idx < len(slides) else None

    await send_json(ws, {
        "type": "response_complete",
        "message": message,
        "slide": slide,
        "thread_id": result.get("thread_id"),
        "stage": result.get("current_stage", "teaching"),
        "current_topic": result.get("current_topic"),
        "topics_covered": result.get("topics_covered", []),
        "topics_remaining": result.get("topics_remaining", []),
        "understanding_level": result.get("understanding_level", "beginner"),
        "assessments_passed": result.get("assessments_passed", 0),
        "questions_asked": result.get("questions_asked", 0),
        "user_name": result.get("user_name"),
        "learning_goal": result.get("learning_goal"),
        "current_slide_index": idx,
        "total_slides": len(slides)
    })