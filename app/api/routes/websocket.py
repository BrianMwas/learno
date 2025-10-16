"""
WebSocket route using proper LangGraph streaming modes.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.ai_teacher import get_teacher_service
from app.utils.error_messages import format_learner_error
from langchain_core.messages import HumanMessage
import json
import uuid
import logging
from datetime import datetime
from starlette.websockets import WebSocketState

router = APIRouter()
logger = logging.getLogger(__name__)


def generate_id() -> str:
    return f"{datetime.utcnow().timestamp()}_{uuid.uuid4().hex[:6]}"


async def send_json(ws: WebSocket, data: dict) -> bool:
    """Send JSON safely, return False if connection closed."""
    try:
        if ws.client_state != WebSocketState.CONNECTED:
            return False
            
        if "message_id" not in data:
            data["message_id"] = generate_id()
            
        await ws.send_json(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as e:
        logger.warning(f"Send failed: {e}")
        return False


@router.websocket("/ws/chat/{thread_id}")
async def websocket_chat(ws: WebSocket, thread_id: str):
    """WebSocket endpoint with proper LangGraph streaming."""
    await ws.accept()
    connection_id = generate_id()
    logger.info(f"Connected: {connection_id} (thread: {thread_id})")

    try:
        teacher_service = get_teacher_service()
        config = {"configurable": {"thread_id": thread_id, "recursion_limit": 50}}

        # Check for new session
        is_new_session = thread_id not in teacher_service.sessions
        
        if is_new_session:
            logger.info(f"New session: {thread_id}")
            existing_state = teacher_service.workflow.graph.get_state(config)
            
            if not existing_state or not existing_state.values or not existing_state.values.get("slides"):
                logger.info("Fresh session - starting introduction")
                await handle_stream(ws, teacher_service, thread_id, "", is_new=True)
            
            teacher_service.sessions.add(thread_id)

        # Main message loop
        while True:
            if ws.client_state != WebSocketState.CONNECTED:
                break
                
            try:
                raw = await ws.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                break
                
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await send_json(ws, {"type": "error", "message": "Invalid JSON"})
                continue
                
            msg_type = data.get("type")

            if msg_type == "ping":
                await send_json(ws, {
                    "type": "pong", 
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            elif msg_type == "message":
                content = data.get("content", "").strip()
                if not content:
                    await send_json(ws, {"type": "error", "message": "Empty message"})
                    continue
                    
                stream_success = await handle_stream(
                    ws, teacher_service, thread_id, content, is_new=False
                )
                
                if not stream_success:
                    break
                    
            else:
                await send_json(ws, {
                    "type": "error", 
                    "message": f"Unknown type: {msg_type}"
                })

    except WebSocketDisconnect:
        logger.info(f"Clean disconnect: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await send_json(ws, {"type": "error", "message": format_learner_error(e)})
    finally:
        try:
            if ws.client_state == WebSocketState.CONNECTED:
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
) -> bool:
    """
    Handle streaming using proper LangGraph stream modes.
    
    Uses dual streaming:
    - "messages" mode for LLM tokens
    - "updates" mode for state changes (slides, stage, etc.)
    """
    try:
        if not await send_json(ws, {
            "type": "stream_start", 
            "message": "Meemo is thinking..."
        }):
            return False

        config = {"configurable": {"thread_id": thread_id}}

        # Prepare input
        if is_new:
            initial_state = teacher_service.workflow.initialize_state()
            if message:  # Only add message if not empty
                initial_state["messages"].append(HumanMessage(content=message))
            input_data = initial_state
        else:
            input_data = {"messages": [HumanMessage(content=message)]}

        # Track state
        accumulated_content = ""
        current_stage = None
        current_slide = None
        seen_nodes = set()

        # ✨ Use multiple stream modes: messages + updates
        async for stream_mode, chunk in teacher_service.workflow.graph.astream(
            input_data,
            config=config,
            stream_mode=["messages", "updates"]  # Dual streaming!
        ):
            if ws.client_state != WebSocketState.CONNECTED:
                return False

            # ========== MESSAGES MODE: LLM Tokens ==========
            if stream_mode == "messages":
                # chunk is a tuple: (message_chunk, metadata)
                message_chunk, metadata = chunk
                
                # Extract token from message chunk
                if hasattr(message_chunk, "content") and message_chunk.content:
                    token = message_chunk.content
                    accumulated_content += token
                    
                    # Stream token to client
                    if not await send_json(ws, {
                        "type": "token",
                        "content": token,
                        "metadata": {
                            "node": metadata.get("langgraph_node"),
                            "step": metadata.get("langgraph_step")
                        }
                    }):
                        return False

            # ========== UPDATES MODE: State Changes ==========
            elif stream_mode == "updates":
                # chunk is a dict: {node_name: node_output}
                for node_name, node_output in chunk.items():
                    if node_name == "__end__":
                        continue

                    # Notify about node execution
                    if node_name not in seen_nodes:
                        seen_nodes.add(node_name)
                        if not await send_json(ws, {
                            "type": "node_start",
                            "node": node_name
                        }):
                            return False

                    # Check for stage changes
                    if "current_stage" in node_output:
                        new_stage = node_output["current_stage"]
                        if new_stage != current_stage:
                            current_stage = new_stage
                            if not await send_json(ws, {
                                "type": "stage_change",
                                "stage": current_stage,
                                "node": node_name
                            }):
                                return False

                    # Check for slide updates
                    if "slides" in node_output:
                        slides = node_output["slides"]
                        slide_idx = node_output.get("current_slide_index", len(slides) - 1)
                        
                        if slides and slide_idx < len(slides):
                            slide_data = slides[slide_idx]
                            
                            # Only send new slides
                            if (current_slide is None or 
                                current_slide.get("slide_number") != slide_data.get("slide_number")):
                                current_slide = slide_data
                                if not await send_json(ws, {
                                    "type": "slide",
                                    "slide": slide_data,
                                    "slide_index": slide_idx,
                                    "total_slides": len(slides)
                                }):
                                    return False

        # Get final state
        final_state = teacher_service.workflow.graph.get_state(config)
        result = final_state.values if final_state else {}

        # Send completion
        if not await send_final(ws, result, accumulated_content):
            return False
            
        if not await send_json(ws, {"type": "stream_end"}):
            return False

        logger.info(f"✅ Stream completed: {len(accumulated_content)} chars")
        return True

    except WebSocketDisconnect:
        logger.info("Client disconnected during stream")
        return False
    except Exception as e:
        logger.error(f"❌ Error in handle_stream: {e}", exc_info=True)
        await send_json(ws, {
            "type": "error",
            "message": format_learner_error(e)
        })
        return False


async def send_final(ws: WebSocket, result: dict, message: str) -> bool:
    """Send final completion message."""
    slides = result.get("slides", [])
    idx = result.get("current_slide_index", 0)
    slide = slides[idx] if slides and idx < len(slides) else None

    return await send_json(ws, {
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