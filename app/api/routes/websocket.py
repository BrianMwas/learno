"""
WebSocket route for streaming chat responses with message deduplication.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.ai_teacher import get_teacher_service
from app.utils.error_messages import format_learner_error, get_stage_error_message
from langgraph.types import Command
import json
import logging
import uuid
from typing import Dict, Set
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

# Track active connections and sent messages per connection
active_connections: Dict[str, Set[str]] = {}
connection_metadata: Dict[str, dict] = {}


def generate_message_id() -> str:
    """Generate unique message ID."""
    return f"{datetime.utcnow().timestamp()}_{uuid.uuid4().hex[:8]}"


def should_send_message(connection_id: str, message_id: str) -> bool:
    """Check if message should be sent (not a duplicate)."""
    if connection_id not in active_connections:
        active_connections[connection_id] = set()
    
    if message_id in active_connections[connection_id]:
        logger.debug(f"Duplicate message {message_id} blocked for {connection_id}")
        return False
    
    active_connections[connection_id].add(message_id)
    return True


def cleanup_old_messages(connection_id: str, max_age_seconds: int = 300):
    """Remove old message IDs to prevent memory bloat."""
    if connection_id in connection_metadata:
        cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        old_messages = [
            msg_id for msg_id, timestamp in connection_metadata[connection_id].get("message_timestamps", {}).items()
            if timestamp < cutoff_time
        ]
        for msg_id in old_messages:
            active_connections[connection_id].discard(msg_id)


async def safe_send_json(websocket: WebSocket, data: dict, connection_id: str) -> bool:
    """
    Safely send JSON with deduplication and error handling.
    Returns True if sent successfully, False otherwise.
    """
    try:
        # Add message ID if not present
        if "message_id" not in data:
            data["message_id"] = generate_message_id()
        
        # Check for duplicates
        if not should_send_message(connection_id, data["message_id"]):
            return False
        
        # Track timestamp for cleanup
        if connection_id not in connection_metadata:
            connection_metadata[connection_id] = {"message_timestamps": {}}
        connection_metadata[connection_id]["message_timestamps"][data["message_id"]] = datetime.utcnow()
        
        # Send message
        await websocket.send_json(data)
        logger.debug(f"Sent message {data['message_id']} to {connection_id}")
        return True
        
    except (WebSocketDisconnect, RuntimeError, ConnectionError) as e:
        logger.warning(f"Failed to send message to {connection_id}: {str(e)}")
        return False


@router.websocket("/ws/chat/{thread_id}")
async def websocket_chat(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for streaming chat with the learning workflow.
    """
    connection_id = f"{thread_id}_{uuid.uuid4().hex[:8]}"
    await websocket.accept()
    logger.info(f"WebSocket connection established: {connection_id} (thread: {thread_id})")
    
    # Initialize connection tracking
    active_connections[connection_id] = set()
    connection_metadata[connection_id] = {
        "thread_id": thread_id,
        "connected_at": datetime.utcnow(),
        "message_timestamps": {}
    }

    try:
        teacher_service = get_teacher_service()

        # Handle new session initialization
        if thread_id not in teacher_service.sessions:
            logger.info(f"New session detected: {thread_id}")
            config = {"configurable": {"thread_id": thread_id}}
            existing_state = teacher_service.workflow.graph.get_state(config)

            if not existing_state or not existing_state.values or len(existing_state.values.get("slides", [])) == 0:
                logger.info("Fresh session - triggering introduction")
                await handle_chat_stream(websocket, teacher_service, thread_id, "(start)", connection_id)
        
        # Periodic cleanup
        cleanup_old_messages(connection_id)

        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message_type = message_data.get("type")
            
            logger.info(f"Received message type '{message_type}' from {connection_id}")

            if message_type == "message":
                user_message = message_data.get("content")
                if not user_message:
                    await safe_send_json(websocket, {
                        "type": "error",
                        "message": "Missing content in message"
                    }, connection_id)
                    continue

                await handle_chat_stream(websocket, teacher_service, thread_id, user_message, connection_id)

            elif message_type == "resume":
                answer = message_data.get("answer")
                if not answer:
                    await safe_send_json(websocket, {
                        "type": "error",
                        "message": "Missing answer in resume"
                    }, connection_id)
                    continue

                await handle_resume_stream(websocket, teacher_service, thread_id, answer, connection_id)

            elif message_type == "ping":
                # Heartbeat response
                await safe_send_json(websocket, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)

            else:
                await safe_send_json(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }, connection_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except RuntimeError as e:
        if "close message has been sent" in str(e):
            logger.info(f"Client already disconnected: {connection_id}")
        else:
            logger.error(f"WebSocket runtime error: {connection_id}: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {str(e)}", exc_info=True)
        try:
            await safe_send_json(websocket, {
                "type": "error",
                "message": str(e)
            }, connection_id)
        except:
            pass
    finally:
        # Cleanup connection tracking
        active_connections.pop(connection_id, None)
        connection_metadata.pop(connection_id, None)
        logger.info(f"Cleaned up connection: {connection_id}")
        try:
            await websocket.close()
        except:
            pass


async def handle_chat_stream(
    websocket: WebSocket,
    teacher_service,
    thread_id: str,
    user_message: str,
    connection_id: str
):
    """Handle streaming for regular chat messages."""
    try:
        logger.info(f"Starting chat stream for {connection_id}")

        await safe_send_json(websocket, {
            "type": "stream_start",
            "message": "Processing your message..."
        }, connection_id)

        config = {"configurable": {"thread_id": thread_id}}
        from langchain_core.messages import HumanMessage

        # Prepare input data
        if thread_id not in teacher_service.sessions:
            logger.info("Initializing new session state")
            initial_state = teacher_service.workflow.initialize_state()
            initial_state["messages"].append(HumanMessage(content=user_message))
            input_data = initial_state
        else:
            logger.info("Using existing session")
            input_data = {"messages": [HumanMessage(content=user_message)]}

        # Stream execution
        has_interrupt = False
        final_result = None
        chunk_count = 0

        async for chunk in teacher_service.workflow.graph.astream(
            input_data,
            config=config,
            stream_mode="values"
        ):
            chunk_count += 1
            
            if "__interrupt__" in chunk:
                has_interrupt = True
                final_result = chunk
                logger.info(f"Interrupt detected at chunk {chunk_count}")
            else:
                # Send progress update
                await safe_send_json(websocket, {
                    "type": "progress",
                    "stage": chunk.get("current_stage", "processing"),
                    "messages_count": len(chunk.get("messages", []))
                }, connection_id)
                final_result = chunk

        # Get final state
        final_state = teacher_service.workflow.graph.get_state(config)
        result = final_state.values if final_state else final_result

        if not has_interrupt and result and "__interrupt__" in result:
            has_interrupt = True
            logger.info("Interrupt detected in final state")

        # Send appropriate response
        if has_interrupt:
            await send_interrupt_response(websocket, result, thread_id, connection_id)
            teacher_service.sessions.add(thread_id)
        else:
            await send_final_response(websocket, result, thread_id, connection_id)
            teacher_service.sessions.add(thread_id)

        await safe_send_json(websocket, {"type": "stream_end"}, connection_id)

    except Exception as e:
        logger.error(f"Error in chat stream: {str(e)}", exc_info=True)
        await safe_send_json(websocket, {
            "type": "error",
            "message": format_learner_error(e),
            "technical_details": str(e) if logger.level <= logging.DEBUG else None
        }, connection_id)


async def handle_resume_stream(
    websocket: WebSocket,
    teacher_service,
    thread_id: str,
    answer: str,
    connection_id: str
):
    """Handle streaming for resume (after interrupt)."""
    try:
        logger.info(f"Starting resume stream for {connection_id}")

        await safe_send_json(websocket, {
            "type": "stream_start",
            "message": "Processing your answer..."
        }, connection_id)

        config = {"configurable": {"thread_id": thread_id}}
        has_interrupt = False
        final_result = None

        async for chunk in teacher_service.workflow.graph.astream(
            Command(resume=answer),
            config=config,
            stream_mode="values"
        ):
            if "__interrupt__" in chunk:
                has_interrupt = True
                final_result = chunk
                logger.info("Another interrupt during resume")
            else:
                await safe_send_json(websocket, {
                    "type": "progress",
                    "stage": chunk.get("current_stage", "processing"),
                    "messages_count": len(chunk.get("messages", []))
                }, connection_id)
                final_result = chunk

        final_state = teacher_service.workflow.graph.get_state(config)
        result = final_state.values if final_state else final_result

        if not has_interrupt and result and "__interrupt__" in result:
            has_interrupt = True

        if has_interrupt:
            await send_interrupt_response(websocket, result, thread_id, connection_id)
        else:
            await send_final_response(websocket, result, thread_id, connection_id)
            teacher_service.sessions.add(thread_id)

        await safe_send_json(websocket, {"type": "stream_end"}, connection_id)

    except Exception as e:
        logger.error(f"Error in resume stream: {str(e)}", exc_info=True)
        await safe_send_json(websocket, {
            "type": "error",
            "message": format_learner_error(e),
            "technical_details": str(e) if logger.level <= logging.DEBUG else None
        }, connection_id)


async def send_interrupt_response(websocket: WebSocket, result: dict, thread_id: str, connection_id: str):
    """Send interrupt notification to client."""
    messages = result.get("messages", [])
    
    ai_message = None
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'ai':
            ai_message = msg.content
            break
    
    if not ai_message:
        ai_message = "Please provide your information"
    
    interrupts = result.get("__interrupt__", [])
    interrupt_value = interrupts[0].value if interrupts else "user_input_required"

    await safe_send_json(websocket, {
        "type": "interrupt",
        "message": ai_message,
        "interrupt_id": interrupt_value,
        "stage": "awaiting_user_input",
        "thread_id": thread_id,
        "current_slide_index": result.get("current_slide_index", 0),
        "total_slides": len(result.get("slides", []))
    }, connection_id)


async def send_final_response(websocket: WebSocket, result: dict, thread_id: str, connection_id: str):
    """Send final response to client."""
    messages = result.get("messages", [])
    ai_message = ""
    
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'ai':
            ai_message = msg.content
            break

    current_slide_index = result.get("current_slide_index", 0)
    slides = result.get("slides", [])

    slide_data = None
    if slides and current_slide_index < len(slides):
        slide_data = slides[current_slide_index]

    await safe_send_json(websocket, {
        "type": "response",
        "message": ai_message,
        "slide": slide_data,
        "thread_id": thread_id,
        "current_stage": result.get("current_stage", "teaching"),
        "current_topic": result.get("current_topic"),
        "topics_covered": result.get("topics_covered", []),
        "current_slide_index": current_slide_index,
        "total_slides": len(slides)
    }, connection_id)