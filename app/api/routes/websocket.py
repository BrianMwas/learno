"""
WebSocket route for streaming chat responses.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.ai_teacher import get_teacher_service
from langgraph.types import Command
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/chat/{thread_id}")
async def websocket_chat(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for streaming chat with the learning workflow.

    Args:
        websocket: WebSocket connection
        thread_id: Session/thread ID for conversation continuity

    Flow:
        1. Client connects
        2. Client sends: {"type": "message", "content": "user message"}
           OR {"type": "resume", "answer": "user answer"}
        3. Server streams back updates in real-time
        4. Server sends final response when done
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for thread_id: {thread_id}")

    try:
        teacher_service = get_teacher_service()

        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            message_type = message_data.get("type")
            logger.info(f"Received WebSocket message type: {message_type}")

            if message_type == "message":
                # Regular chat message
                user_message = message_data.get("content")
                if not user_message:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Missing content in message"
                    })
                    continue

                await handle_chat_stream(websocket, teacher_service, thread_id, user_message)

            elif message_type == "resume":
                # Resume from interrupt with answer
                answer = message_data.get("answer")
                if not answer:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Missing answer in resume"
                    })
                    continue

                await handle_resume_stream(websocket, teacher_service, thread_id, answer)

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for thread_id: {thread_id}")
    except Exception as e:
        logger.error(f"WebSocket error for thread_id {thread_id}: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

async def handle_chat_stream(websocket: WebSocket, teacher_service, thread_id: str, user_message: str):
    """
    Handle streaming for regular chat messages.
    """
    try:
        logger.info(f"Starting chat stream for thread_id: {thread_id}")

        # Send initial status
        await websocket.send_json({
            "type": "stream_start",
            "message": "Processing your message..."
        })

        # Get config for checkpointer
        config = {"configurable": {"thread_id": thread_id}}

        # Prepare input data
        from langchain_core.messages import HumanMessage

        if thread_id not in teacher_service.sessions:
            logger.info(f"New session - initializing state")
            initial_state = teacher_service.workflow.initialize_state()
            initial_state["messages"].append(HumanMessage(content=user_message))
            input_data = initial_state
        else:
            logger.info(f"Existing session")
            input_data = {"messages": [HumanMessage(content=user_message)]}

        # Stream the graph execution
        has_interrupt = False
        final_result = None

        try:
            logger.info("Starting stream iteration...")
            chunk_count = 0
            
            # CRITICAL FIX: Use stream_mode="values" only (not a list)
            # This ensures we get the full state values including __interrupt__
            async for chunk in teacher_service.workflow.graph.astream(
                input_data,
                config=config,
                stream_mode="values"  # ← KEY FIX: single mode, not list
            ):
                chunk_count += 1
                logger.info(f"Chunk {chunk_count}: keys={list(chunk.keys()) if isinstance(chunk, dict) else 'N/A'}")

                # Check for interrupt in this chunk
                if "__interrupt__" in chunk:
                    has_interrupt = True
                    final_result = chunk
                    logger.info(f"🔴 INTERRUPT DETECTED in chunk {chunk_count}")
                    logger.info(f"Interrupt data: {chunk['__interrupt__']}")
                    # Don't break - let it finish naturally
                else:
                    # Send progress update to client
                    await websocket.send_json({
                        "type": "progress",
                        "stage": chunk.get("current_stage", "processing"),
                        "messages_count": len(chunk.get("messages", []))
                    })
                    final_result = chunk

        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}", exc_info=True)
            raise

        # After streaming completes, get final state
        final_state = teacher_service.workflow.graph.get_state(config)
        result = final_state.values if final_state else final_result

        # Double-check for interrupt in final state
        if not has_interrupt and result and "__interrupt__" in result:
            has_interrupt = True
            logger.info("🔴 INTERRUPT DETECTED in final state")

        logger.info(f"Stream complete - has_interrupt: {has_interrupt}")
        logger.info(f"Final state keys: {list(result.keys()) if result else 'None'}")

        if has_interrupt:
            # Extract interrupt information
            interrupts = result.get("__interrupt__", [])
            interrupt_value = interrupts[0].value if interrupts else "Please provide your information"

            # CRITICAL: Get the actual AI message that was just added before the interrupt
            # The last message in the conversation is what Meemo actually said
            messages = result.get("messages", [])
            ai_prompt = messages[-1].content if messages else interrupt_value

            logger.info(f"Sending interrupt to client")
            logger.info(f"Interrupt reason: {interrupt_value}")
            logger.info(f"AI prompt: {ai_prompt[:100]}")

            teacher_service.sessions.add(thread_id)

            # Send interrupt notification with the AI's actual message
            await websocket.send_json({
                "type": "interrupt",
                "message": ai_prompt,  # ← Send the AI's message, not the interrupt value
                "interrupt_reason": interrupt_value,  # Include reason for debugging
                "stage": "awaiting_user_input",
                "thread_id": thread_id,
                "current_slide_index": result.get("current_slide_index", 0),
                "total_slides": len(result.get("slides", []))
            })
        else:
            # No interrupt - send final response
            teacher_service.sessions.add(thread_id)

            ai_message = result["messages"][-1].content if result.get("messages") else ""
            current_slide_index = result.get("current_slide_index", 0)
            slides = result.get("slides", [])

            # Get slide data
            if slides and current_slide_index < len(slides):
                slide_data = slides[current_slide_index]
            else:
                slide_data = None

            # Send final response
            await websocket.send_json({
                "type": "response",
                "message": ai_message,
                "slide": slide_data,
                "thread_id": thread_id,
                "current_stage": result.get("current_stage", "teaching"),
                "current_topic": result.get("current_topic"),
                "topics_covered": result.get("topics_covered", []),
                "current_slide_index": current_slide_index,
                "total_slides": len(slides)
            })

        # Send stream end
        await websocket.send_json({
            "type": "stream_end"
        })

    except Exception as e:
        logger.error(f"Error in chat stream: {str(e)}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })


async def handle_resume_stream(websocket: WebSocket, teacher_service, thread_id: str, answer: str):
    """
    Handle streaming for resume (after interrupt).
    """
    try:
        logger.info(f"Starting resume stream for thread_id: {thread_id}")

        # Send initial status
        await websocket.send_json({
            "type": "stream_start",
            "message": "Processing your answer..."
        })

        # Get config for checkpointer
        config = {"configurable": {"thread_id": thread_id}}

        # Stream the resume
        has_interrupt = False
        final_result = None

        try:
            chunk_count = 0
            
            # CRITICAL FIX: Use astream (async) and stream_mode="values" only
            async for chunk in teacher_service.workflow.graph.astream(
                Command(resume=answer),
                config=config,
                stream_mode="values"  # ← KEY FIX: single mode, not list
            ):
                chunk_count += 1
                logger.info(f"Resume chunk {chunk_count}: keys={list(chunk.keys()) if isinstance(chunk, dict) else 'N/A'}")

                # Check for another interrupt
                if "__interrupt__" in chunk:
                    has_interrupt = True
                    final_result = chunk
                    logger.info(f"🔴 ANOTHER INTERRUPT DETECTED in chunk {chunk_count}")
                    logger.info(f"Interrupt data: {chunk['__interrupt__']}")
                else:
                    # Send progress update
                    await websocket.send_json({
                        "type": "progress",
                        "stage": chunk.get("current_stage", "processing"),
                        "messages_count": len(chunk.get("messages", []))
                    })
                    final_result = chunk

        except Exception as e:
            logger.error(f"Error during resume streaming: {str(e)}", exc_info=True)
            raise

        # Get the final state
        final_state = teacher_service.workflow.graph.get_state(config)
        result = final_state.values if final_state else final_result

        # Double-check for interrupt in final state
        if not has_interrupt and result and "__interrupt__" in result:
            has_interrupt = True
            logger.info("🔴 INTERRUPT DETECTED in final state after resume")

        logger.info(f"Resume complete - has_interrupt: {has_interrupt}")

        if has_interrupt:
            # Another interrupt occurred
            interrupts = result.get("__interrupt__", [])
            interrupt_value = interrupts[0].value if interrupts else "Please provide your information"

            logger.info(f"Sending another interrupt to client: {interrupt_value}")

            # Send interrupt notification
            await websocket.send_json({
                "type": "interrupt",
                "message": interrupt_value,
                "stage": "awaiting_user_input",
                "thread_id": thread_id
            })
        else:
            # No interrupt - send final response
            teacher_service.sessions.add(thread_id)

            ai_message = result["messages"][-1].content if result.get("messages") else ""
            current_slide_index = result.get("current_slide_index", 0)
            slides = result.get("slides", [])

            # Get slide data
            if slides and current_slide_index < len(slides):
                slide_data = slides[current_slide_index]
            else:
                slide_data = None

            # Send final response
            await websocket.send_json({
                "type": "response",
                "message": ai_message,
                "slide": slide_data,
                "thread_id": thread_id,
                "current_stage": result.get("current_stage", "teaching"),
                "current_topic": result.get("current_topic"),
                "topics_covered": result.get("topics_covered", []),
                "current_slide_index": current_slide_index,
                "total_slides": len(slides)
            })

        # Send stream end
        await websocket.send_json({
            "type": "stream_end"
        })

    except Exception as e:
        logger.error(f"Error in resume stream: {str(e)}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })