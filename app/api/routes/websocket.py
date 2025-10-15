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

        # For brand new sessions, automatically send Meemo's introduction
        if thread_id not in teacher_service.sessions:
            logger.info("🆕 New session - triggering Meemo's automatic introduction")
            # Check if there's any state in the checkpointer
            config = {"configurable": {"thread_id": thread_id}}
            existing_state = teacher_service.workflow.graph.get_state(config)

            if not existing_state or not existing_state.values or len(existing_state.values.get("slides", [])) == 0:
                logger.info("No existing slides - this is truly a fresh session")
                # Trigger the introduction without user input
                await handle_chat_stream(websocket, teacher_service, thread_id, "(start)")
        

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

        await websocket.send_json({
            "type": "stream_start",
            "message": "Processing your message..."
        })

        config = {"configurable": {"thread_id": thread_id}}

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
        chunk_count = 0

        try:
            logger.info("Starting stream iteration...")
            
            async for chunk in teacher_service.workflow.graph.astream(
                input_data,
                config=config,
                stream_mode="values"
            ):
                chunk_count += 1
                logger.info(f"Chunk {chunk_count}: keys={list(chunk.keys()) if isinstance(chunk, dict) else 'N/A'}")

                # Check for interrupt
                if "__interrupt__" in chunk:
                    has_interrupt = True
                    final_result = chunk
                    logger.info(f"🔴 INTERRUPT DETECTED in chunk {chunk_count}")
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
            logger.error(f"Error during streaming: {str(e)}", exc_info=True)
            raise

        # Get final state after streaming
        final_state = teacher_service.workflow.graph.get_state(config)
        result = final_state.values if final_state else final_result

        # Double-check for interrupt in final state
        if not has_interrupt and result and "__interrupt__" in result:
            has_interrupt = True
            logger.info("🔴 INTERRUPT DETECTED in final state")

        logger.info(f"Stream complete - has_interrupt: {has_interrupt}")

        if has_interrupt:
            # Extract the last AI message (what Meemo actually said to the user)
            messages = result.get("messages", [])
            
            # Find the last AI message
            ai_message = None
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'ai':
                    ai_message = msg.content
                    break
            
            if not ai_message:
                # Fallback if no AI message found
                ai_message = "Please provide your information"
            
            # Get interrupt metadata
            interrupts = result.get("__interrupt__", [])
            interrupt_value = interrupts[0].value if interrupts else "user_input_required"

            logger.info(f"Sending interrupt to client")
            logger.info(f"AI message: {ai_message[:100]}...")
            logger.info(f"Interrupt ID: {interrupt_value}")

            teacher_service.sessions.add(thread_id)

            # Send interrupt notification with the AI's actual message
            await websocket.send_json({
                "type": "interrupt",
                "message": ai_message,  # What Meemo said
                "interrupt_id": interrupt_value,  # Internal identifier
                "stage": "awaiting_user_input",
                "thread_id": thread_id,
                "current_slide_index": result.get("current_slide_index", 0),
                "total_slides": len(result.get("slides", []))
            })
        else:
            # No interrupt - send final response
            teacher_service.sessions.add(thread_id)

            messages = result.get("messages", [])
            ai_message = ""
            
            # Get last AI message
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'ai':
                    ai_message = msg.content
                    break

            current_slide_index = result.get("current_slide_index", 0)
            slides = result.get("slides", [])

            # Get slide data
            slide_data = None
            if slides and current_slide_index < len(slides):
                slide_data = slides[current_slide_index]

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
        logger.info(f"Starting resume stream for thread_id: {thread_id} with answer: {answer[:50]}")

        await websocket.send_json({
            "type": "stream_start",
            "message": "Processing your answer..."
        })

        config = {"configurable": {"thread_id": thread_id}}

        # Stream the resume
        has_interrupt = False
        final_result = None
        chunk_count = 0

        try:
            async for chunk in teacher_service.workflow.graph.astream(
                Command(resume=answer),
                config=config,
                stream_mode="values"
            ):
                chunk_count += 1
                logger.info(f"Resume chunk {chunk_count}: keys={list(chunk.keys()) if isinstance(chunk, dict) else 'N/A'}")

                # Check for another interrupt
                if "__interrupt__" in chunk:
                    has_interrupt = True
                    final_result = chunk
                    logger.info(f"🔴 ANOTHER INTERRUPT in chunk {chunk_count}")
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

        # Double-check for interrupt
        if not has_interrupt and result and "__interrupt__" in result:
            has_interrupt = True
            logger.info("🔴 INTERRUPT in final state after resume")

        logger.info(f"Resume complete - has_interrupt: {has_interrupt}")

        if has_interrupt:
            # Another interrupt occurred - extract AI message
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

            logger.info(f"Sending another interrupt: {ai_message[:100]}...")

            await websocket.send_json({
                "type": "interrupt",
                "message": ai_message,
                "interrupt_id": interrupt_value,
                "stage": "awaiting_user_input",
                "thread_id": thread_id,
                "current_slide_index": result.get("current_slide_index", 0),
                "total_slides": len(result.get("slides", []))
            })
        else:
            # No interrupt - send final response
            teacher_service.sessions.add(thread_id)

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

        await websocket.send_json({
            "type": "stream_end"
        })

    except Exception as e:
        logger.error(f"Error in resume stream: {str(e)}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })