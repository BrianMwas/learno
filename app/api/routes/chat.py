from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse, SessionInfoResponse, SlideNavigationResponse, SlideContent
from app.services.ai_teacher import get_teacher_service
import json

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message through the learning workflow.
    Returns AI response with slide content and learning state.
    """
    try:
        teacher_service = get_teacher_service()
        ai_message, slide, thread_id, current_stage = await teacher_service.chat(
            user_message=request.message,
            thread_id=request.thread_id,
            user_name=request.user_name
        )

        # Get session info for additional context
        session_info = teacher_service.get_session_info(thread_id)

        return ChatResponse(
            message=ai_message,
            slide=slide,
            thread_id=thread_id,
            current_stage=current_stage,
            current_topic=session_info.get("current_topic"),
            topics_covered=session_info.get("topics_covered", []),
            current_slide_index=session_info.get("current_slide_index", 0),
            total_slides=session_info.get("total_slides", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{thread_id}", response_model=SessionInfoResponse)
async def get_session_info(thread_id: str):
    """
    Get information about a learning session.
    """
    try:
        teacher_service = get_teacher_service()
        session_info = teacher_service.get_session_info(thread_id)

        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionInfoResponse(**session_info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/slides/navigate/{thread_id}/{direction}", response_model=SlideNavigationResponse)
async def navigate_slide(thread_id: str, direction: str):
    """
    Navigate to next or previous slide.
    Direction should be 'next' or 'previous'.
    """
    if direction not in ["next", "previous"]:
        raise HTTPException(status_code=400, detail="Direction must be 'next' or 'previous'")

    try:
        teacher_service = get_teacher_service()
        slide_data = teacher_service.navigate_slide(thread_id, direction)

        if not slide_data:
            raise HTTPException(status_code=404, detail="Cannot navigate in that direction or session not found")

        session_info = teacher_service.get_session_info(thread_id)

        slide = SlideContent(
            title=slide_data.get("title", ""),
            content=slide_data.get("content", ""),
            code_example=slide_data.get("code_example"),
            visual_description=slide_data.get("visual_description", "")
        )

        return SlideNavigationResponse(
            slide=slide,
            slide_index=session_info.get("current_slide_index", 0),
            total_slides=session_info.get("total_slides", 0)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/slides/{thread_id}")
async def get_all_slides(thread_id: str):
    """
    Get all slides for a learning session.
    """
    try:
        teacher_service = get_teacher_service()
        slides = teacher_service.get_all_slides(thread_id)

        if not slides:
            raise HTTPException(status_code=404, detail="No slides found for this session")

        return {"slides": slides, "total": len(slides)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response (for future implementation).
    Currently returns the same as non-streaming.
    """
    try:
        teacher_service = get_teacher_service()
        ai_message, slide, thread_id, current_stage = await teacher_service.chat(
            user_message=request.message,
            thread_id=request.thread_id,
            user_name=request.user_name
        )

        session_info = teacher_service.get_session_info(thread_id)

        response_data = {
            "message": ai_message,
            "slide": slide.dict(),
            "thread_id": thread_id,
            "current_stage": current_stage,
            "current_topic": session_info.get("current_topic"),
            "topics_covered": session_info.get("topics_covered", []),
            "current_slide_index": session_info.get("current_slide_index", 0),
            "total_slides": session_info.get("total_slides", 0)
        }

        async def generate():
            yield json.dumps(response_data)

        return StreamingResponse(generate(), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
