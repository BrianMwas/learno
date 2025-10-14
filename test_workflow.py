"""
Test script to debug the AI teacher workflow
Run this from your project root: python -m test_workflow
"""
import asyncio
import logging
from app.services.ai_teacher import get_teacher_service

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_introduction_flow():
    """Test the complete introduction flow with interrupts."""
    print("\n" + "="*80)
    print("TESTING INTRODUCTION FLOW")
    print("="*80 + "\n")

    teacher = get_teacher_service()
    thread_id = "test-intro-flow"

    # Step 1: Initial greeting
    print("\n--- Step 1: Initial Greeting ---")
    try:
        message, slide, tid, stage = await teacher.chat("Hi there!", thread_id)
        print(f"Stage: {stage}")
        print(f"Message: {message[:200]}")
        print(f"Slide title: {slide.title}")
        print(f"Slide content: {slide.content[:200]}")

        if stage != "awaiting_user_input":
            print("❌ ERROR: Expected 'awaiting_user_input' stage for name collection")
            return False
        else:
            print("✅ Correctly waiting for user name")

    except Exception as e:
        print(f"❌ ERROR in step 1: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Step 2: Provide name
    print("\n--- Step 2: Providing Name ---")
    try:
        message, slide, tid, stage = teacher.resume_with_answer(thread_id, "Alice")
        print(f"Stage: {stage}")
        print(f"Message: {message[:200]}")

        if stage != "awaiting_user_input":
            print("❌ ERROR: Expected 'awaiting_user_input' stage for learning goal collection")
            return False
        else:
            print("✅ Correctly waiting for learning goal")

    except Exception as e:
        print(f"❌ ERROR in step 2: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Provide learning goal
    print("\n--- Step 3: Providing Learning Goal ---")
    try:
        message, slide, tid, stage = teacher.resume_with_answer(thread_id, "I want to understand cells better")
        print(f"Stage: {stage}")
        print(f"Message: {message[:200]}")
        print(f"Slide title: {slide.title}")

        if stage == "awaiting_user_input":
            print("❌ ERROR: Should not be awaiting input anymore")
            return False
        elif stage == "teaching":
            print("✅ Introduction complete, moved to teaching!")
        else:
            print(f"⚠️  Unexpected stage: {stage}")

        # Check session info
        session_info = teacher.get_session_info(thread_id)
        print(f"\nSession info: {session_info}")

        return True

    except Exception as e:
        print(f"❌ ERROR in step 3: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_chat():
    """Test a simple chat without the full flow to debug."""
    print("\n" + "="*80)
    print("TESTING SIMPLE CHAT")
    print("="*80 + "\n")

    teacher = get_teacher_service()
    thread_id = "test-simple"

    try:
        # Enable more verbose logging
        import logging
        logging.getLogger("app.services.learning_workflow").setLevel(logging.DEBUG)

        print("Sending message: 'Hello'")
        message, slide, tid, stage = await teacher.chat("Hello", thread_id)

        print(f"\n--- Response ---")
        print(f"Stage: {stage}")
        print(f"Thread ID: {tid}")
        print(f"Message: {message}")
        print(f"Slide: {slide}")

        # Check if graph actually executed
        session_info = teacher.get_session_info(thread_id)
        print(f"\nSession Info: {session_info}")

        return True

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 Starting AI Teacher Workflow Tests\n")

    # Run simple chat test first
    asyncio.run(test_simple_chat())

    # Then run full introduction flow
    # asyncio.run(test_introduction_flow())
