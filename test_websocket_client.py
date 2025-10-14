"""
Test WebSocket client for streaming chat.
Run this to test the WebSocket streaming functionality.
"""
import asyncio
import websockets
import json


async def test_websocket_chat():
    """Test the WebSocket chat streaming."""
    thread_id = "test_ws_session"
    uri = f"ws://localhost:8000/api/v1/ws/chat/{thread_id}"

    print(f"\n{'='*80}")
    print("WEBSOCKET STREAMING TEST")
    print(f"{'='*80}\n")

    async with websockets.connect(uri) as websocket:
        print("✓ Connected to WebSocket\n")

        # Step 1: Send initial message
        print("--- Step 1: Sending 'Hello' ---")
        await websocket.send(json.dumps({
            "type": "message",
            "content": "Hello"
        }))

        # Receive and display all responses
        interrupt_received = False
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)

                message_type = data.get("type")
                print(f"\n[{message_type.upper()}]", end="")

                if message_type == "custom":
                    # Custom stream event from node
                    custom_data = data.get("data", {})
                    print(f" {custom_data}")

                elif message_type == "update":
                    # Node update
                    update_data = data.get("data", {})
                    print(f" Node update: {list(update_data.keys())}")

                elif message_type == "interrupt":
                    # Interrupt - waiting for user input
                    message = data.get("message")
                    print(f"\n\n🛑 INTERRUPT: {message}")
                    interrupt_received = True

                elif message_type == "response":
                    # Final response
                    message = data.get("message")
                    stage = data.get("current_stage")
                    print(f"\n\n📝 Response (stage={stage}):")
                    print(f"Message: {message[:200]}...")

                elif message_type == "stream_end":
                    print("\n\n✓ Stream ended")
                    break

                elif message_type == "error":
                    print(f"\n\n❌ Error: {data.get('message')}")
                    break

            except websockets.exceptions.ConnectionClosed:
                print("\n\n⚠️  Connection closed")
                break

        if not interrupt_received:
            print("\n❌ Expected interrupt but didn't receive one!")
            return False

        # Step 2: Resume with name
        print("\n\n--- Step 2: Resuming with name 'Alice' ---")
        await websocket.send(json.dumps({
            "type": "resume",
            "answer": "Alice"
        }))

        # Receive and display all responses
        second_interrupt = False
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)

                message_type = data.get("type")
                print(f"\n[{message_type.upper()}]", end="")

                if message_type == "custom":
                    custom_data = data.get("data", {})
                    print(f" {custom_data}")

                elif message_type == "update":
                    update_data = data.get("data", {})
                    print(f" Node update: {list(update_data.keys())}")

                elif message_type == "interrupt":
                    message = data.get("message")
                    print(f"\n\n🛑 INTERRUPT: {message}")
                    second_interrupt = True

                elif message_type == "response":
                    message = data.get("message")
                    stage = data.get("current_stage")
                    topic = data.get("current_topic")
                    print(f"\n\n📝 Response (stage={stage}, topic={topic}):")
                    print(f"Message: {message[:200]}...")

                elif message_type == "stream_end":
                    print("\n\n✓ Stream ended")
                    break

                elif message_type == "error":
                    print(f"\n\n❌ Error: {data.get('message')}")
                    break

            except websockets.exceptions.ConnectionClosed:
                print("\n\n⚠️  Connection closed")
                break

        if not second_interrupt:
            print("\n❌ Expected second interrupt but didn't receive one!")
            return False

        # Step 3: Resume with learning goal
        print("\n\n--- Step 3: Resuming with goal 'I want to learn about cells' ---")
        await websocket.send(json.dumps({
            "type": "resume",
            "answer": "I want to learn about cells"
        }))

        # Receive and display all responses
        final_response_received = False
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)

                message_type = data.get("type")
                print(f"\n[{message_type.upper()}]", end="")

                if message_type == "custom":
                    custom_data = data.get("data", {})
                    print(f" {custom_data}")

                elif message_type == "update":
                    update_data = data.get("data", {})
                    print(f" Node update: {list(update_data.keys())}")

                elif message_type == "response":
                    message = data.get("message")
                    stage = data.get("current_stage")
                    topic = data.get("current_topic")
                    total_slides = data.get("total_slides")
                    print(f"\n\n📝 Final Response:")
                    print(f"  Stage: {stage}")
                    print(f"  Topic: {topic}")
                    print(f"  Total Slides: {total_slides}")
                    print(f"  Message: {message[:200]}...")
                    final_response_received = True

                elif message_type == "stream_end":
                    print("\n\n✓ Stream ended")
                    break

                elif message_type == "error":
                    print(f"\n\n❌ Error: {data.get('message')}")
                    break

            except websockets.exceptions.ConnectionClosed:
                print("\n\n⚠️  Connection closed")
                break

        if final_response_received:
            print("\n\n🎉 SUCCESS! WebSocket streaming works with interrupts!")
            return True
        else:
            print("\n\n❌ Did not receive final response")
            return False


if __name__ == "__main__":
    try:
        result = asyncio.run(test_websocket_chat())
        if result:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
