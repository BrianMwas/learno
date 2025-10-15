"""
Test WebSocket client for streaming chat - Natural conversation flow.
Run this to test the WebSocket streaming functionality.
"""
import asyncio
import websockets
import json


async def test_websocket_chat():
    """Test the WebSocket chat streaming with natural interrupt handling."""
    thread_id = "test_ws_natural_flow_v3"
    uri = f"ws://localhost:8000/api/v1/ws/chat/{thread_id}"

    print(f"\n{'='*80}")
    print("WEBSOCKET STREAMING TEST - NATURAL CONVERSATION FLOW")
    print(f"{'='*80}\n")

    async with websockets.connect(uri) as websocket:
        print("✓ Connected to WebSocket\n")

        # Step 0: Send initial message to trigger Meemo's greeting (no user input yet)
        print("--- Step 0: Triggering initial greeting ---")
        await websocket.send(json.dumps({
            "type": "message",
            "content": "(start)"  # Trigger initial greeting
        }))

        # Receive initial greeting (should NOT have an interrupt yet)
        initial_greeting_received = False
        
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                message_type = data.get("type")

                print(f"  [{message_type.upper()}]", end="")

                if message_type == "stream_start":
                    print(f" {data.get('message')}")
                
                elif message_type == "progress":
                    stage = data.get("stage")
                    print(f" Stage: {stage}")

                elif message_type == "response":
                    greeting = data.get("message")
                    print(f"\n\n👋 MEEMO'S GREETING:")
                    print(f"   {greeting[:200]}...")
                    initial_greeting_received = True

                elif message_type == "interrupt":
                    print(f"\n\n❌ UNEXPECTED INTERRUPT!")
                    print("   (Should not interrupt on initial greeting)")
                    return False

                elif message_type == "stream_end":
                    print("\n✓ Stream ended")
                    break

                elif message_type == "error":
                    print(f"\n❌ Error: {data.get('message')}")
                    return False

            except websockets.exceptions.ConnectionClosed:
                print("\n⚠️  Connection closed")
                break

        if not initial_greeting_received:
            print("\n❌ FAILED: Did not receive initial greeting!")
            return False
        
        print("\n✅ Initial greeting received (no interrupt - waiting for user response)")

        # Step 1: User responds with "Hello" - NOW it should ask for name and interrupt
        print("\n--- Step 1: User says 'Hello' ---")
        await websocket.send(json.dumps({
            "type": "message",
            "content": "Hello"
        }))

        # Should receive interrupt asking for name
        name_interrupt_received = False
        
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                message_type = data.get("type")

                print(f"  [{message_type.upper()}]", end="")

                if message_type == "stream_start":
                    print(f" {data.get('message')}")
                
                elif message_type == "progress":
                    stage = data.get("stage")
                    print(f" Stage: {stage}")

                elif message_type == "interrupt":
                    interrupt_message = data.get("message")
                    interrupt_id = data.get("interrupt_id")
                    print(f"\n\n🛑 NAME INTERRUPT RECEIVED!")
                    print(f"   ID: {interrupt_id}")
                    print(f"   Message: {interrupt_message[:200]}...")
                    name_interrupt_received = True

                elif message_type == "stream_end":
                    print("\n✓ Stream ended")
                    break

                elif message_type == "error":
                    print(f"\n❌ Error: {data.get('message')}")
                    return False

            except websockets.exceptions.ConnectionClosed:
                print("\n⚠️  Connection closed")
                break

        if not name_interrupt_received:
            print("\n❌ FAILED: Expected interrupt for name but didn't receive one!")
            return False
        
        print("\n✅ Name interrupt received correctly")

        # Step 2: Resume with name - should ask for goal and interrupt
        print("\n--- Step 2: Providing name 'Alice' ---")
        await websocket.send(json.dumps({
            "type": "resume",
            "answer": "Alice"
        }))

        # Should receive interrupt asking for learning goal
        goal_interrupt_received = False
        
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                message_type = data.get("type")

                print(f"  [{message_type.upper()}]", end="")

                if message_type == "stream_start":
                    print(f" {data.get('message')}")
                
                elif message_type == "progress":
                    stage = data.get("stage")
                    print(f" Stage: {stage}")

                elif message_type == "interrupt":
                    interrupt_message = data.get("message")
                    interrupt_id = data.get("interrupt_id")
                    print(f"\n\n🛑 GOAL INTERRUPT RECEIVED!")
                    print(f"   ID: {interrupt_id}")
                    print(f"   Message: {interrupt_message[:200]}...")
                    goal_interrupt_received = True

                elif message_type == "stream_end":
                    print("\n✓ Stream ended")
                    break

                elif message_type == "error":
                    print(f"\n❌ Error: {data.get('message')}")
                    return False

            except websockets.exceptions.ConnectionClosed:
                print("\n⚠️  Connection closed")
                break

        if not goal_interrupt_received:
            print("\n❌ FAILED: Expected interrupt for learning goal but didn't receive one!")
            return False
        
        print("\n✅ Goal interrupt received correctly")

        # Step 3: Resume with learning goal - should get final welcome and start teaching
        print("\n--- Step 3: Providing goal 'I want to learn about cells' ---")
        await websocket.send(json.dumps({
            "type": "resume",
            "answer": "I want to learn about cells"
        }))

        # Should receive final welcome and teaching should start
        final_response_received = False
        final_stage = None
        
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                message_type = data.get("type")

                print(f"  [{message_type.upper()}]", end="")

                if message_type == "stream_start":
                    print(f" {data.get('message')}")
                
                elif message_type == "progress":
                    stage = data.get("stage")
                    print(f" Stage: {stage}")

                elif message_type == "response":
                    final_message = data.get("message")
                    final_stage = data.get("current_stage")
                    topic = data.get("current_topic")
                    total_slides = data.get("total_slides")
                    
                    print(f"\n\n📝 FINAL WELCOME & TEACHING START!")
                    print(f"   Stage: {final_stage}")
                    print(f"   Topic: {topic}")
                    print(f"   Total Slides: {total_slides}")
                    print(f"   Message: {final_message[:150]}...")
                    final_response_received = True

                elif message_type == "interrupt":
                    print(f"\n\n❌ UNEXPECTED INTERRUPT!")
                    print("   (Should not interrupt after goal - should start teaching)")
                    return False

                elif message_type == "stream_end":
                    print("\n✓ Stream ended")
                    break

                elif message_type == "error":
                    print(f"\n❌ Error: {data.get('message')}")
                    return False

            except websockets.exceptions.ConnectionClosed:
                print("\n⚠️  Connection closed")
                break

        if not final_response_received:
            print("\n❌ FAILED: Did not receive final welcome response!")
            return False
        
        if final_stage != "teaching":
            print(f"\n❌ FAILED: Expected stage 'teaching' but got '{final_stage}'")
            return False

        print("\n✅ Final welcome received and teaching started!")
        
        # Summary
        print("\n" + "="*80)
        print("🎉 SUCCESS! Natural conversation flow works perfectly!")
        print("="*80)
        print("\n📊 COMPLETE FLOW:")
        print("   0. ✓ Connected → Meemo greeted (no interrupt)")
        print("   1. ✓ Said 'Hello' → Meemo asked for name (INTERRUPT)")
        print("   2. ✓ Provided 'Alice' → Meemo asked for goal (INTERRUPT)")
        print("   3. ✓ Provided goal → Welcome message + Teaching started")
        print("\n💡 This creates a natural back-and-forth conversation!")
        
        return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_websocket_chat())
        if result:
            print("\n✅ ALL TESTS PASSED!")
        else:
            print("\n❌ TESTS FAILED")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()