"""Debug script to test interrupt behavior"""
from app.services.learning_workflow import LearningWorkflow
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt

workflow = LearningWorkflow()
state = workflow.initialize_state()
state["messages"].append(HumanMessage(content="Hello"))

config = {"configurable": {"thread_id": "debug_test"}}

print("\n=== Invoking graph ===")
try:
    result = workflow.graph.invoke(state, config=config)

    # Check if result has __interrupt__ key
    if "__interrupt__" in result:
        print(f"\n✓ Graph paused with interrupt!")
        print(f"Interrupts: {result['__interrupt__']}")
    else:
        print(f"\n✗ Graph completed WITHOUT interrupt")
        print(f"Result keys: {list(result.keys())}")
        print(f"Result stage: {result.get('current_stage')}")
        print(f"Result user_name: {result.get('user_name')}")
        print(f"Messages: {len(result.get('messages', []))}")
except GraphInterrupt as e:
    print(f"\n✓ GraphInterrupt exception raised!")
    print(f"Interrupt value: {e.interrupts[0].value if e.interrupts else 'None'}")
except Exception as e:
    print(f"\n✗ Different exception: {type(e).__name__}: {e}")
