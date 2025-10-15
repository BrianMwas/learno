"""Test streaming directly with the graph"""
from app.services.learning_workflow import LearningWorkflow
from langchain_core.messages import HumanMessage

workflow = LearningWorkflow()
state = workflow.initialize_state()
state["messages"].append(HumanMessage(content="Hello"))

config = {"configurable": {"thread_id": "test_stream_direct"}}

print("\n=== Testing stream modes ===\n")

# Test with values mode
print("1. Streaming with mode='values':")
for i, chunk in enumerate(workflow.graph.stream(state, config=config, stream_mode="values")):
    print(f"   Chunk {i}: keys={list(chunk.keys())}, has __interrupt__={'__interrupt__' in chunk}")
    if "__interrupt__" in chunk:
        print(f"   🛑 INTERRUPT FOUND: {chunk['__interrupt__']}")
        break

print("\n2. Getting final state:")
final_state = workflow.graph.get_state(config)
print(f"   Has __interrupt__: {'__interrupt__' in final_state.values}")
if "__interrupt__" in final_state.values:
    print(f"   Interrupt value: {final_state.values['__interrupt__']}")
