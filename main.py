from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

checkpointer = InMemorySaver()

def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    user_name = config["configurable"].get("user_name")
    
    system_msg = f"You are a helpful assistant. Address the user as {user_name}. Also be friendly and open with a greeting where necessary."
    return [{"role": "system", "content": system_msg}] + state["messages"]

class WeatherResponse(BaseModel):
    conditions: str
    
def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
    
model = init_chat_model(
    "openai:gpt-4.1",
    temperature=0
)

agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt=prompt,
    checkpointer=checkpointer,
    response_format=WeatherResponse
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "and SF"}]},
    config={"configurable": {"user_name": "John Smith", "thread_id": "one"}}
)

# Get the structured response
final_message = response["structured_response"]
print(final_message.conditions)