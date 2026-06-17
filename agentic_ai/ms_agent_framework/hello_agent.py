from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

from dotenv import load_dotenv
load_dotenv()

credential = AzureCliCredential()
client = FoundryChatClient(
    project_endpoint="https://your-foundry-service.services.ai.azure.com/api/projects/your-foundry-project",
    model="gpt-5.4-mini",
    credential=credential,
)

agent = client.as_agent(
    name="HelloAgent",
    instructions="You are a friendly assistant. Keep your answers brief.",
)

# Non-streaming: get the complete response at once
result = await agent.run("What is the largest city in France?")
print(f"Agent: {result}")

# Streaming: receive tokens as they are generated
print("Agent (streaming): ", end="", flush=True)
async for chunk in agent.run("Tell me a one-sentence fun fact.", stream=True):
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()

# NOTE: approval_mode="never_require" is for sample brevity.
# Use "always_require" in production for user confirmation before tool execution.
@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."

agent = Agent(
    client=client,
    name="WeatherAgent",
    instructions="You are a helpful weather agent. Use the get_weather tool to answer questions.",
    tools=[get_weather],
)


# Use a session to maintain conversation context so the agent remembers what was said earlier.
# Use AgentSession to maintain context across multiple calls:
agent = Agent(
    client=client,
    name="ConversationAgent",
    instructions="You are a friendly assistant. Keep your answers brief.",
)

# Create a session to maintain conversation history
session = agent.create_session()

# First turn
result = await agent.run("My name is Alice and I love hiking.", session=session)
print(f"Agent: {result}\n")

# Second turn — the agent should remember the user's name and hobby
result = await agent.run("What do you remember about me?", session=session)
print(f"Agent: {result}")

# Add context to your agent so it can remember user preferences, past interactions, or external knowledge.
# Define a context provider that stores user info in session state and injects personalization instructions:
class UserMemoryProvider(ContextProvider):
    """A context provider that remembers user info in session state."""

    DEFAULT_SOURCE_ID = "user_memory"

    def __init__(self):
        super().__init__(self.DEFAULT_SOURCE_ID)

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Inject personalization instructions based on stored user info."""
        user_name = state.get("user_name")
        if user_name:
            context.extend_instructions(
                self.source_id,
                f"The user's name is {user_name}. Always address them by name.",
            )
        else:
            context.extend_instructions(
                self.source_id,
                "You don't know the user's name yet. Ask for it politely.",
            )

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Extract and store user info in session state after each call."""
        for msg in context.input_messages:
            text = msg.text if hasattr(msg, "text") else ""
            if isinstance(text, str) and "my name is" in text.lower():
                state["user_name"] = text.lower().split("my name is")[-1].strip().split()[0].capitalize()

# Create an agent with the context provider:
agent = Agent(
    client=client,
    name="MemoryAgent",
    instructions="You are a friendly assistant.",
    context_providers=[UserMemoryProvider()],
)

# Run it — the agent now has access to the context:
session = agent.create_session()

# The provider doesn't know the user yet — it will ask for a name
result = await agent.run("Hello! What's the square root of 9?", session=session)
print(f"Agent: {result}\n")

# Now provide the name — the provider stores it in session state
result = await agent.run("My name is Alice", session=session)
print(f"Agent: {result}\n")

# Subsequent calls are personalized — name persists via session state
result = await agent.run("What is 2 + 2?", session=session)
print(f"Agent: {result}\n")

# Inspect session state to see what the provider stored
provider_state = session.state.get("user_memory", {})
print(f"[Session State] Stored user name: {provider_state.get('user_name')}")

