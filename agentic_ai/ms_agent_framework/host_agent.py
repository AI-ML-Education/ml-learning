from typing import Any

from agent_framework import Agent
from agent_framework.azure import AgentFunctionApp
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()


# 1. Instantiate the agent with the chosen deployment and instructions.
def _create_agent() -> Any:
    """Create the Joker agent."""
    return Agent(
        client=FoundryChatClient(
            project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            model=os.environ["FOUNDRY_MODEL"],
            credential=AzureCliCredential(),
        ),
        name="Joker",
        instructions="You are good at telling jokes.",
    )


# 2. Register the agent with AgentFunctionApp so Azure Functions exposes the required triggers.
app = AgentFunctionApp(agents=[_create_agent()], enable_health_check=True, max_poll_retries=50)

"""
Expected output when invoking `POST /api/agents/Joker/run` with plain-text input:

HTTP/1.1 202 Accepted
{
  "status": "accepted",
  "response": "Agent request accepted",
  "message": "Tell me a short joke about cloud computing.",
  "conversation_id": "<guid>",
  "correlation_id": "<guid>"
}
"""

# Run Locally using Azure Functions Core Tools:
"""
`func start`

then invoke:

`curl -X POST http://localhost:7071/api/agents/Joker/run \
  -H "Content-Type: text/plain" \
  -d "Tell me a short joke about cloud computing."`

"""