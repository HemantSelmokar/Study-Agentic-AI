import os
from dotenv import load_dotenv, find_dotenv

from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands.models.openai import OpenAIModel
from strands_tools import calculator, current_time

load_dotenv(find_dotenv(usecwd=True))

# ---------------------------------------------------------------------------
# Model provider toggle — set MODEL_PROVIDER=openai or MODEL_PROVIDER=ollama
# in the shared .env file (c:\AI Agent study\.env) to switch models.
# ---------------------------------------------------------------------------
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
OPENAI_MODEL_ID = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-nano")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_LLM_MODEL", "gemma4:31b-cloud")
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if MODEL_PROVIDER == "openai":
    model = OpenAIModel(
        client_args={"api_key": os.getenv("OPENAI_API_KEY", "")},
        model_id=OPENAI_MODEL_ID,
    )
else:
    # Cloud-routed Ollama models fail with streaming; use additional_args to disable it
    model = OllamaModel(
        model_id=OLLAMA_MODEL_ID,
        host=OLLAMA_HOST,
        additional_args={"stream": False},
    )


# Define a custom tool as a Python function using the @tool decorator
@tool
def letter_counter(word: str, letter: str) -> int:
    """
    Count occurrences of a specific letter in a word.

    Args:
        word (str): The input word to search in
        letter (str): The specific letter to count

    Returns:
        int: The number of occurrences of the letter in the word
    """
    if not isinstance(word, str) or not isinstance(letter, str):
        return 0

    if len(letter) != 1:
        raise ValueError("The 'letter' parameter must be a single character")

    return word.lower().count(letter.lower())

# Create an agent with the selected model (OpenAI or Ollama, per MODEL_PROVIDER) and tools
agent = Agent(
    model=model,
    tools=[calculator, current_time, letter_counter],
)

# Interactive terminal loop — type your question and press Enter.
# Type 'exit' or 'quit' to stop the program.
print("\n🤖 Agent is ready! Ask me anything. Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nGoodbye! 👋")
        break

    if not user_input:
        continue

    if user_input.lower() in ("exit", "quit"):
        print("Goodbye! 👋")
        break

    agent(user_input)
    print()  # blank line for readability between turns