# /// script
# dependencies = [
#   "langgraph==0.6.4",
#   "langchain==0.3.27",
#   "langchain-openai==0.3.30",
# ]
# ///

"""
How to use:

`.env` file:

```text
OPENAI_API_BASE=http://h100-80-1:8000/v1
OPENAI_API_KEY=sk-proj-1234567890
```

Running:

```bash
uv run --env-file .env scripts/test_agent.py
```
"""

import asyncio

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def write_file(file_path: str, content: str) -> str:
    """Write a file to the filesystem."""
    print("[test] `write_file` called! Content: ```", content, "```", sep="")
    return f"File {file_path} written successfully."


async def main():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        # model="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        # model="lmsys/gpt-oss-120b-bf16",
    )

    agent = create_react_agent(
        model=llm,
        tools=[write_file],
    )
    # ReAct loop: the agent decides when to call MCP tools vs reply directly.
    result = await agent.ainvoke(
        {"messages": "write a file called test.txt with the content 'Hello, world!'"},
    )

    # 'result' is the agent's state; last message is the model's final answer.
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
