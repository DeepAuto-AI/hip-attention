import os

import httpx
from openai import OpenAI

port = os.getenv("SRT_PORT", "8913")

client = OpenAI(
    base_url=f"http://localhost:{port}/v1",
    api_key="API_KEY",
    http_client=httpx.Client(timeout=60 * 60),  # 60 min timeout
)
chat_completion = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    # This is around 1_000_040 tokens
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What are some highly rated restaurants in San Francisco? "
            * 100_000,
        },
    ],
    temperature=0.01,
    stream=True,
    max_tokens=10000,
)

for chat in chat_completion:
    if chat.choices[0].delta.content is not None:
        print(chat.choices[0].delta.content, end="")
