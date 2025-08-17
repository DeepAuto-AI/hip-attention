# /// script
# dependencies = [
#   "openai==1.99.9",
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
uv run --env-file .env scripts/test_structured_output.py
```
"""

import os
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# 1) Define your schema
class Todo(BaseModel):
    title: str = Field(..., description="Short action item")
    due: Optional[str] = Field(None, description="ISO date YYYY-MM-DD if present")
    tags: List[str] = Field(default_factory=list, description="Zero or more tags")


client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE", None),
)

note = "Ship the SDK docs by 2025-08-20 and tag as docs, release."

# 2) Ask the model to produce data matching the schema
completion = client.chat.completions.parse(
    model="gpt-4o-mini",  # models listed to support Structured Outputs with chat
    messages=[
        {"role": "system", "content": "Extract a TODO item from the user note."},
        {"role": "user", "content": note},
    ],
    response_format=Todo,  # <<— pass the Pydantic class directly
    temperature=0.7,
)

# 3) Get a Pydantic instance back
todo: Todo = completion.choices[0].message.parsed
print(todo)
print(todo.model_dump())  # Python dict
