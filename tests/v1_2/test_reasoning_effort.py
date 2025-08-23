import json

import requests

SGLANG_URL = "http://localhost:8000/v1/chat/completions"
PROMPT = "There are exactly three positive real numbers $k$ such that the function\[f(x)=\frac{(x-18)(x-72)(x-98)(x-k)}{x}.\]defined over the positive real numbers achieves its minimum value at exactly two positive real numbers $x$. Find the sum of these three values of $k$. Please explain your reasoning step-by-step."


# --- API Request Function ---
def get_chat_completion(prompt, reasoning_effort=None):
    headers = {"Content-Type": "application/json"}

    # Base payload for the chat completion request
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",  # Specify the model you are using
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that thinks carefully.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 65536,
    }

    # Add the reasoning_effort parameter to the payload if it's provided
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    print(payload)

    try:
        response = requests.post(
            SGLANG_URL, headers=headers, data=json.dumps(payload), timeout=1000
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":

    def exam(effort):
        print("\n" + "=" * 50 + "\n")
        print(f"--- Running with {effort} ---")
        standard_response = get_chat_completion(PROMPT, effort)

        if standard_response and standard_response.get("choices"):
            print(f"\n[Response with {effort}]:")
            content = standard_response["choices"][0]["message"]["content"]
            print(content.replace("\n", "\\n"))
            print(f"0x{hash(content):X} {len(content):=}")
        else:
            print("\nCould not get a valid standard response.")

    exam("low")
    exam("medium")
    exam("high")
    exam(None)
