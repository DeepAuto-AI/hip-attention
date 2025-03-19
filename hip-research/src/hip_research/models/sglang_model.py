import requests
from transformers import AutoTokenizer


class SglangModel:
    def __init__(self, endpoint: str, tokenizer: AutoTokenizer):
        self.endpoint = endpoint
        self.tokenizer = tokenizer

    def generate(
        self,
        input_ids=None,
        input_text=None,
        max_tokens=128,
        verbose=True,
        need_chat_prompt=False,
        system_message="You are helpful assistant.",
        handle_deepseek=False,
    ) -> str:
        if input_text is None:
            assert input_ids is not None
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)[
                0
            ]
        else:
            assert input_ids is None

        if not need_chat_prompt:
            response = requests.post(
                f"{self.endpoint}/generate",
                json={
                    "text": input_text,
                    "sampling_params": {
                        # "top_k": 1, # greedy,
                        "top_p": 1e-6,
                        "max_new_tokens": max_tokens,
                    },
                },
            )

            assert response.status_code == 200, response.json()

            if verbose:
                print(response.json())

            if isinstance(response.json(), dict):
                if isinstance(response.json()["text"], list):
                    responses = response.json()["text"][0]
                else:
                    responses = response.json()["text"]
            else:
                responses = response.json()[0]["text"]

            final_response = responses
        else:
            assert isinstance(input_text, (str, list))

            if isinstance(input_text, str):
                input_text = [
                    {
                        "role": "system",
                        "content": f"Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{system_message}",
                    },
                    {"role": "user", "content": input_text},
                ]
                if system_message is None:
                    input_text.pop(0)

            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": "anything",
                    "messages": input_text,
                    "max_tokens": max_tokens,
                    "top_p": 0.000000000001,
                },
            )

            if verbose:
                print(response.json())

            final_response = response.json()["choices"][0]["message"]["content"]

        if handle_deepseek:
            eothink = "</think>"
            if eothink in final_response:
                final_response = final_response[
                    final_response.find(eothink) + len(eothink) :
                ]

        return final_response
