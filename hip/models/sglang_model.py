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
    ):
        if input_text is None:
            assert input_ids is not None
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)[0]
        else:
            assert input_ids is None

        if not need_chat_prompt:
            response = requests.post(
                f"{self.endpoint}/generate",
                json={
                    "text": input_text,
                    "sampling_params": {
                        "top_k": 1, # greedy
                        "max_new_tokens": max_tokens,
                    },
                },
            )
            
            assert response.status_code == 200, response.json()
            
            if verbose:
                print(response.json())
            
            if isinstance(response.json(), dict):
                if isinstance(response.json()['text'], list):
                    responses = response.json()['text'][0]
                else:
                    responses = response.json()['text']
            else:
                responses = response.json()[0]['text']
            
            return responses
        else:
            assert isinstance(input_text, str)
            
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": "anything",
                    "messages": [
                        {'role': 'system', 'content': 'Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are helpful assistant.'},
                        {'role': 'user', 'content': input_text},
                    ],
                    "max_tokens": max_tokens,
                    "top_p": 0.000000000001,
                },
            )
            
            if verbose:
                print(response.json())
            
            return response.json()['choices'][0]['message']['content']