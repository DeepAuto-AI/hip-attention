import requests
from transformers import AutoTokenizer

class SglangModel:
    def __init__(self, endpoint: str, tokenizer: AutoTokenizer):
        self.endpoint = endpoint
        self.tokenizer = tokenizer
    
    def generate(self, input_ids=None, input_text=None, max_tokens=128):
        if input_text is None:
            assert input_ids is not None
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)[0]
        else:
            assert input_ids is None

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
        
        print(response.json())
        
        if isinstance(response.json(), dict):
            responses = response.json()['text'][0]
        else:
            responses = response.json()[0]['text']
        
        return responses