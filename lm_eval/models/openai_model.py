import os
from environs import Env
import openai
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

@dataclass
class Openai_Gen_Conf:
    """Parameters with the names of OpenAI API"""
    max_tokens: Optional[int] = 30
    temperature: Optional[float] = 1.0

class Openai_Model:
    def __init__(self):
        ...
    
    def set_api_key(self):
        env = Env()
        env.read_env()
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    def complete(self, prompt, n) -> List[str]:
        response = openai.Completion.create(
            model = self.model,
            prompt = prompt,
            n = n,
           **self.get_kws()
        )
        choices = [c.text for c in response.choices]
        return choices
    
    def chatcomplete(self, prompt, n) -> List[str]:
        """Use the chat endpoint as a completion endpoint"""
        response = openai.ChatCompletion.create(
            model = self.model,
            messages=[{"role": "user", "content": prompt},],
            n = n,
            **self.get_kws()
        )
        choices = [c['message']['content'] for c in response['choices']]
        return choices
    
    def generate(self, input_text, num_output: Optional[int] = None):
        num_output = num_output or self.gen_config.num_output or 1
        if self.endpoint == 'chat':
            choices = self.chatcomplete(input_text, num_output)
        elif self.endpoint == 'complete':
            choices = self.complete(input_text, num_output)
        else:
            raise ValueError(f'endpoint: {self.endpoint}')
        return choices

    @staticmethod
    def is_chat_endpoint(model):
        return 'gpt' in model