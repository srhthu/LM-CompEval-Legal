import os
from environs import Env
import openai
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

@dataclass
class OpenAI_Conf:
    """Model name and OpenAI generation parameters"""
    model: str
    num_output: int = 1
    max_tokens: Optional[int] = 30
    temperature: Optional[float] = 1.0
    stop: Union[str, List[str]] = '\n'

class OpenAI_Model:
    """
    Wrapper of openai chat and completion endpoint.

    Attributes:
        - endpoint: chat for chat models and complete for lm
    """
    def __init__(self, config: Union[OpenAI_Conf, dict]):
        self.config = config if isinstance(config, OpenAI_Conf) else OpenAI_Conf(**config)
        self.model = config.model
        self.endpoint = 'chat' if self.is_chat_endpoint(self.model) else 'complete'
        self.set_api_key()
    
    def set_api_key(self):
        env = Env()
        env.read_env()
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    def get_gen_kws(self):
        """Return generation arguments in dict"""
        return dict(
            max_tokens = self.config.max_tokens,
            temperature = self.config.temperature,
            stop = self.config.stop
        )

    def complete(self, prompt, n) -> List[str]:
        response = openai.Completion.create(
            model = self.model,
            prompt = prompt,
            n = n,
           **self.get_gen_kws()
        )
        choices = [c.text for c in response.choices]
        return choices
    
    def _chat_complete(self, messages, n) -> List[str]:
        """Use the chat endpoint as a completion endpoint"""
        response = openai.ChatCompletion.create(
            model = self.model,
            messages= messages,
            n = n,
            **self.get_kws()
        )
        choices = [c['message']['content'] for c in response['choices']]
        return choices

    def chatcomplete(self, prompt: Union[str, List[dict]], n) -> List[str]:
        """
        Args:
            prompt: a context or messages
            n: number of output
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt},]
        else:
            messages = prompt
        return self._chat_complete(messages, n)
    
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