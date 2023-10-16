import json
from typing import Dict, Any, List
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tiktoken import Encoding

class TaskBase:
    """
    The base class of Task that produce task data e.g. prompts.
    """
    def __init__(self):
        self._task_data = {}
    
    def build_subtask(self, name) -> List[Dict[str, Any]]:
        """Return subtask data"""
        raise NotImplementedError
    
    def get_all_subtask(self) -> List[str]:
        """Return the name of subtasks"""
        raise NotImplementedError

    def get_task_data(self):
        """Return task data for evaluation"""
        raise NotImplementedError
    
    def cut_text(self, text, max_len):
        """Truncate text to max length"""
        tokenizer = self.tokenizer
        if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            # handle transformers tokenizer
            outs = tokenizer(text, truncation = True, max_length = max_len)
            new_text = tokenizer.decode(
                outs.input_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces = True, # ensure chinese characters are not splited by space
            )
        elif isinstance(tokenizer, Encoding):
            # handle OpenAI tokenizer
            token_ids = tokenizer.encode(text)[:max_len]
            new_text = tokenizer.decode(token_ids)
            rep_chr = chr(65533) # this is the token of error utf-8 codes
            # replace the last error token if exists
            if new_text[-1] == rep_chr:
                new_text = new_text[:-1]
        else:
            raise ValueError(f'Unknown tokenizer type: {tokenizer.__class__.__name__}')
        return new_text