"""
Generate prompts for legal judgment prediction.
"""

import re
from pathlib import Path
import pandas as pd
import json
import random
import numpy as np
import pickle
from collections import Counter
from transformers import PreTrainedTokenizer
from tiktoken import Encoding
from dataclasses import dataclass
import jieba
from typing import Union, List, Dict, Any

from .base import TaskBase
from lm_eval.utils import read_jsonl, read_json

@dataclass
class JudgmentPrediction_Config:
    data_path: str
    prompt_config_file: str
    query_max_len: int = 1000
    demo_max_len: int = 400

class JudgmentPrediction_Task(TaskBase):
    """
    Legal judgment prediction as a multi-class classification task.

    Subtasks:
    {free, multi}-{0..5}-shot
        - free is free generation, multi is multi-choice question
        - support zero-shot and few-shot
    
    Data Folder:
        - `test_data.json`: dict of fields of idx, fact, charge, sim_demo_idx, cdd_charge_list
        - `train_data.json`: dict of fields of idx, fact, charge
        - `charge2id.json`: mapping from charge names to charge label id
    
    Prompt Configuration:
        Stored as a dict of fields:
            - instruction_{free, multi}_{zs, fs}: instruction of four settings
            - demo_template: which has the slot of *input* and *output*
    Prompt:
        Build: <instruction> + <options> + <demos> + <query>
        Note: 
            - <options> is present in multi-choice setting
            - <demos> is present in few-shot setting
    """
    def __init__(
        self, 
        config: Union[JudgmentPrediction_Config, dict], 
        tokenizer: Union[PreTrainedTokenizer, Encoding]
    ):
        self.config = (config if isinstance(config,JudgmentPrediction_Config) 
                       else  JudgmentPrediction_Config(**config))

        self.prompt_config = read_json(self.config.prompt_config_file)
        data_dir = Path(self.config.data_path)

        train_ds = read_jsonl(data_dir / 'train_data.json')
        self.train_ds_map = {k['idx']:k for k in train_ds}

        self.test_ds = read_jsonl(data_dir / 'test_data.json')
        
        self.tokenizer = tokenizer

        label2id = read_json(data_dir / 'charge2id.json')
        self.label2id = {self.convert_label(k):v for k,v in label2id.items()}
    
    def get_task_data(self):
        return self.test_ds

    @staticmethod
    def convert_label(label: str):
        """Remove the square brackets in original charge names"""
        return re.sub(r'[\[\]]', '',label)
    
    def get_all_subtask(self):
        return [f'{t}-{i}shot' for t in ['free', 'multi'] for i in range(5)]
    
    def build_subtask(self, name)->List[Dict[str, Any]]:
        """
        Return a list of example prompt data, each of which is a dict of idx and prompt
        """
        # parse subtask name to get task type and demo number
        ttype, n_shot = name.split('-')
        n_shot = int(n_shot[0])

        prompt_data = [
            {
                'idx': example['idx'], 
                'prompt': self.build_example_prompt(example, ttype, n_shot)
            } for example in self.test_ds
        ]
        return prompt_data

    def build_example_prompt(self, example, task_type, n_shot):
        # get instruction
        p_config = self.prompt_config
        shot_s = 'zs' if n_shot == 0 else 'fs'
        instruct = p_config[f'instruction_{task_type}_{shot_s}']

        # determin label candidate list
        if task_type == 'multi':
            instruct += '\n' + '\n'.join(map(self.convert_label, example['cdd_charge_list']))

        # add demonstrations
        all_demo_str = []
        for demo_id in example['sim_demo_idx'][:n_shot]:
            demo_example = self.train_ds_map[demo_id]
            demo_str = p_config['demo_template'].format(
                input = self.cut_text(demo_example['fact'], self.config.demo_max_len),
                output = self.convert_label(demo_example['charge'])
            )
            all_demo_str.append(demo_str)

        # add query example
        query_str = p_config['demo_template'].format(
            input = self.cut_text(example['fact'], self.config.query_max_len),
            output = ''
        )
        
        # join all prompt components
        all_demo_str = '\n\n'.join(all_demo_str)
        if len(all_demo_str) > 0:
            all_demo_str = '\n\n' + all_demo_str
        prompt = instruct + all_demo_str + '\n\n' + query_str
        return prompt
