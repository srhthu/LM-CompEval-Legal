from copy import deepcopy
import torch
from transformers import LlamaConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM,AutoTokenizer, PreTrainedModel, GenerationConfig, AutoConfig
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM
from dataclasses import dataclass, asdict
from typing import Optional, Union, List, Dict, Any

@dataclass
class HF_Model_Config:
    """
    Class that holds arguments to build transformers model.
    Args:
        base_model: the name or path of the base model
        peft_model: model_id of peft model
        is_seq2seq: whether base model is seq2seq or causal lm
        trust_remote_code: should be true if using third-party models
        device_map: default to 'auto' to distribute parameters among all gpus
        save_memory: If true, generate one output each time. Otherwise decode multiple outputs together
    """
    base_model: str = None
    peft_model: str = None
    is_seq2seq: bool = False
    trust_remote_code: bool = False
    torch_dtype: Union[str, torch.dtype] = torch.float16
    device_map: Union[str, Dict] = 'auto'
    save_memory: bool = True

    def __post_init__(self):
        # infer base_model from peft config if not specified
        if not (self.base_model or self.peft_model):
            raise ValueError(f'Please specify at least one of `base_model` or `peft_model`')
        if not self.base_model:
            self.base_model = PeftConfig.from_pretrained(self.peft_model).base_model_name_or_path
        
        # convert torch_dtype
        if not isinstance(self.torch_dtype, torch.dtype):
            if self.torch_dtype == 'float16':
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = 'auto'

@dataclass
class HF_Gen_Conf:
    """LM generation arguments similar to transformers.GenerationConfig"""
    num_return_sequences: Optional[int] = 1
    max_new_tokens: Optional[int] = 30
    # sampling strategy
    do_sample: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    def to_dict(self):
        return {k: deepcopy(v) for k,v in asdict(self).items() if v is not None}

class HF_AutoModel:
    """
    Initialize a huggingface model and handle generation.

    Attributes:
        - is_encoder_decoder: If true, load with Seq2SeqLM and 
                                do not remove prefix of generate results.
    """
    def __init__(
        self, 
        config: Union[HF_Model_Config, dict], 
        gen_config: Union[HF_Gen_Conf, dict] = {}
    ):
        self.config = config if isinstance(config, HF_Model_Config) else HF_Model_Config(**config)
        self.gen_config = gen_config if isinstance(gen_config, HF_Gen_Conf) else HF_Gen_Conf(**gen_config)

        self.init_tokenizer()

        self._model = None
        # the model will be initialized when accessed
    
    def init_tokenizer(self):
        config = self.config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, trust_remote_code = config.trust_remote_code
        )
        # update generation config
        # fix pad_token_id issue: https://github.com/huggingface/transformers/issues/25353
        self.gen_config.eos_token_id = self.tokenizer.eos_token_id
        self.gen_config.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None 
            else self.tokenizer.eos_token_id
        )

    @property
    def model(self):
        if self._model is None:
            self.init_model()
        return self._model
    
    def init_model(self):
        """Determin is_encoder_decoder, Load base model and peft model if any"""
        config = self.config
        kws = self.get_init_kws()
        print(f'Initialize base model: {config.base_model}')
        # determin is_encoder_decoder
        hf_cfg = AutoConfig.from_pretrained(config.base_model, trust_remote_code = config.trust_remote_code)
        self.is_encoder_decoder = hf_cfg.is_encoder_decoder
        # initialize CausalLM or Seq2SeqLM
        if self.is_encoder_decoder:
            auto_cls = AutoModelForSeq2SeqLM
        else:
            has_auto_map = hasattr(hf_cfg, "auto_map")
            if not has_auto_map or 'AutoModelForCausalLM' in hf_cfg.auto_map:
                auto_cls = AutoModelForCausalLM
            elif 'AutoModelForSeq2SeqLM' in hf_cfg.auto_map:
                # to handle prefix-lm e.g. chatglm
                auto_cls = AutoModelForSeq2SeqLM
            else:
                raise ValueError(f'Cannot determin the Auto class')
        base_model = auto_cls.from_pretrained(config.base_model, **kws)
        # load peft model
        if config.peft_model:
            print(f'Load peft model from {config.peft_model}')
            model = PeftModel.from_pretrained(base_model, config.peft_model) 
        else:
            model = base_model
        self._base_model = base_model
        self._model = model
    
    def reload_peft_model(self, model_id):
        """Reload another peft model"""
        # update model config
        self.config.peft_model = model_id
        if self._model is None:
            self.init_model()
        else:
            print(f'Reload peft model from {model_id}')
            self._model = PeftModel.from_pretrained(
                self._base_model, model_id, **self.get_init_kws()
            )

    def get_init_kws(self):
        config = self.config
        return dict(
            trust_remote_code = config.trust_remote_code,
            torch_dtype = config.torch_dtype,
            device_map = config.device_map
        )
    
    def generate(self, input_text: str, num_output: Optional[int] = None, return_ids = False)->List[str]:
        """Generate conditioned on one input"""
        num_output = num_output or self.gen_config.num_return_sequences or 1
        # tokenize
        enc = self.tokenizer([input_text], return_tensors = 'pt')
        inputs = {k:v.cuda() for k,v in enc.items()}
        
        # generation config
        hf_gen_cfg = GenerationConfig(**self.gen_config.to_dict())
        # kws = self.get_generation_kws()
        if self.config.save_memory:
            hf_gen_cfg.num_return_sequences = 1
            # kws['num_return_sequences'] = 1
            output_ids = [self.model.generate(generation_config = hf_gen_cfg, **inputs)[0] for _ in range(num_output)]
        else:
            hf_gen_cfg.num_return_sequences = num_output
            # kws['num_return_sequences'] = num_output
            output_ids = self.model.generate(generation_config = hf_gen_cfg, **inputs)
        
        # print(output_ids)
        pure_output_ids = [self.remove_prefix(k, inputs['input_ids']) for k in output_ids]
        choices = self.tokenizer.batch_decode(pure_output_ids, skip_special_tokens=True)

        return (choices, output_ids) if return_ids else choices

    def get_hf_generation_config(self):
        arg_names = ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty']
        kws = {k:v for k,v in self.gen_config.to_dict().items() if k in arg_names}
        return GenerationConfig(**kws)
    
    def get_generation_kws(self):
        arg_names = ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty']
        kws = {k:v for k,v in self.gen_config.to_dict().items() if k in arg_names}
        return kws

    def remove_prefix(self, tensor, input_ids):
        """Remove the input ids from the output ids"""
        if self.is_encoder_decoder:
            return tensor
        else:
            return tensor[len(input_ids[0]):]