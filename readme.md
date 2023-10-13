# A Comprehensive Evaluation of Large Language Models on Legal Judgment Prediction

## Usage
### Evaluate Huggingface models:
To evaluate a huggingface model on all 10 sub settings (`{free,multi}-{0..5}shot`):
```Bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--config ./config/default_hf.json \
--output_dir ./runs/<model_name> \
--model_type hf \
--model <path of model>
```
To evaluate some of the whole settings, add one more argument:
```Bash
--sub_tasks 'free-0shot,free-2shot,multi-0shot,multi-2shot'
```
The huggingface paths of the evaluated models in the paper are
-  ChatGLM: `THUDM/chatglm-6b` (add `--is_seq2seq`)
-  BLOOMZ: `bigscience/bloomz-7b1-mt`
-  Vicuna: `lmsys/vicuna-13b-delta-v1.1`

> - If the evaluation process is interupted, just run it again with the same parameters. The process saves model outputs immediately and will skip previous finished samples when resuming.  
> - Samples that trigger a GPU out-of-memory error will be skipped. You can change the configurations and run the process again. (See suggested GPU configurations below)

**Suggested GPU configurations**
- 7B model
  - 1 GPU with RAM around 24G (RTX 3090, A5000)
  - If total RAM >=**32G**, e.g., 2\*RTX3090 or 1\*V100(32G), add the `--speed` argument for faster inference.
- 13B model
  - 2 GPU with RAM >= 24G (e.g., 2\*V100)
  - If total RAM>=**64G**, e.g., 3\*RTX3090 or 2\*V100, add the `--speed` argument for faster inference
> When context is long, e.g., in multi-4shot setting, 1 GPU of 24G RAM may be insufficient for 7B model. You have to eigher increase the number of GPUs or decrease the demonstration length (default to 500) by modifying the *demo_max_len* parameter in `config/default_hf.json`

### Evaluate OpenAI Models via API


## Tests
Test the model
```Bash
# test decoder-only lm
CUDA_VISIBLE_DEVICES=0 python -m tests.model_generate --model gpt2
# test prefix-lm
CUDA_VISIBLE_DEVICES=0 python -m tests.model_generate --model THUDM/chatglm-6b
# test encoder-decoder lm
CUDA_VISIBLE_DEVICES=0 python -m tests.model_generate --model google/t5-v1_1-base 
google/t5-v1_1-base
```
Test the building of task data
```Bash
python -m tests.ljp_task
```