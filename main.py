"""
Comprehensive evaluation of a language model on Legal Judgment Prediction
"""
from pathlib import Path
import json

from lm_eval.models import HF_AutoModel
from lm_eval.tasks.ljp import JudgmentPrediction_Task
from lm_eval.evaluate import evaluate_subtasks


def main(run_config, output_dir, sub_tasks):
    # Initialize model
    model = HF_AutoModel(run_config['model_config'], run_config['gen_config'])
    # Initialize task
    task = JudgmentPrediction_Task(run_config['task_config'], model.tokenizer)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / 'run_config', 'w') as f:
        json.dump(run_config, f, indent = 4, ensure_ascii = False)
    
    evaluate_subtasks(model, task, output_dir = output_dir, sub_tasks = sub_tasks)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help = 'default config file path')
    parser.add_argument('--output_dir', help = 'path to save evaluation results')
    parser.add_argument('--model_type', help = 'openai or hf')
    parser.add_argument('--model', help = 'path of the model')
    parser.add_argument('--sub_tasks', help = 'subtask names saperated by comma', default = 'all')
    parser.add_argument('--speed', action = 'store_true', 
                        help = 'Improve inference speed by consuming more GPU memory. Recommended when GPU memory > 24G')
    
    args = parser.parse_args()
    
    run_config = json.load(open(args.config))
    if args.model_type == 'hf':
        run_config['model_config']['base_model'] = args.model
        run_config['model_config']['save_memory'] = not args.speed
    elif args.model_type == 'openai':
        ...
    else:
        raise ValueError(f'model_type: {args.model_type}')
    
    main(run_config, args.output_dir, args.sub_tasks)
    