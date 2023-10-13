"""Evaluate a model on subtasks, save results and metrics"""
import os
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import traceback
from sklearn.metrics import precision_recall_fscore_support
import time
from typing import List

from lm_eval.models import HF_AutoModel
from lm_eval.tasks import JudgmentPrediction_Task, TaskBase
from lm_eval.utils import read_jsonl, save_jsonl
from lm_eval.parser import BM25_Parser

def perform_task_timely_save(
    tasks, map_func, save_file, 
    id_field = 'idx', skip_error = True, simple_log = True
):
    """
    Apply map_func on each task and save returned results to a file immediately.
     
    This function support resume after disruption by skipping finished tasks in save_file

    Args:
        - tasks: a list of task to be applied on by map_func
        - map_func: a callable function to apply on each task
        - save_file: file path to save map_func results
        - id_field: the field name to identify examples
        - skip_error: if error occurs in map_func, whether to skip the example
        - simple_log: True to print error message, otherwise print the traceback
    """
    # Load previous finished tasks if exist
    if Path(save_file).exists():
        prev_task = [json.loads(k) for k in open(save_file, encoding='utf8')]
    else:
        prev_task = []
    prev_idx = set([k[id_field] for k in prev_task])
    print(f'Previous finished: {len(prev_idx)}. Total: {len(tasks)}')
    
    # Keep unfinished tasks
    left_tasks = list(filter(lambda k:k[id_field] not in prev_idx, tasks))
    
    # Perform tasks
    for sample in tqdm(left_tasks, ncols = 80):
        try:
            results = map_func(sample)
            # add id field
            if id_field not in results:
                results[id_field] = sample[id_field]
            # write results to a file
            with open(save_file, 'a', encoding='utf8') as f:
                f.write(json.dumps(results, ensure_ascii=False) + '\n')
        except Exception as e:
            if simple_log:
                err_str = str(e)
            else:
                err_str = traceback.format_exc()
            tqdm.write(f'Error {id_field}={sample[id_field]}, {err_str}')
            if not skip_error:
                exit()

def evaluate_subtasks(
    model, 
    task: JudgmentPrediction_Task, 
    output_dir: str, 
    sub_tasks: str = 'all'
):
    """
    Args:
        - model: a HF_Model or Openai_Model
        - task: a TaskBase instance
        - output_dir: directory to save subtask results. Each subtask will create its own sub-directory.
        - sub_tasks: 'all' to evaluate on all sub_tasks, or subtask names joint by comma (,)
    
    Output dir structure:
        <sub_task_name>
            raw_output.txt
        ...
        eval_results.txt
    
    Each line of `eval_results.txt` is a json dict of time, subtask, metrics
    """
    # Create output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    metric_file = output_dir / 'eval_results.txt'

    # prepare some objects for evaluation
    test_ds = task.get_task_data()
    label2id = task.label2id
    grounds = [label2id[task.convert_label(k['charge'])] for k in test_ds]
    parser = BM25_Parser(label2id)

    if sub_tasks == 'all':
        sub_tasks = task.get_all_subtask()
    else:
        sub_tasks = sub_tasks.split(',')
    
    def ljp_handler(data):
        choices = model.generate(data['prompt'])
        return {'idx': data['idx'], 'choices': choices, 'prompt': data['prompt']}

    for task_name in sub_tasks:
        # Build task data
        print(f'Build {task_name} data')
        task_data = task.build_subtask(task_name)

        # Create subtask dir
        subtask_dir = output_dir / task_name
        subtask_dir.mkdir(exist_ok = True)
        save_file = subtask_dir / 'raw_output.txt'
        # Perform subtask
        perform_task_timely_save(task_data, ljp_handler, save_file, id_field = 'idx')
        
        # Evaluate subtask
        outputs = read_jsonl(save_file)
        idx2out = {k['idx']: k['choices'] for k in outputs}
        finish_all = all([k['idx'] in idx2out for k in test_ds])
        if not finish_all:
            print((
                'Inference of some examples failed. Skip evaluation.\n'
                'To finish all test examples, run the script again.\n'
                'If the failure is due to limited resources, e.g., GPU Memory,'
                'adjust some hyperparameters, e.g., max_len, and run the script again.'   
            ))
            continue
        # parse open generated text to pre-defined label names
        print('Save parsed results')
        parse_results = [{'idx': k['idx'], 'pred': parser(k['choices'])} for k in outputs]
        save_jsonl(parse_results, subtask_dir / 'parse_results.txt')

        # prepare for calculating metrics
        idx2pred = {k['idx']: label2id[k['pred']] for k in parse_results}
        preds = [idx2pred[k['idx']] for k in test_ds]
        metrics = get_ljp_metrics(preds, grounds)
        metrics = {k:float(v) for k,v in metrics.items()} # for serialization
        log = {'time': time.time(), 'subtask': task_name, 'metrics': metrics}
        print('Evaluation results:\n' + str(log))
        with open(metric_file, 'a') as f:
            f.write(json.dumps(log) + '\n')

def get_ljp_metrics(preds: List[int], targets: List[int]):
    preds = np.array(preds, dtype = np.int64)
    targets = np.array(targets, dtype = np.int64)
    acc = (preds == targets).astype(np.float32).mean()
    p,r,f1, _ = precision_recall_fscore_support(targets, preds, average = 'macro')
    metrics = {'acc': acc,
               'precision': p,
               'recall': r,
               'f1': f1}
    return metrics