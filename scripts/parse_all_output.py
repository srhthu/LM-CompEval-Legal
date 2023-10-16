import json
import os
import sys
from pathlib import Path
import time
from typing import List
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))


from lm_eval.parser import BM25_Parser
from lm_eval.utils import read_jsonl, save_jsonl
from lm_eval.tasks import JudgmentPrediction_Task

all_run_dir = Path('runs/paper_version')

def main():
    label2id = json.load(open('/storage/rhshui/workspace/LM-CompEval-Legal/data_hub/ljp/charge2id_clean.json'))
    test_ds = [json.loads(k) for k in open('/storage/rhshui/workspace/LM-CompEval-Legal/data_hub/ljp/test_data.json')]
    grounds = [label2id[JudgmentPrediction_Task.convert_label(k['charge'])] for k in test_ds]
    parser = BM25_Parser(label2id)

    sub_tasks = [f'{tt}-{n}shot' for tt in ['free', 'multi'] for n in range(5)]

    for model in ['gpt4', 'chatgpt', 'bloomz_7b', 'chatglm_6b', 'vicuna_13b']:
        output_dir = all_run_dir / model
        metric_file = output_dir / 'eval_results.txt'
        for task_name in sub_tasks:
            subtask_dir = output_dir / task_name
            save_file = subtask_dir / 'raw_output.txt'
            parse_file = subtask_dir / 'parse_results.txt'
            if parse_file.exists():
                continue
            # Parse
            print(f'Parse: {save_file}')
            outputs = read_jsonl(save_file)
            idx2out = {k['idx']: k['choices'] for k in outputs}
            finish_all = all([k['idx'] in idx2out for k in test_ds])
            assert finish_all
            parse_results = [{'idx': k['idx'], 'pred': parser(k['choices'])} for k in outputs]
            save_jsonl(parse_results, parse_file)
            # Evaluate
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

if __name__ == '__main__':
    main()