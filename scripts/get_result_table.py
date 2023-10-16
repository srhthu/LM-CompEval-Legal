"""
Merge results of all models
"""
import json
import pandas as pd
from pathlib import Path

def merge_metric(exp_dir, metric)->pd.DataFrame:
    """Merge the specified metric of all models"""
    sub_tasks = [f'{tt}-{n}shot' for tt in ['free', 'multi'] for n in range(5)]

    lines = []
    for model in ['gpt4', 'chatgpt', 'bloomz_7b', 'chatglm_6b', 'vicuna_13b']:
        eval_file = Path(exp_dir) / model / 'eval_results.txt'
        if eval_file.exists():
            with open(eval_file) as f:
                model_results = [json.loads(k) for k in f]
            task2metric = {rec['subtask']: rec['metrics'][metric] for rec in model_results}
        else:
            task2metric = {}
        lines.append([model] + [task2metric.get(k) for k in sub_tasks])
    df = pd.DataFrame(lines, columns = ['model'] + sub_tasks)
    return df

def main(exp_dir, metric, save_path):
    df = merge_metric(exp_dir, metric)
    tot_score = (df['free-0shot'] + df['multi-0shot'] + df['free-2shot'] + df['multi-2shot']) / 4
    df.insert(1, 'score', tot_score)
    df = df.sort_values(by = ['score'], ascending = False)
    df.to_csv(save_path, index = False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', help = 'dir of all model results', default = 'runs/paper_version')
    parser.add_argument('--metric', help = 'metric name, f1 or acc', default = 'f1')
    parser.add_argument('--save_path', help = 'where to save the table', default = 'resources/f1.csv')
    args = parser.parse_args()
    main(**vars(args))