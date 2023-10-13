import json

def read_jsonl(path):
    return [json.loads(k) for k in open(path, encoding='utf8')]

def read_json(path):
    return json.load(open(path, encoding='utf8'))

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')