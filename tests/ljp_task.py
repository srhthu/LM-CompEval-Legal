"""Test Legal Judgment Prediction Task"""
from tiktoken import encoding_for_model
from transformers import AutoTokenizer

from lm_eval.tasks.ljp import JudgmentPrediction_Task

def test_build_task(args, tokenizer):
    tasker = JudgmentPrediction_Task(
        config = {
            'data_path': args.data_path,
            'prompt_config_file': args.prompt_config
        },
        tokenizer = tokenizer
    )
    for subtask in tasker.get_all_subtask():
        print(f'Building subtask: {subtask}')
        data = tasker.build_subtask(subtask)
        print(data[0])

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_path', default = './data_hub/ljp')
    parser.add_argument('--prompt_config', default = './config/ljp_prompt.json')

    args = parser.parse_args()

    # print('#'*15 + 'Test ChatGPT')
    # tokenizer = encoding_for_model('gpt-3.5-turbo')
    # test_build_task(args, tokenizer)
    # print('\n\n')

    print('#'*15 + 'Test ChatGLM')
    tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code = True)
    test_build_task(args, tokenizer)

