"""
Test the function of CausalLM and Seq2SeqLM model generation
"""
from lm_eval.models import HF_AutoModel, HF_Model_Config, HF_Gen_Conf

def display(source, targets):
    print(f'Input: {source}')
    for i,t in enumerate(targets):
        print(f'Output #{i}: {t}')

def main(args):
    model = HF_AutoModel(
        HF_Model_Config(
            base_model = args.model,
            peft_model = args.peft_model,
            trust_remote_code = True,
            save_memory = False
        ),
        gen_config = HF_Gen_Conf(
            num_return_sequences = 3
        )
    )

    source = 'Today is a good day.'
    print(f'Test save_memory={model.config.save_memory}')
    targets, out_ids = model.generate(source, return_ids = True)
    display(source, targets)
    all_tokens = [model.tokenizer.convert_ids_to_tokens(k) for k in out_ids]
    for i, tokens in enumerate(all_tokens):
        print('Tokens #{}: {}'.format(i, ' '.join(tokens)))

    # print('Test save_memory=False')
    # model.config.save_memory = False
    # targets = model.generate(source)
    # print(targets)

    # print('Test greedy decoding')
    # model.gen_config.do_sample = False
    # targets = model.generate(source, num_output=1)
    # print(targets)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--peft_model')
    # parser.add_argument('--is_seq2seq', action = 'store_true')

    args = parser.parse_args()

    main(args)

    