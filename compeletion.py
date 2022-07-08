import pickle
import argparse
from gpt3 import inference_gpt3_compeletion
from tqdm import tqdm

import debugger


def main(args):
    data = pickle.load(open(args.ckpt, "rb"))

    config = data['config']
    ckpt_model = config['model']
    config['model'] = args.model

    prompt_texts = data['result']['prompt_texts']
    completion_texts = []
    prompt_plus_completion_texts = []

    for prompt_text in tqdm(prompt_texts):
        response, completion_text = inference_gpt3_compeletion(prompt_text=prompt_text, max_tokens=128)
        completion_texts.append(completion_text)
        prompt_plus_completion_texts.append(prompt_text + completion_text)

    result = {"prompt_texts": prompt_texts,
              "completion_texts": completion_texts,
              "prompt_plus_completion_texts": prompt_plus_completion_texts,
              "prompt_tokens": data['result']['prompt_tokens'],
              "prompt_examples": data['result']['prompt_examples']}

    gpt3_data = {}
    gpt3_data['config'] = config
    gpt3_data['result'] = result
    gpt3_data['raw_response'] = response

    output_fn = args.ckpt.replace(ckpt_model, args.model)
    with open(output_fn, 'wb') as fout:
        pickle.dump(gpt3_data, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    assert "generate" in args.ckpt
    assert args.model in ['ada', 'babbage']

    main(args)

