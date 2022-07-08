import random
import json
import pickle
import argparse
import os
import easydict
import yaml
import numpy as np
from hashlib import md5
from tqdm import tqdm

from dataset import PromptCorpus

from gpt3_utils import inference_gpt3_prediction

def get_config_hash(cfg):
    label_mapping_hash = md5(str.encode('-'.join(tuple(cfg.label_mapping.values())))).hexdigest()[:3]
    template_hash = md5(str.encode(cfg.template)).hexdigest()[:3]
    hash_str = label_mapping_hash + template_hash
    return hash_str

def main(corpus_config, args):
    cfg = easydict.EasyDict(corpus_config)
    print(cfg)
    corpus = PromptCorpus(**cfg)

    corpus_config["model"] = args.model
    corpus_config["temperature"] = args.temperature
    corpus_config["do_sample"] = args.do_sample
    corpus_config["topk"] = args.topk

    restricted_token_text = [corpus.tokenizer.decode([token]) for token in corpus.restricted_token]
    result = []

    for i, data in enumerate(tqdm(corpus)):
        ret = run(data, restricted_token=restricted_token_text, max_permutation=args.max_permutation)
        result.append(ret)

    cfg_fname = os.path.split(args.config)[-1].replace(".yaml", "")
    cfg_hash_str = get_config_hash(cfg)
    dump_fname = f"{cfg_fname}_{cfg.n_shot}_shot_{args.model}_seed{args.seed}_{cfg.sample_mode}_hash{cfg_hash_str}.pkl"

    output_ckpt = {"result": result, "config": corpus_config}
    pickle.dump(output_ckpt,
                open(os.path.join(args.output, dump_fname), 'wb'))

def run(data, restricted_token, max_permutation):

    prompt_sequences = data['raw_sequence']
    prediction_texts = []
    prediction_dists = []
    for sequence in prompt_sequences[:max_permutation]:
        raw_response, gpt3_prediction_dist = inference_gpt3_prediction(sequence, engine=args.model)

        prediction_dist = []
        for token in restricted_token:
            try:
                logits = gpt3_prediction_dist[token]
            except KeyError as e:
                logits = -1e10
            prediction_dist.append(logits)
        # prediction_dist = [gpt3_prediction_dist[token] for token in restricted_token]
        prediction_text = restricted_token[np.argmax(prediction_dist)]

        prediction_texts.append(prediction_text)
        prediction_dists.append(prediction_dist)
    prediction_texts = ' '.join(' '.join(prediction_texts).split())
    result = {"prediction_text": prediction_texts,
              "prediction_dist": prediction_dists,
              "label": data["label"],
              "prompt_examples": data["train_metadata"],
              "prompt_sequence_text": data["raw_sequence"],
              "prompt_sequence_index": data["input_sequence"],
              "raw_response": raw_response}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--nshot", "-n", type=int, default=0)
    parser.add_argument("--test_data_path", type=str, default="")

    parser.add_argument("--output", "-o", type=str, default="default_output")

    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--ngram", type=int, default=0)
    parser.add_argument("--max_generation_length", "-l", type=int, default=128)
    parser.add_argument("--temperature", "-t", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--train_sample_mode", type=str, default="")
    parser.add_argument("--max_permutation", type=int, default=24)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    corpus_config = yaml.safe_load(open(args.config))

    if args.test_data_path:
        print(f"override test data path from {corpus_config['test_data_path']} to {args.test_data_path}")
        corpus_config['test_data_path'] = args.test_data_path

    if args.nshot > 0:
        print(f"override n-shot from {corpus_config['n_shot']} to {args.nshot}")
        corpus_config['n_shot'] = args.nshot

    if args.train_sample_mode:
        print(f"override train data sample mode from {corpus_config['sample_mode']} to {args.train_sample_mode}")
        corpus_config["sample_mode"] = args.train_sample_mode

    main(corpus_config=corpus_config, args=args)
