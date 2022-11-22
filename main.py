import re
import os
import time
import pickle
import random
import logging
import argparse
import yaml
import easydict
from hashlib import md5
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ImmutableLM
from dataset import PromptCorpus

from utils import dynamic_batching

import debugger

logger = logging.getLogger(__name__)


def init_model(args):
    model = ImmutableLM(args.model)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device = args.device
        if device != None and device < device_count:
            model.cuda(device) # Use the last GPU
    return model

def inference_mode(model: ImmutableLM, dataset: DataLoader, restricted_token):
    result = []
    for data in tqdm(dataset):
        with torch.no_grad():
            model.eval()
            model.backbone.eval()
            output = model(data, restricted_token=restricted_token)
            result.append(output)
    return result

def generation_mode(model: ImmutableLM, dataset: DataLoader, config: easydict.EasyDict, args):
    template_text = re.sub("({.*?})|\'", "", config.template[1:]).replace("\\n", '\n')
    label_text = ' '.join(config.label_mapping.values())
    allowed_text = template_text + " " + label_text
    allowed_tokens = tuple(model.tokenizer.encode(allowed_text))
    print(f"allowed tokens {model.tokenizer.decode(allowed_tokens)}")

    data = iter(dataset).__next__()
    with torch.no_grad():
        model.eval()
        model.backbone.eval()
        # output = model.balance_generate(
        output = model.generate(
            data, max_length=args.max_generation_length, no_repeat_ngram_size=args.ngram,
            allowed_tokens=allowed_tokens,
            prefix=config.template[2:].split("{")[0],
            temperature=args.temperature, do_sample=args.do_sample, top_k=args.topk)
        result = output

    return result

def get_config_hash(cfg):
    label_mapping_hash = md5(str.encode('-'.join(tuple(cfg.label_mapping.values())))).hexdigest()[:3]
    template_hash = md5(str.encode(cfg.template)).hexdigest()[:3]
    hash_str = label_mapping_hash + template_hash
    return hash_str

def main(corpus_config, args):

    cfg = easydict.EasyDict(corpus_config)

    if 'gpt2' not in args.model:
        cfg.tokenizer_path = args.model

    print(cfg)
    corpus = PromptCorpus(**cfg)

    corpus_config["model"] = args.model
    corpus_config["temperature"] = args.temperature
    corpus_config["do_sample"] = args.do_sample
    corpus_config["topk"] = args.topk


    dataset = DataLoader(corpus, batch_size=1, shuffle=False)


    model = init_model(args)

    cfg_fname = os.path.split(args.config)[-1].replace(".yaml", "")
    cfg_hash_str = get_config_hash(cfg)

    if args.generate:
        result = generation_mode(model=model, dataset=dataset, args=args, config=cfg)
        dump_fname = f"generate_{args.ngram}gram_{cfg_fname}_{cfg.n_shot}_shot_{args.model}_seed{args.seed}_{cfg.sample_mode}_temperature{args.temperature}_top{args.topk}_hash{cfg_hash_str}.pkl"
    else:
        result = inference_mode(model=model, dataset=dataset, restricted_token=corpus.restricted_token)
        dump_fname = f"{cfg_fname}_{cfg.n_shot}_shot_{args.model}_seed{args.seed}_{cfg.sample_mode}_hash{cfg_hash_str}.pkl"

    output_ckpt = {"result": result, "config": corpus_config}
    pickle.dump(output_ckpt,
                open(os.path.join(args.output, dump_fname), 'wb'))

if __name__ == '__main__':

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
    parser.add_argument("--device", type=int, default=0)

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

