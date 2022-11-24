import json
import time
import torch
import random
import pickle
import argparse
import logging

from collections import defaultdict
from transformers import GPT2Tokenizer, AutoTokenizer
from torch.utils.data import DataLoader
from model import ImmutableLM
from tqdm import tqdm

from itertools import permutations

from utils import corpus_sampling, create_prompt

import debugger

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
import os

class PromptCorpus:
    def __init__(self, train_data_path="data/train.jsonl",
                 test_data_path="data/dev.jsonl", tokenizer_path='distilgpt2',
                 n_shot=10, label_mapping={0: "bad", 1: "good"},
                 corpus_params={"sentence_1_string": "", "sentence_2_string": "", "label_string": ""},
                 template="f'Review: {sentence_1}\nSentiment: {label_text}\n\n'",
                 sample_mode="balance", permutation_max_size=24, sentence_pair=False):

        if 'gpt2' in tokenizer_path:
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        else:
            if 'gpt-neo-' in tokenizer_path:
                prefix = 'EleutherAI/'
            else:
                prefix = ''
            self.tokenizer = AutoTokenizer.from_pretrained(prefix+tokenizer_path)
        self.kshot = n_shot
        self.max_sequence_length = 1022

        self.label_mapping = label_mapping

        self.restricted_token = []
        for label_str in self.label_mapping.values():
            label_index = self.tokenizer.encode(f" {label_str}")
            assert len(label_index) == 1, "label should be single token"
            self.restricted_token += label_index

        self.restricted_token = tuple(self.restricted_token)

        # self.restricted_token = (neg_index[0], pos_index[0]) # (" True", " False")

        full_train_data = self.load_jsonl(train_data_path)

        self.train_data = corpus_sampling(full_train_data, kshot=self.kshot, mode=sample_mode,
                                          label_str=corpus_params["label_str"])
        self.test_data = self.load_jsonl(test_data_path)

        self.template = template
        self.sentence_pair = sentence_pair
        self.corpus_params = corpus_params
        self.permutation_max_size = permutation_max_size

        logger.info(f"{self.kshot}-shot, label_mapping: {label_mapping}, "
                    f"template: {template}")
        # print("{} for negative, {} for positive".format(label_mapping[0], candidate_mapping[1]))
        # print("{} as template".format(template))
        self._cache = {}

    def __len__(self):
        return len(self.test_data)

    @staticmethod
    def load_jsonl(fp):
        data = []
        with open(fp) as fin:
            for i, line in enumerate(fin):
                decoded = json.loads(line)
                decoded["index"] = i  # add index for all samples
                data.append(decoded)
        return data

    # Train example concat order: 1-1, 2-2, 3-6, 4-24
    @staticmethod
    def permute_train_prompts(train_prompts, max_count=24):
        """
        :param train_prompts: list of strings ["Sent: xxx\nLabel: xxx\n", "Sent: xxx\nLabel: xxx\n"]
        :return:
        """

        if len(train_prompts) > 4:
            print("Use subset of full permutations")
            subset = []
            while len(set(subset)) != max_count:
                subset.append(tuple(random.sample(train_prompts, len(train_prompts))))
            subset = list(set(subset))
            subset.sort()
        else:
            print("Use full permutations")
            train_prompts_permutation = list(permutations(train_prompts))
            sample_size = min(len(train_prompts_permutation), max_count)
            subset = random.sample(train_prompts_permutation, sample_size)

        return [''.join(elem) for elem in subset]

    # Test example: add one token per time
    @staticmethod
    def test_data_plus_one_token(corpus_data_point,
                                text_str="sentence", label_str="label"):
        text = corpus_data_point[text_str]
        text = text.split()
        label = corpus_data_point[label_str]
        data_point_index = corpus_data_point["index"]
        augmented_test_data = []
        for i in range(1, len(text)+1):
            augmented_test_data.append({text_str: ' '.join(text[:i]),
                                        label_str: label,
                                        "index": data_point_index})
        return augmented_test_data

    def __getitem__(self, item):

        train_prompts = []
        label_str = self.corpus_params["label_str"]
        if self.sentence_pair:
            sentence_1_str = self.corpus_params["sentence_1_str"]
            sentence_2_str = self.corpus_params["sentence_2_str"]
        else:
            sentence_1_str = self.corpus_params["sentence_1_str"]

        train_labels = []
        for data in self.train_data:
            if self.sentence_pair:
                train_sentence = (data[sentence_1_str], data[sentence_2_str])
            else:
                train_sentence = (data[sentence_1_str], )

            train_label = data[label_str]
            train_labels.append(train_label)
            train_label_text = self.label_mapping[train_label]
            p = create_prompt(template=self.template, sentence=train_sentence,
                              label_text=train_label_text, test=False,
                              sentence_pair=self.sentence_pair)
            train_prompts.append(p)

        # use cache to ensure the consistency of train examples across different test examples
        if "train_prompts_permutation" in self._cache:
            train_prompts_permutation = self._cache["train_prompts_permutation"]
        else:
            train_prompts_permutation = self.permute_train_prompts(train_prompts, max_count=self.permutation_max_size)
            self._cache = {"train_prompts_permutation": train_prompts_permutation}
            print("train_prompts_length: ", len(self.tokenizer.encode(train_prompts_permutation[0])))

        if self.sentence_pair:
            test_sentence = (self.test_data[item][sentence_1_str], self.test_data[item][sentence_2_str])
        else:
            test_sentence = (self.test_data[item][sentence_1_str],)
        test_label = self.test_data[item][label_str]
        test_label_text = self.label_mapping[test_label]

        test_sequence = create_prompt(template=self.template, sentence=test_sentence,
                                      label_text=test_label_text, test=True,
                                      sentence_pair=self.sentence_pair)
        input_sequences = [] # train + test
        input_sequences_prompt = [] # train only
        raw_sequences = []

        # test_index = self.test_data[item]["index"]

        for train_sequence in train_prompts_permutation:
            raw_sequence = ''.join([train_sequence, test_sequence])
            raw_sequence = raw_sequence.strip(" ")
            raw_sequence_train_only = train_sequence

            input_sequence = self.tokenizer.encode(raw_sequence, add_special_tokens=True)
            input_sequence_prompt = self.tokenizer.encode(train_sequence, add_special_tokens=True)
            # If the sequence is longer than 1024, trim from the start of the sequence
            input_sequence = input_sequence[-self.max_sequence_length:]
            input_sequence_prompt = input_sequence_prompt[-self.max_sequence_length:]
            input_sequences.append(torch.LongTensor(input_sequence))
            input_sequences_prompt.append(torch.LongTensor(input_sequence_prompt))

            raw_sequences.append(raw_sequence)

        #todo: refactor this part, only create single train example as prompt

        _d = {}
        for train_label, train_prompt in zip(train_labels, train_prompts):
            if train_label in _d:
                _d[train_label].append(train_prompt)
            else:
                _d[train_label] = [train_prompt]

        train_prompts_ids = [self.tokenizer.encode("".join(prompts)) for prompts in _d.values()]

        return {"input_sequence": torch.stack(input_sequences, dim=0),
                "label": test_label,
                "raw_sequence": raw_sequences,
                "train_metadata": self.train_data,
                "test_index": self.test_data[item]["index"],
                "input_sequences_prompt": torch.stack(input_sequences_prompt, dim=0),
                "train_prompts_ids": train_prompts_ids}

if __name__ == "__main__":

    import yaml
    import easydict
    corpus_config = yaml.safe_load(open("config/rte.yaml"))
    cfg = easydict.EasyDict(corpus_config)
    # logger.info(cfg)
    print(cfg)
    rte = PromptCorpus(**cfg)
    # rte = PromptCorpus(train_data_path=corpus_config["train_data_path"],
    #                    test_data_path=corpus_config["test_data_path"],
    #                    tokenizer_path=corpus_config["tokenizer_path"],
    #                    n_shot=corpus_config["n_shot"],
    #                    label_mapping=corpus_config["label_mapping"],
    #                    corpus_params=corpus_config["corpus_params"],
    #                    template=corpus_config["template"],
    #                    sample_mode=corpus_config["sample_mode"],
    #                    sentence_pair=corpus_config["sentence_pair"])
    datapoint = rte[0]
