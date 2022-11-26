import random
import torch
import torch.nn.functional as F

from string import printable
from collections import defaultdict

def get_model_prefix(model_name):
    if 'gpt2' in model_name:
        return ''
    elif 'bloom' in model_name:
        return 'bigscience/'
    elif model_name == 'sharded-gpt-j-6B':
        return 'sgugger/'
    elif 'gpt-j-6B-8bit-sharded' in model_name:
        return 'ethzanalytics/'
    else:
        return 'EleutherAI/'

def get_model_tokenizer(model_name):
    return 'EleutherAI/gpt-j-6B' if 'gpt-j-6B' in model_name else model_name

def compute_acc(prediction, gtruth):
    correct_counter = 0
    for p, g in zip(prediction, gtruth):
        if p == g:
            correct_counter += 1
    return correct_counter/len(prediction)


def dynamic_batching(batch):
    max_len = max([len(elem["input_sequence"]) for elem in batch])
    input_sequences = []
    attention_masks = []
    labels = []
    for i in range(len(batch)):
        fill_zero_length = max_len - len(batch[i]["input_sequence"])
        input_sequences.append(F.pad(batch[i]["input_sequence"], (0, fill_zero_length), value=50256))
        attention_masks.append(F.pad(batch[i]["attention_mask"], (0, fill_zero_length), value=0))
        labels.append(batch[i]["label"])
    batch_pad = {"input_sequence": torch.stack(input_sequences, dim=0),
                 "attention_mask": torch.stack(attention_masks, dim=0),
                 "label": torch.stack(labels, dim=0)}

    return batch_pad

def pad_sequence(input_sequences):
    max_len = max([len(elem) for elem in input_sequences])
    input_sequences = []
    attention_masks = []
    labels = []
    for i in range(len(input_sequences)):
        fill_zero_length = max_len - len(input_sequences[i]["input_sequence"])
        input_sequences.append(F.pad(input_sequences[i]["input_sequence"], (0, fill_zero_length), value=50256))
        attention_masks.append(F.pad(input_sequences[i]["attention_mask"], (0, fill_zero_length), value=0))
    input_sequence = torch.stack(input_sequences, dim=0)
    attention_mask = torch.stack(attention_masks, dim=0)

    return input_sequence, attention_mask

def filter_printable(sentence):
    sentence = ''.join([ch for ch in sentence if ch in printable])
    return " ".join(sentence.split())

def group_by_label(corpus, label_str="label"):
    grouped_corpus = defaultdict(list)
    for elem in corpus:
        label = elem[label_str]
        grouped_corpus[label].append(elem)
    return grouped_corpus


def corpus_sampling(train_corpus, kshot, mode='balance', label_str="label"):
    grouped_corpus = group_by_label(train_corpus, label_str=label_str)
    selected = []
    if mode == "balance":
        for label in grouped_corpus:
            selected += random.sample(grouped_corpus[label], kshot)
    elif mode == "random":
        print("random sample train corpus, k shot = k examples")
        selected += random.sample(train_corpus, kshot)
        # selected += random.sample(train_corpus, kshot * len(grouped_corpus))
    else:
        raise NotImplementedError("Please choose mode between balance and random")
    return selected


def create_prompt(template, sentence, label_text, test=False, sentence_pair=False):
    """
    :param template:  "f'Review: {sentence_1}\nSentiment: {label_text}\n\n'"
    :param sentence: tuple, e.g., (sent1, ) or (sent1, sent2)
    :param label_text: string, e.g., good or bad
    :param test: Boolean
    :param sentence_pair: Boolean
    :return:
    """

    if sentence_pair:
        assert len(sentence) == 2, "you should input sentence pair"
        assert "sentence_1" in template, "sentence_1 not found in template"
        assert "sentence_2" in template, "sentence_2 not found in template"
        sentence_1, sentence_2 = sentence
        sentence_1 = ' '.join(sentence_1.split())
        sentence_2 = ' '.join(sentence_2.split())

    else:
        assert len(sentence) == 1, "you should input single sentence as string"
        assert type(sentence) == tuple
        assert "sentence_1" in template, "sentence_1 not found in template"
        assert "sentence_2" not in template, "sentence_2 should not exist in template"

        sentence_1 = sentence[0]
        sentence_1 = ' '.join(sentence_1.split())

    if test:
        template = template[:template.index("{label_text}")] + "'" # end ' for complete f-string
        assert "{label_text}" not in template, "should remove label text for test data"
        # template = template.replace(" {label_text}\\n\\n", "")
        template_text = eval(template)
        # template_text = f"Review: {sentence_1}\nSentiment:"
    else:
        # template_text = f"Review: {text}\nSentiment: {label_text}\n\n"
        template_text = eval(template)
    return template_text