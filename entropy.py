import pickle
import sys
import debugger
from evaluate import evaluate_file
from collections import Counter
import numpy as np
from scipy.stats import pearsonr, spearmanr

import argparse
import json

def kl(distp, distq):
    total_sum = 0.0
    for p, q in zip(distp, distq):
        total_sum += (-1.0* p * np.log(q/p))
    return total_sum

def cal_entropy(dist):
    return sum([-p*np.log(p) for p in dist])

def selection(fn_gen, fn_true, topk=4):
    data_gen = pickle.load(open(fn_gen, 'rb'))
    acc = evaluate_file(fn_true)

    label_mapping = data_gen['config']['label_mapping']
    permutation_max_size = len(data_gen['result'][0]['prompt_sequence_text']) #data_gen['config']['permutation_max_size']
    # print(label_mapping)
    prediction_texts = list(zip(*[elem['prediction_text'].split() for elem in data_gen['result']]))
    all_labels = list(label_mapping.keys())
    all_labels.sort()

    entropys = []
    for i in range(len(prediction_texts)):
        dist = []
        labeltext2freq = Counter(prediction_texts[i])
        for label in all_labels:
            label_text = label_mapping[label]
            label_freq = 1e-5  # prevent zero term
            if label_text in labeltext2freq:
                label_freq = labeltext2freq[label_text]
            dist.append(label_freq)

        dist = np.array(dist)
        dist = dist / dist.sum()
        entropy = cal_entropy(dist)
        entropys.append(entropy)
    pearsonr_metric, spearmanr_metric = pearsonr(entropys, acc), spearmanr(entropys, acc)
    print(pearsonr_metric, spearmanr_metric)

    gg = list(zip(entropys, acc))
    gg.sort(key=lambda x: x[0], reverse=True)
    if len(gg) == 2:
        print(f"1-shot case, only two examples, change topk from {topk} to 1")
        topk = 1
    assert len(gg) > topk, f"total permutation is less than {topk}"
    subset_acc = [elem[1] for elem in gg[:topk]]
    acc_mean, acc_std = np.mean(acc), np.std(acc)
    subset_acc_mean, subset_acc_std = np.mean(subset_acc), np.std(subset_acc)
    print(f"Before: mean {acc_mean}, std {acc_std}")
    print(f"After: mean {subset_acc_mean}, std {subset_acc_std}")
    result = {"acc_stats": (acc_mean, acc_std),
              "topk_acc_stats": (subset_acc_mean, subset_acc_std),
              "topk": topk, "entropys": entropys, "acc": acc,
              "ckpt": fn_true, "ckpt_gen": fn_gen,
              "pearsonr_corr": pearsonr_metric, "spearmanr_corr": spearmanr_metric,
              "topk": topk}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true", "-t", type=str, required=True)
    parser.add_argument("--fake", "-f", type=str, required=True)
    parser.add_argument("--topk", "-k", type=int, default=True)
    parser.add_argument("--save", "-s", type=str, required=True)

    args = parser.parse_args()
    result = selection(fn_gen=args.fake, fn_true=args.true, topk=args.topk)
    json.dump(result, open(args.save, 'w'))


if __name__ == '__main__':
    main()
