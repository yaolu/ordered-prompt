import os
import sys
import pickle
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import debugger


def compute_acc(prediction, gtruth):
    count = 0
    for p,g in zip(prediction, gtruth):
        if p == g:
            count += 1
    return round(1.0*count/len(prediction), 4)


def load_result_file(fname):
    data = pickle.load(open(f"{fname}", "rb"))
    return data["config"], data["result"]


def evaluate_file(fname):
    config, result = load_result_file(fname)
    print(fname, config)

    predictions = []
    for elem in result:
        prediction_str = elem["prediction_text"]
        for k, v in config["label_mapping"].items():
            prediction_str = prediction_str.replace(v, k)
        predictions.append(prediction_str.split())

    if type(result[0]['label']) == list:
        gtruth = [elem['label'][0] for elem in result]
    elif type(result[0]['label']) == str:
        gtruth = [elem['label'] for elem in result]

    target_labels = list(set(gtruth))

    # label2id = {}
    # for i, label in enumerate(target_labels):
    #     label2id[label] = i
    metrics = []
    # gtruth_ids = [label2id[label] for label in gtruth]

    if "cb_" in fname :
        for prediction in list(zip(*predictions)):
            metric = f1_score(gtruth, prediction, labels=target_labels, average=None)
            metrics.append(sum(metric)/len(metric))
        print("f1: ", metrics)
    from collections import Counter
    if ("qqp_" in fname) or ("mrpc_" in fname):
        for prediction in list(zip(*predictions)):
            metric = f1_score(gtruth, prediction, labels=target_labels)
            metrics.append(metric)
        print("f1: ", metrics)
        raise ValueError
    if "cola_" in fname:
        for prediction in list(zip(*predictions)):
            metric = matthews_corrcoef(gtruth, prediction, labels=target_labels)
            #
            # prediction_ids = [label2id[label] for label in prediction]
            # metric = matthews_corrcoef(gtruth_ids, prediction_ids)
            metrics.append(metric)
        print("matthews: ", metrics)
    print([compute_acc(elem, gtruth) for elem in list(zip(*predictions))])
    return [compute_acc(elem, gtruth) for elem in list(zip(*predictions))]


if __name__ == "__main__":
    fname = sys.argv[-1]
    if not os.path.exists(fname):
        raise FileNotFoundError("python evaluate.py path_to_pickle")
    evaluate_file(fname)
