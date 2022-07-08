import re
import json
import sys
import pickle
import easydict

from utils import filter_printable

import debugger

def augment_data_from_generation(generation_data, cfg, min_length=5):
    example_reg_str = "(" + cfg.template[2:-1].replace("\\n",'\n').replace("{sentence_1}", ".*?")\
        .replace("{sentence_2}",".*?").replace("{label_text}", ".*?").replace("\n\n", "\n").replace("  ", " ") + ")"
    sentence_1_reg_str = cfg.template[2:-1].replace("\\n",'\n').replace("{sentence_1}"," (.*?)")\
        .replace("{sentence_2}",".*?").replace("{label_text}", ".*?").replace("\n\n", "\n").replace("  "," ")

    if cfg.sentence_pair:
        sentence_2_reg_str = cfg.template[2:-1].replace("\\n", '\n').replace("{sentence_1}",".*?")\
            .replace("{sentence_2}", "(.*?)").replace("{label_text}", ".*?").replace("\n\n", "\n").replace("  "," ")

    label_reg_str = cfg.template[2:-1].replace("\\n",'\n').replace("{sentence_1}",".*?")\
        .replace("{sentence_2}",".*?").replace("{label_text}", "(.*?)").replace("\n\n", "\n").replace("  "," ")

    prompt_prefix = cfg.template[2:].split("{")[0]

    output = []
    for elem in generation_data['completion_texts']:
        input_examples = re.findall(example_reg_str, prompt_prefix+elem, re.S)
        for example in input_examples:
            sentence_1 = re.findall(sentence_1_reg_str,  example, re.S)
            #assert len(sentence_1) == 1, f"found {len(sentence_1)} sentence_1 in this example"
            if len(sentence_1) != 1: sentence_1 = ["\n"]
            sentence_1 = filter_printable(sentence_1[0])
            if cfg.sentence_pair:
                sentence_2 = re.findall(sentence_2_reg_str,  example, re.S)
                assert len(sentence_2) == 1, f"found {len(sentence_2)} sentence_2 in this example"
                sentence_2 = filter_printable(sentence_2[0])
            label = re.findall(label_reg_str,  example, re.S)
            assert len(label) == 1, f"found {len(label)} label in this example"
            label = filter_printable(label[0])

            label2index = {}
            for (k, v) in cfg.label_mapping.items():
                label2index[v] = str(k)
            if label in label2index:
                label = label2index[label]
            else:
                # if not exist, use random label
                label = list(cfg.label_mapping.keys())[0]

            if cfg.sentence_pair:
                if len(sentence_1.split()) >= min_length and len(sentence_2.split()) >= min_length:
                    extracted_data = {cfg.corpus_params['sentence_1_str']: sentence_1, cfg.corpus_params['sentence_2_str']: sentence_2, cfg.corpus_params['label_str']: label}
                    output.append(extracted_data)
            else:
                if len(sentence_1.split()) >= min_length:
                    extracted_data = {cfg.corpus_params['sentence_1_str']: sentence_1, cfg.corpus_params['label_str']: label}
                    output.append(extracted_data)

    return output


if __name__ == "__main__":
    fn = sys.argv[-1]
    data = pickle.load(open(fn, "rb"))
    cfg = easydict.EasyDict(data['config'])
    generation_data = data['result']

    output = augment_data_from_generation(generation_data=generation_data, cfg=cfg)

    fn = fn.replace("generate_", "augment_generate_")

    s = [json.dumps(elem) for elem in output]
    s = set(s)

    with open(f"{fn.replace('.pkl', '')}.jsonl", 'w') as fout:
        for elem in s:
            # s = json.dumps(elem)
            fout.write(elem+'\n')
