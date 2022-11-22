import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


class ImmutableLM(nn.Module):
    def __init__(self, model_path):
        super(ImmutableLM, self).__init__()

        if ('gpt-2' in model_path) or ('gpt2' in model_path):
            self.backbone = GPT2LMHeadModel.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.backbone_name = model_path
        else:
            if 'gpt-neo-' in model_path:
                prefix = 'EleutherAI/'

            self.backbone = AutoModelForCausalLM.from_pretrained(prefix + model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(prefix + model_path)
            self.backbone_name = model_path.split('/')[-1]

    def get_restricted_token_probability(self, logits, restricted_token, label_length=1, normalize=False):
        prob_dist = logits[:, -label_length:, :].squeeze().softmax(-1)
        restricted_token = list(restricted_token)
        prob_dist = prob_dist[:, torch.LongTensor(restricted_token)]
        if normalize:
            prob_dist /= prob_dist.min()

        prediction_prob, prediction_index = prob_dist.topk(1)
        prediction = [restricted_token[elem.squeeze()] for elem in prediction_index]
        # prediction = restricted_token[prediction_index]
        prediction_prob = [elem.item() for elem in prediction_prob]
        return prediction, prediction_prob, prob_dist

    def forward(self, data, restricted_token, label_length=1):
        """
        This is a lazy version without support for batch inference.
        """
        input_sequence = data["input_sequence"].to(self.backbone.device)
        if len(input_sequence.shape) == 3:
            input_sequence = input_sequence.squeeze(0)
        with torch.no_grad():
            # this is a super lazy/ugly implementation for OOM issue, refactor if have time
            split = (input_sequence.shape[0] > 20) and (self.backbone_name == 'gpt2-xl')
            if split:
                logits = []
                for sub_sequence in torch.split(input_sequence, 6, dim=0):
                    outputs = self.backbone(sub_sequence)
                    logits.append(outputs[0])
                logits = torch.cat(logits, dim=0)
            else:
                outputs = self.backbone(input_sequence)
                logits = outputs[0]

            prediction, prediction_prob, prob_dist = self.get_restricted_token_probability(logits=logits,
                                                                                restricted_token=restricted_token,
                                                                            label_length=label_length)


        return {"prediction_token": prediction,
                "prediction_text": self.tokenizer.decode(prediction),
                "prediction_prob": prediction_prob,
                "prediction_dist": prob_dist.tolist(),
                "label": data["label"],
                "prompt_examples": data["train_metadata"],
                "prompt_sequence_text": data["raw_sequence"],
                "prompt_sequence_index": data["input_sequence"]}

    @staticmethod
    def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
        return generated_ngrams

    @staticmethod
    def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - ngram_size
        ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
        return banned_ngrams.get(ngram_idx, [])

    def _calc_banned_ngram_tokens(self,
            ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int):
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]

        generated_ngrams = self._get_ngrams(ngram_size, prev_input_ids, num_hypos)

        banned_tokens = [
            self._get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
            for hypo_idx in range(num_hypos)
        ]

        return banned_tokens

    def forbid_ngram_wrapper(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, ngram_size: int, allowed_tokens: list) -> torch.FloatTensor:
        if ngram_size > 0:
            num_batch_hypotheses = scores.shape[0]
            cur_len = input_ids.shape[-1]
            banned_batch_tokens = self._calc_banned_ngram_tokens(ngram_size, input_ids, num_batch_hypotheses, cur_len)
            banned_batch_tokens_filtered = []
            for banned_tokens in banned_batch_tokens:
                banned_batch_tokens_filtered.append(list(set(banned_tokens) - set(allowed_tokens)))
            #if banned_batch_tokens[0]:
            #    raise ValueError

            for i, banned_tokens in enumerate(banned_batch_tokens_filtered):
                scores[i, banned_tokens] = -float("inf")
        return scores

    @staticmethod
    def temperature_wrapper(scores: torch.FloatTensor, temperature: float, allowed_tokens: list) -> torch.FloatTensor:
        # if allowed token has max prob, then do argmax generation, otherwise do sample
        for i in range(len(scores)):
            if scores[i].argmax() in allowed_tokens:
                scores[i, scores[i].argmax()] = 99999

        return scores / temperature

    @staticmethod
    def topk_wrapper(scores: torch.FloatTensor, top_k: int) -> torch.FloatTensor:
        if top_k > 0:
            filter_value = -99999
            indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
            scores = scores.masked_fill(indices_to_remove, filter_value)
        # else do nothing
        return scores

    @torch.no_grad()
    def inference_generation_oom(self, input_sequence):
        bsz, seq_len = input_sequence.shape

        # split = (bsz > 20) and (self.backbone_name in ['gpt2-xl', 'gpt2-large'])
        split = 4
        if self.backbone_name in ['gpt2-xl', 'gpt2-large']:
            split = 2
        else:
            split = 0
        if input_sequence.shape[1] > 700 and self.backbone_name in ['gpt2-medium', 'gpt2-xl', 'gpt2-large']:
            split = 1


        example_per_batch = 1
        if split > 0:
            scores = []
            example_per_batch = split

            for sub_sequence in torch.split(input_sequence, example_per_batch, dim=0):
                outputs = self.backbone(sub_sequence, output_attentions=True, return_dict=True)
                scores.append(outputs['logits'][:, -1, :])
            scores = torch.cat(scores, dim=0)
        else:
            output = self.backbone(input_sequence, output_attentions=True, return_dict=True)
            scores = output['logits'][:, -1, :]
        return scores

    @torch.no_grad()
    def generate(self, data, max_length=128,
                 no_repeat_ngram_size=0, allowed_tokens=(), prefix="",
                 temperature=1.0, do_sample=False, top_k=-1):

        assert temperature > 0, "temperature must be positive"
        if not do_sample and temperature != 1.0:
            print("sampling is disabled, tune temperature will not work")

        if len(data['input_sequences_prompt'].shape) == 3:
            input_sequences = data["input_sequences_prompt"].squeeze(dim=0).to(self.backbone.device)
        else:
            input_sequences = data["input_sequences_prompt"].to(self.backbone.device)

        bsz = input_sequences.shape[0]
        if prefix:
            additional_prompt = torch.LongTensor(self.tokenizer.encode(prefix.strip(" "))).unsqueeze(dim=0).repeat(bsz, 1)
            input_sequences = torch.cat((input_sequences, additional_prompt.to(self.backbone.device)), dim=-1)

        running_sequences = input_sequences.clone()
        running_sequences = running_sequences[:, -1000:]
        completion_tokens = []
        attentions = []
        for step in tqdm(range(max_length)):
            with torch.no_grad():
                scores = self.inference_generation_oom(running_sequences)

            scores = self.forbid_ngram_wrapper(running_sequences, scores,
                                               ngram_size=no_repeat_ngram_size,
                                               allowed_tokens=allowed_tokens)

            scores = self.temperature_wrapper(scores=scores,
                                              temperature=temperature,
                                              allowed_tokens=allowed_tokens)

            scores = self.topk_wrapper(scores=scores, top_k=top_k)

            if do_sample:
                prediction_tokens = torch.multinomial(scores.softmax(dim=-1), num_samples=1)
            else:
                _, prediction_tokens = scores.topk(1)

            # _, prediction_tokens = scores.topk(1)

            running_sequences = torch.cat((running_sequences, prediction_tokens), dim=1)
            completion_tokens.append(prediction_tokens.squeeze())

            running_sequences = running_sequences[:, -1000:]

        prompt_tokens = input_sequences.cpu()
        prompt_plus_completion_tokens = running_sequences.cpu()
        completion_tokens = torch.stack(completion_tokens, dim=1).cpu()

        prompt_texts = [self.tokenizer.decode(sentence) for sentence in prompt_tokens]
        prompt_plus_completion_texts = [self.tokenizer.decode(sentence) for sentence in prompt_plus_completion_tokens]
        completion_texts = [self.tokenizer.decode(elem) for elem in completion_tokens]

        return {"prompt_plus_completion_tokens": prompt_plus_completion_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_plus_completion_texts": prompt_plus_completion_texts,
                "prompt_texts": prompt_texts,
                "completion_texts": completion_texts,
                "attentions": attentions,
                "prompt_examples": data["train_metadata"]}

    @torch.no_grad()
    def balance_generate(self, data, max_length=256,
                 no_repeat_ngram_size=0, allowed_tokens=(), prefix="",
                 temperature=1.0, do_sample=False, top_k=-1):

        assert temperature > 0, "temperature must be positive"
        if not do_sample and temperature != 1.0:
            print("sampling is disabled, tune temperature will not work")

        input_sequences_all = [torch.cat(elem) for elem in data['train_prompts_ids']]
        results = []
        for input_sequences_prompt in input_sequences_all:
            if len(input_sequences_prompt.shape) == 3:
                input_sequences = input_sequences_prompt.squeeze(dim=0).to(self.backbone.device)
            elif len(input_sequences_prompt.shape) == 1:
                input_sequences = input_sequences_prompt.unsqueeze(dim=0).to(self.backbone.device)
            else:
                input_sequences = input_sequences_prompt.to(self.backbone.device)

            running_sequences = input_sequences.clone()
            running_sequences = running_sequences.repeat([10, 1])

            bsz = running_sequences.shape[0]
            if prefix:
                additional_prompt = torch.LongTensor(self.tokenizer.encode(prefix.strip(" "))).unsqueeze(dim=0).repeat(bsz, 1)
                running_sequences = torch.cat((running_sequences, additional_prompt.to(self.backbone.device)), dim=-1)

            completion_tokens = []
            attentions = []
            for step in tqdm(range(max_length)):
                with torch.no_grad():
                    scores = self.inference_generation_oom(running_sequences)

                scores = self.forbid_ngram_wrapper(running_sequences, scores,
                                                   ngram_size=no_repeat_ngram_size,
                                                   allowed_tokens=allowed_tokens)

                scores = self.temperature_wrapper(scores=scores,
                                                  temperature=temperature,
                                                  allowed_tokens=allowed_tokens)

                scores = self.topk_wrapper(scores=scores, top_k=top_k)

                if do_sample:
                    prediction_tokens = torch.multinomial(scores.softmax(dim=-1), num_samples=1)
                else:
                    _, prediction_tokens = scores.topk(1)

                running_sequences = torch.cat((running_sequences, prediction_tokens), dim=1)
                completion_tokens.append(prediction_tokens.squeeze())

            prompt_tokens = input_sequences.cpu()
            prompt_plus_completion_tokens = running_sequences.cpu()
            completion_tokens = torch.stack(completion_tokens, dim=1).cpu()

            prompt_texts = [self.tokenizer.decode(sentence) for sentence in prompt_tokens]
            prompt_plus_completion_texts = [self.tokenizer.decode(sentence) for sentence in prompt_plus_completion_tokens]

            completion_texts = [self.tokenizer.decode(elem) for elem in completion_tokens]

            result = {"prompt_plus_completion_tokens": prompt_plus_completion_tokens,
                      "prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "prompt_plus_completion_texts": prompt_plus_completion_texts,
                      "prompt_texts": prompt_texts,
                      "completion_texts": completion_texts,
                      "attentions": attentions}
            results.append(result)

        from collections import defaultdict
        d = defaultdict(list)
        for key in results[0]:
            for result in results:
                d[key].append(result[key])
        return d



if __name__ == "__main__":
    lm = ImmutableLM(model_path='distilgpt2')


