import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from gzsl.classification.util.preprocessing import draw_matrix
from collections import OrderedDict


class IntentDataset(Dataset):

    def __init__(self, df, tokenizer, intents, descriptions, concepts, actions, description_first,
                 uttr_len, desc_len,similarity_matrix_intent=None,similarity_matrix_utter=None, mlm_percent=0.15,sampling_strategy='intents', neg_k=None,
                 infer_intents=None,train_or_evaluate="train"):
        self.intents = intents
        self.uttrs = [tokenizer(text, padding='max_length', truncation=True, max_length=uttr_len)
                      for text in df.text.values]

        self.descriptions = [tokenizer(descriptions[intent], padding='max_length', truncation=True, max_length=desc_len)
                             for intent in intents]

        self.labels = [intents.index(intent) for intent in df.intents.values]

        self.num_intents = len(intents)
        self.num_uttrs = len(self.uttrs)
        self.tokenizer = tokenizer
        self.neg_k = neg_k
        self.infer_intents = infer_intents
        if infer_intents:
            self.infer_intents_ids = [intents.index(intent) for intent in infer_intents]
        self.description_first = description_first
        self.sampling_strategy = sampling_strategy
        self.mlm_percent=mlm_percent
        if similarity_matrix_intent is None:
            similarity_matrix_intent = np.eye(self.num_intents)


        self.intent_similarity = self.additive_smoothing(similarity_matrix_intent, self.num_intents)
        self.similar_uttrs = similarity_matrix_utter


        self.uttr_len=uttr_len
        self.desc_len=desc_len
        self.train_or_evaluate=train_or_evaluate

    def __len__(self):
        return self.num_uttrs

    def mask_tokens(self, inputs, special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs == 0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels

    def encode_nli_pair(self, uttr_idx, intent_idx):

        first = self.descriptions[intent_idx] if self.description_first else self.uttrs[uttr_idx]
        second = self.uttrs[uttr_idx] if self.description_first else self.descriptions[intent_idx]
        input_ids = first['input_ids'] + [self.tokenizer.sep_token_id] + second['input_ids'][1:]
        attention_mask = first['attention_mask'] + [1] + second['attention_mask'][1:]

        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask)




    def encode_uttr(self, uttr_id):
        return torch.LongTensor(self.uttrs[uttr_id]["input_ids"]), torch.LongTensor(self.uttrs[uttr_id]["attention_mask"])


    def __getitem__(self, idx):
        if self.neg_k:
            chosen_uttrs = [idx]

            chosen_intents = np.random.choice(
                np.arange(self.num_intents), p=self.intent_similarity[self.labels[idx]],
                size=self.neg_k + 1, replace=False
            )
            candidates = self.similar_uttrs[idx] or \
                         [i for i, uttr in enumerate(range(self.num_uttrs)) if
                          self.labels[i] != self.labels[idx]]
            chosen_uttrs += np.random.choice(candidates, size=self.neg_k).tolist()



            u_ids, u_attention_mask = zip(*[self.encode_nli_pair(u, self.labels[idx]) for u in chosen_uttrs])
            i_ids, i_attention_mask = zip(*[self.encode_nli_pair(idx, i) for i in chosen_intents])


            uttr_only_ids, uttr_only_attention_mask = zip(*[self.encode_uttr(u) for u in chosen_uttrs])
            uttr_only_ids = torch.stack(uttr_only_ids)
            uttr_only_attention_mask = torch.stack(uttr_only_attention_mask)

            u_ids = torch.stack(u_ids)
            u_attention_mask = torch.stack(u_attention_mask)
            i_ids = torch.stack(i_ids)
            i_attention_mask = torch.stack(i_attention_mask)

            uttr_ids = torch.cat([u_ids,i_ids],dim=0)
            uttr_attention_mask = torch.cat([u_attention_mask,i_attention_mask],dim=0)

            right_position_index = np.argmax([float(self.labels[u] == self.labels[idx]) for u in chosen_uttrs])
            right_input_ids = uttr_ids[right_position_index].unsqueeze(0)
            right_attention_mask = uttr_attention_mask[right_position_index].unsqueeze(0)

            mask_input_ids, mask_lb = self.mask_tokens(right_input_ids.clone().cpu(), mlm_probability=self.mlm_percent)


            sample = {
                'pair_ids': uttr_ids,
                'pair_attention_mask': uttr_attention_mask,

                'mask_input_ids': mask_input_ids.squeeze(0),
                'attention_mask': right_attention_mask.squeeze(0),
                'mask_lb': mask_lb,

                'uttr_only_ids': uttr_only_ids,
                'uttr_only_attention_mask': uttr_only_attention_mask,
                'right_input_ids': torch.LongTensor(self.uttrs[idx]["input_ids"]),
                'right_attention_mask': torch.LongTensor(self.uttrs[idx]["attention_mask"]),

                'embed_enc': torch.tensor([float(u == idx) for u in chosen_uttrs]),
                'label': self.labels[idx],

                'label_enc': torch.tensor([float(self.labels[u] == self.labels[idx]) for u in chosen_uttrs]+[float(self.labels[idx] == i) for i in chosen_intents]),
            }
        else:
            chosen_uttrs = [idx]
            chosen_intents = np.arange(self.num_intents)
            uttr_ids, uttr_attention_mask = zip(*[self.encode_nli_pair(u, i) for u in chosen_uttrs for i in chosen_intents])


            sample = {
                'pair_ids': torch.stack(uttr_ids),
                'pair_attention_mask': torch.stack(uttr_attention_mask),
                'label': self.labels[idx],
                'label_enc': torch.tensor([float(self.labels[u] == i) for u in chosen_uttrs for i in chosen_intents]),
            }
        return sample

    @staticmethod
    def additive_smoothing(matrix, n_categories, alpha=0.001):
        for i in range(matrix.shape[0]):
            n = matrix[i].sum()
            matrix[i] = (matrix[i] + alpha) / (n + alpha * n_categories)
        return matrix




def get_dataset(cfg, df, tokenizer, intents, descriptions,
                concepts, actions,train_or_evaluate,mlm_percent=0.15,similarity_matrix_intent=None,similarity_matrix_utter=None,sampling_strategy='intents', k_neg=None, infer_intents=None) -> IntentDataset:
    if cfg.model.model_type == 'nli_ca' or cfg.model.model_type == 'nli_strict':
        return IntentDataset(
            df, tokenizer, intents, descriptions, concepts, actions,
            cfg.experiment.intent_desc_first, cfg.dataset.uttr_len, cfg.dataset.desc_len,
            similarity_matrix_intent,similarity_matrix_utter,mlm_percent, sampling_strategy,k_neg, infer_intents,train_or_evaluate
        )
    else:
        ValueError("Unknown model type")
