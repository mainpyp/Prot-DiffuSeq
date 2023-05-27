import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torch
import json
import itertools
import tqdm
import psutil
from transformers import AutoTokenizer
import datasets
from datasets import Dataset as Dataset2


def get_sentence_list(path):
    sentence_lst = {'src':[], 'trg': []}
    with open(path, 'r') as f_reader:
        for index, row in tqdm.tqdm(enumerate(f_reader)):
            line = json.loads(row)
            sentence_lst['src'].append(line['src'].strip())
            sentence_lst['trg'].append(line['trg'].strip())
    return sentence_lst

def create_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

    # adds 3di tokens that are lowercase
    threedi = ['a','c','d','e','f','g','h','i','k','l','m','n','p','q','r','s','t','v','w','y']
    tokenizer.add_tokens(threedi)

    return tokenizer

def tokenizing(sentence_lst, vocab_dict):

    raw_datasets = Dataset2.from_dict(sentence_lst)

    def tokenize_function(examples):
        input_id_x = tokenizer(examples['src'], add_special_tokens=True, truncation=True, max_length=128)['input_ids']
        input_id_y = tokenizer(examples['trg'], add_special_tokens=True, truncation=True, max_length=128)['input_ids']
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        keep_in_memory=False,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    
    print('### tokenized_datasets', tokenized_datasets)
    
if __name__ == "__main__":
    path = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/datasets/ProtMinimalSeqToStruc/train.jsonl"
    print(f"loading from {path}")
    print(f"\nCreate sentence list")
    sentence_list = get_sentence_list(path)
    print("\nCreate tokenizer")
    tokenizer = create_tokenizer()
    print("\nTokenizing")
    tokenizing(sentence_list, tokenizer)