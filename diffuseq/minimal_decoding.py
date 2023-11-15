import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict
from torch.utils.data.distributed import DistributedSampler

import json
import tqdm

from transformers import AutoTokenizer, T5Tokenizer
import datasets
import os
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
    # tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)

    # adds 3di tokens that are lowercase
    # threedi = ['a','c','d','e','f','g','h','i','k','l','m','n','p','q','r','s','t','v','w','y']
    # tokenizer.add_tokens(threedi)

    return tokenizer


def tokenizing(paths, tokenizer):
    
    def tokenize_function(examples):
            input_id_x = tokenizer(examples['src'], add_special_tokens=True, truncation=True)['input_ids']
            input_id_y = tokenizer(examples['trg'], add_special_tokens=True, truncation=True)['input_ids']
            result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

            return result_dict
    datadict = DatasetDict()
    for path in paths:
        print(f"### path: {path}")
        sentence_lst = get_sentence_list(path)
        raw_datasets = Dataset2.from_dict(sentence_lst)

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=['src', 'trg'],
            keep_in_memory=True,
            desc="Running tokenizer on dataset"
        )
        
        if "train" in path:
            datadict['train'] = tokenized_datasets
        elif "test" in path:
            datadict['test'] = tokenized_datasets
        elif "valid" in path:
            datadict['valid'] = tokenized_datasets
        else:
            raise ValueError("Path not valid")
    print('### datadict', datadict)
        
    datadict.push_to_hub("Rostlab/ProstT5Dataset")
    print('### tokenized_datasets', tokenized_datasets)
    
if __name__ == "__main__": 
    path_train = "datasets/ProtTotalCorrect/train.jsonl"
    path_test = "datasets/ProtTotalCorrect/test_474.jsonl"
    path_valid = "datasets/ProtTotalCorrect/val_474.jsonl"
    
    paths = [path_test, path_valid, path_train]
    print("\nCreate tokenizer")
    tokenizer = create_tokenizer()
    print("\nTokenizing")
    tokenizing(paths, tokenizer)
