# import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torch
import json
import gc
import itertools
import tqdm
import psutil
import datasets
from datasets import Dataset as Dataset2
from datasets import load_dataset

def load_data_text(
    batch_size, 
    seq_len, 
    deterministic=False, 
    data_args=None, 
    model_emb=None,
    split='train', 
    loaded_vocab=None,
    loop=True,
    af_ids_int=None,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#'*30, '\nLoading text data...')

    training_data = get_corpus(data_args, seq_len, split=split, loaded_vocab=loaded_vocab)
    print("TEST SET")

    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb
    )
    
    if split != 'test':
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            # drop_last=True,
            sampler=sampler,
            # shuffle=True,
            num_workers=1,
        )
        
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            # drop_last=True,
            # sampler=sampler, # this was commented out
            shuffle=True,
            num_workers=1,
        )

    if loop:
        print("INFITE LOADER")
        return infinite_loader(data_loader)
    else:
        print("ITERABLE")
        return iter(data_loader)

def infinite_loader(data_loader):
    while True:
        yield from data_loader

def helper_tokenize(sentence_lst, vocab_dict, seq_len, preload: bool = True, split: str = None):
    """ sentence_lst: 
                    keys: 'src', 'trg', sometimes: 'af_id'
                    value: shape (bsz, len)
        vocab_dict: vocab_dict is a dict of word2id
    """

    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    
    

    if preload and split == 'train':
        print("#"*72)
        print(f"# LOADING DATASET FROM THE HUB, SET PRELOAD TO NONE TO DISABLE THAT BEHAVIOUR ({split}) #")
        print("#"*72)
        
        print(f"FREE UP SPACE: DELETING SENTECE LIST")
        del sentence_lst
        gc.collect()
        
        tokenized_datasets = load_dataset("adrianhenkel/tokenized-total-512-reduced", 
                                          cache_dir="/datacontainer/.cache", split="train")
        
        print(f"LOADING COMPLETE: {tokenized_datasets}")
    else:
         # Dataset2 is the the dataset from huggingface and not from torch.utils.data
        print("#"*72)
        print(f"# CREATING DATASET FROM SCRATCH PRELOAD: {preload} ({split})#")
        print("#"*72)
         
        raw_datasets = Dataset2.from_dict(sentence_lst)
        
        print(raw_datasets)
        print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        def tokenize_function(examples):
            print(len(examples["src"]))
            input_id_x = vocab_dict.encode_token(examples['src'])
            input_id_y = vocab_dict.encode_token(examples['trg'])
            result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

            return result_dict
        
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=['src', 'trg'],
            desc="Running tokenizer on dataset",
        )
        
        del sentence_lst
        gc.collect()
        
        
        
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets shape', tokenized_datasets.shape)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    
    def merge_and_mask(group_lst):
        lst = []
        mask = []

        half_size = (seq_len - 3) // 2
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]
            
            src = group_lst['input_id_x'][i][:half_size]
            trg = group_lst['input_id_y'][i][:half_size]
            src.append(end_token)
            trg.append(end_token)

            lst.append(src + [vocab_dict.sep_token_id] + trg)
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst
    
    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        desc=f"merge and mask",
    )
    
    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        desc=f"padding",
    )
    
    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    del tokenized_datasets
    del lm_datasets
    gc.collect()
    
    print(f"RAM used after deleting: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None):

    print('#'*30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))
    
    sentence_lst = {'src':[], 'trg': []}

    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train.jsonl'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/valid.jsonl'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"


    af_ids = []
    with open(path, 'r') as f_reader:
        for index, row in tqdm.tqdm(enumerate(f_reader)):
            line = json.loads(row)
            sentence_lst['src'].append(line['src'].strip())
            sentence_lst['trg'].append(line['trg'].strip())
            if 'af_id' in line.keys():
                # check if af in sentence_lst and if not add it
                af_ids.append(line['af_id'].strip())

    # create af id lookup dict, each id gets a unique int
    af_ids_lookup = {af_id: idx for idx, af_id in enumerate(af_ids)}
    af_ids_int = [af_ids_lookup[af_id] for af_id in af_ids]
    
    if len(af_ids_int) > 0:
        sentence_lst['af_ids_int'] = af_ids_int
    
    if "af_ids_int" in sentence_lst.keys():
        print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2], sentence_lst['af_ids_int'][:2])
    else:
        print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2])
        
    # get tokenizer.
    vocab_dict = loaded_vocab

    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len, preload=True, split=split)
    
    del sentence_lst
    gc.collect()
    
    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            input_ids = self.text_datasets['train'][idx]['input_ids']
            hidden_state = self.model_emb(torch.tensor(input_ids))

            # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
            arr = np.array(hidden_state, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])
            if 'af_ids_int' in self.text_datasets['train'][idx].keys():
                out_kwargs['af_ids_int'] = np.array(self.text_datasets['train'][idx]['af_ids_int'])

            return arr, out_kwargs

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int8).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int8).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result
