import torch
# bert results
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, GPT2TokenizerFast
import sys, yaml, os
import json

import numpy as np

def get_knn(model_emb, text_emb, dist='cos'):
    """This function is apparently not used since it is called in round_func"""
    if dist == 'cos':
        adjacency = model_emb @ text_emb.transpose(1, 0).to(model_emb.device)
    elif dist == 'l2':
        # here we calculate the difference between model and text embedding 
        # this is saved as adjacency
        adjacency = model_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - \
        text_emb.unsqueeze(0).expand(model_emb.size(0), -1, -1)
        # This is the second part of the euclidean distance calculation
        # root of sum of squares of adjacent vectors (difference)
        # boils down to that: norm = lambda x: torch.sqrt(torch.sum(x**2))
        # https://de.wikipedia.org/wiki/Frobeniusnorm
        adjacency = -torch.norm(adjacency, dim=-1)

    topk_out = torch.topk(adjacency, k=6, dim=0)
    return topk_out.values, topk_out.indices

def get_efficient_knn(model_emb, text_emb):
    # text embed shape text emb shape: torch.Size([12800, 256])
    # model emb shape torch.Size([50, 256])
    print("text emb shape", text_emb.shape)
    print("model emb shape", model_emb.shape)
    # emb norm shape torch.Size([50, 1])
    emb_norm = (model_emb**2).sum(-1).view(-1, 1) # vocab
    print("emb norm shape", emb_norm.shape)
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) # d, bsz*seqlen
    print("text emb t shape", text_emb_t.shape)
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) # bsz*seqlen, 1
    print("arr norm shape", arr_norm.shape)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t) # (vocab, d) x (d, bsz*seqlen)
    print("dist shape", dist.shape)
    dist = torch.clamp(dist, 0.0, np.inf)
    print("dist shape", dist.shape)
    # topk returns the largest k elements of the given input tensor along a given dimension.
    topk_out = torch.topk(-dist, k=1, dim=0)

    print(f"{len(topk_out.values)} {len(topk_out.indices)}")
    print(f"{len(topk_out.values[0])} {len(topk_out.indices[0])}")
    print(topk_out.values, "\n" ,topk_out.indices)
    return topk_out.values, topk_out.indices

def rounding_func(text_emb_lst, model, tokenizer, emb_scale_factor=1.0):
    """This function is apparently NOT used in the code."""
    decoded_out_lst = []
    
    model_emb = model.weight  # input_embs
    down_proj_emb2 = None

    # L2 also called Euclidean distance is calculated as 
    # the square root of the sum of the squared differences between the two vectors.
    dist = 'l2'
    
    for text_emb in text_emb_lst:
        import torch
        text_emb = torch.tensor(text_emb)
        # print(text_emb.shape)
        if len(text_emb.shape) > 2:
            text_emb = text_emb.view(-1, text_emb.size(-1))
        else:
            text_emb = text_emb
        val, indices = get_knn((down_proj_emb2 if dist == 'cos' else model_emb),
                                text_emb.to(model_emb.device), dist=dist)
    
        decoded_out_lst.append(tokenizer.decode_token(indices[0]))

    return decoded_out_lst

def compute_logp(args, model, x, input_ids):
    word_emb = model.weight
    sigma = 0.1
    if args.model_arch == '1d-unet':
        x = x.permute(0, 2, 1)

    bsz, seqlen, dim = x.shape

    x_flat = x.reshape(-1, x.size(-1)).unsqueeze(0)  # 1, bsz*sample*seqlen, dim
    word_emb_flat = word_emb.unsqueeze(1)  # vocab, 1,  dim
    diff = (x_flat - word_emb_flat) ** 2  # vocab, seqlen, dim

    logp_expanded = -diff.sum(dim=-1) / (2 * sigma ** 2)  # vocab, seqlen
    logp_expanded = logp_expanded.permute((1, 0))

    ce = torch.nn.CrossEntropyLoss(reduction='none')
    loss = ce(logp_expanded, input_ids.view(-1)).view(bsz, seqlen)

    return loss

def get_weights(model, args):
    if hasattr(model, 'transformer'):
        input_embs = model.transformer.wte  # input_embs
        down_proj = model.down_proj
        model_emb = down_proj(input_embs.weight)
        print(model_emb.shape)
        model = torch.nn.Embedding(model_emb.size(0), model_emb.size(1))
        print(args.emb_scale_factor)
        model.weight.data = model_emb * args.emb_scale_factor

    elif hasattr(model, 'weight'):
        pass
    else:
        assert NotImplementedError
        
    model.weight.requires_grad = False
    return model

def denoised_fn_round(args, model, text_emb, t):
    # print(text_emb.shape) # bsz, seqlen, dim
    model_emb = model.weight  # input_embs
    # print(t)
    old_shape = text_emb.shape
    old_device = text_emb.device

    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    # val, indices = get_knn(model_emb, text_emb.to(model_emb.device), dist=dist)
    val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
    rounded_tokens = indices[0]
    # print(rounded_tokens.shape)
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)

    return new_embeds