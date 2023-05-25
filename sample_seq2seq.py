"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round
from diffuseq.text_datasets import load_data_text

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)


    ##### MODEL AND DIFFUSION #####
    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    ##### TOKENIZER AND EMBEDDING #####
    tokenizer = load_tokenizer(args)
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim, 
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)

    set_seed(args.seed2)

    print(f"### Sampling...on {args.split}")

    ##### DATA #####
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(),  # using the same embedding wight with tranining data
        loop=False
    )

    start_t = time.time()

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    
    print("### Generation will be saved to:\n", out_path)
    
    if os.path.isfile(out_path):
        print("### File already exists. Please delete it first if you want to force generation.")
        return

    all_test_data = []

    idx = 0

    try:
        while True:
            # cond keys: input_ids and input mask
            # input_ids shape: 50 (batch size), 256 (hidden dim)
            # we have the embeddings of n_batch_size sequences 
            batch, cond = next(data_valid)
    
            # Split data per nodes
            if idx % world_size == rank:  
                all_test_data.append(cond)
            idx += 1

    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(dist_util.dev())

    if idx % world_size and rank >= idx % world_size:
        # Dummy data for Remainder : for dist.barrier()
        all_test_data.append({})  

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    for cond in iterator:
        if not cond:  # Barrier for Remainder
            for i in range(world_size):
                dist.barrier()
            continue

        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        noise = th.randn_like(x_start) # init random noise
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        # only on sequence embedding noise
        x_noised = th.where(input_ids_mask == 0, x_start, noise)  # replaces sequence embedding with noise
        model_kwargs = {}

        if args.step >= args.diffusion_steps:
            print(f"step: {args.step} >= diffusion steps: {args.diffusion_steps}")
            args.use_ddim = False
            step_gap = 1
        else:
            print(f"OTHERWISE step: {args.step} == diffusion steps: {args.diffusion_steps}")
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step


        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        # this is the sampling function (takes most of the time)
        # len samples (list) 2k (len of diffusion steps t (?))
        print(f"Beginning sampling on {sample_shape[0]} sequences...")
        start_batch = time.time()
        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )
        end_batch = time.time()
        print(f"Sampling done in {end_batch - start_batch} seconds.")
        
        # sample shape: torch.Size([10, 256, 256])
        sample = samples[-1]

        # pass through lm_head of the TransformerNetModel
        # logits not normalized or rather output before softmax
        # logits shape torch.Size([10, 256, 50])
        # bsz, seqlen, vocab
        logits = model.get_logits(sample)  

        # Returns the k (1) largest elements of the given 
        # input tensor along a given dimension.
        cands = th.topk(logits, k=1, dim=-1)

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        # tokenizer = load_tokenizer(args)

        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            # tokens = tokenizer.decode_token(seq)
            len_x = args.seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        for i in range(world_size):
            if i == rank:  # Write files sequentially
                fout = open(out_path, 'a')  # appends to file    
                # The dataset is split into batches, so the length of the word_lst_recover is 50
                print(f"i: {i}, rank: {rank}")
                for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                    print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
                fout.close()
            dist.barrier()

    logger.log(f'### Written the decoded output to: {out_path}')
        
    # Get AF IDs from test dataset
    dataset_dir = os.path.join(args.data_dir, args.split + '.jsonl')
    logger.log(f"Extracting AF IDs")
    af_ids = []
    contains_af_id = True
    with open(dataset_dir, 'r') as f:
        for line in f:
            if "af_id" not in line:
                logger.log(f"Dataset does not contain af_id. Aborting...")
                contains_af_id = False
                break
            af_ids.append(json.loads(line)["af_id"])
    
    if contains_af_id:
        assert os.path.isfile(out_path), f"Output file {out_path} does not exist"
        with open(out_path, 'r') as f:
            lines = f.readlines()
            lines = [json.loads(line) for line in lines]

            if not len(lines) == len(af_ids): 
                print(f"CAUTION: Number of lines ({len(lines)}) does not match number of af_ids ({len(af_ids)})")
            
            for line in lines:
                line["af_id"] = af_ids.pop(0)

        with open(out_path, 'w') as f:
            # converts to string
            lines = [json.dumps(line) + "\n" for line in lines]
            f.writelines(lines)
        logger.log(f'### Written the decoded output with af ids to {out_path}')
    else:
        print("No af_id found in dataset. Skipping extraction of af_ids")
    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    


if __name__ == "__main__":
    main()
