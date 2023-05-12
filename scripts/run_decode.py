import os, sys, glob
import argparse, json
from decode_utils import remove_brackets
sys.path.append('.')
sys.path.append('..')


def parse_arguments():
    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--model_dir', type=str, default='', help='path to the folder of diffusion model')
    parser.add_argument('--n_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--seeds', type=int, nargs='+', default=[101], help='random seed')
    parser.add_argument('--step', type=int, default=2000, help='if less than diffusion training steps, like 1000, use ddim sampling')

    parser.add_argument('--bsz', type=int, default=50, help='batch size')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'], help='dataset split used to decode')

    parser.add_argument('--top_p', type=int, default=-1, help='top p used in sampling, default is off')
    parser.add_argument('--pattern', type=str, default='ema', help='training pattern')
    
    return parser.parse_args()

def generate_samples(args: argparse.Namespace, out_dir: str = 'generation_outputs'):
    """ This function is used to generate samples from all checkpoints 
    in a folder and all given seeds.
    """
    for lst in glob.glob(args.model_dir):
        print(lst)
        checkpoints = sorted(glob.glob(f"{lst}/{args.pattern}*.pt"))[::-1]

        if not os.path.isdir(out_dir):
            print(f'creating {out_dir}...')
            os.mkdir(out_dir)

        ##### GENERATE SAMPLES #####
        print(f"\nGenerating samples for {len(checkpoints)} checkpoints and {len(args.seeds)} seeds...\n")
        for checkpoint_one in checkpoints:

            assert os.path.isfile(checkpoint_one), f'{checkpoint_one} not found'

            for seed in args.seeds:
                COMMAND = f'python -m torch.distributed.launch --nproc_per_node={args.n_gpus} --master_port={12233 + int(seed)} --use_env sample_seq2seq.py ' \
                f'--model_path {checkpoint_one} --step {args.step} ' \
                f'--batch_size {args.bsz} --seed2 {seed} --split {args.split} ' \
                f'--out_dir {out_dir} --top_p {args.top_p} '
                print("Running: ", COMMAND)
                
                os.system(COMMAND)


def convert_to_fasta(all_generated_files: list) -> list:
    generated_fastas = []
    for path in all_generated_files:
        # load generation with keys ref, rec, source and af_id
        print("Reading generation from ", path)
        with open(path, "r") as f:
            jsonl = [json.loads(line) for line in f]

        # write reference and recovery to fasta files
        ref_path = path.replace(".json", "_ref.fasta")
        print("Writing reference file to ", ref_path)
        with open(ref_path, "w") as f:
            for seq in jsonl:
                f.write(f">{seq['af_id']}\n{remove_brackets(seq['reference'])}\n")

        rec_path = path.replace(".json", "_rec.fasta")
        print("Writing recovery file to ", rec_path)
        with open(rec_path, "w") as f:
            for seq in jsonl:
                f.write(f">{seq['af_id']}\n{remove_brackets(seq['recovery'])}\n")
        
        generated_fastas.append(ref_path)
        generated_fastas.append(rec_path)

    return generated_fastas


if __name__ == '__main__':

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    args = parse_arguments()
    out_dir = 'generation_outputs'

    #generate_samples(args, out_dir=out_dir)

    #### GET ALL GENERATED FILES ####
    for lst in glob.glob(args.model_dir):
        checkpoints = sorted(glob.glob(f"{lst}/{args.pattern}*.pt"))[::-1]
        parse_path = lambda x: "/".join(x.split('/')[-3:]).replace(".pt", ".pt.samples/").replace("diffusion_models", out_dir)
        generations = [parse_path(ckpt) for ckpt in checkpoints]
        all_generated_files = sorted(glob.glob(f"{generations[0]}*.json"))[::-1]

    #### CONVERT GENERATED JSON TO FASTA ####
    all_generated_fastas = convert_to_fasta(all_generated_files)

    ######################
    #### RUN ESM FOLD ####
    ######################

    print('#'*30, 'decoding finished...')