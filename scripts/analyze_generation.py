"""This script analyses the generation of a model. 
It takes as input the path to the generation folder where the generations of a 
model for each checkpoint are stored.

--input_path: path to the generation folder

/mnt/home/mheinzinger/deepppi1tb/foldseek/foldseek_latest/foldseek/bin/foldseek easy-search \
    /mnt/home/mheinzinger/deepppi1tb/ProSST5/analysis_prosst5/reference_structures/AFDB/val/ \
    /path/to/output/pdb \
    /path/to/foldseek_results/aln.m8 \
    /path/to/foldseek_results/tmpFolder --format-output "query,target,pident,evalue,bits,alntmscore,lddt" \
    --exhaustive-search 1
"""
import argparse
import glob
import json
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    args = parser.parse_args()
    return args


def create_esm_predictions(input_path: str) -> None:
    """Creates predictions for the ESM model. By calling a different file."""

    # list of checkpints (FOLDERS in dir)
    checkpoints = sorted(glob.glob(f"{input_path}/*.samples"))[::-1]

    for checkpoint in checkpoints:

        assert os.path.isdir(checkpoint), f'{checkpoint} not found'

        def get_seed(x): return x.split("/")[-1].split(".")[0]
        seeds = [get_seed(x) for x in glob.glob(f"{checkpoint}/*.fasta")]
        print(seeds)
        for seed in seeds:

            # create output pdb folder
            fasta_path = os.path.join(checkpoint, seed + ".fasta")
            pdb_path = os.path.join(checkpoint, seed + "_pdb")

            assert os.path.isfile(fasta_path), f'{fasta_path} not found'

            # run ESMFold if not already done
            if not os.path.isdir(pdb_path):
                print("\n\nPDB output in: ", pdb_path)
                COMMAND = f'python /mnt/home/mheinzinger/deepppi1tb/ESMFold/esm/scripts/fold.py ' \
                    f' -i {fasta_path} ' \
                    f'-o {pdb_path}'
                print("\n\nRunning ESM prediction: ", COMMAND)

                # os.system(COMMAND)

                pLDDT_path = os.path.join(checkpoint, seed + "_esmfold_pLDDT.csv")
                assert os.path.isfile(pdb_path), f'{pdb_path} not found'
                COMMAND = f'python /mnt/home/mheinzinger/deepppi1tb/collabfold/parse_pLDDT.py {pdb_path}    {pLDDT_path}'
                print("\n\nParsing pLDDT: ", COMMAND)
                # os.system(COMMAND)
            else:
                print(
                    f"ESM prediction for {seed} already exists. Please refer to {pdb_path} or delete this folder to generate new predictions.")


if __name__ == "__main__":
    args = parse_arguments()
    create_esm_predictions(args.input_path)
