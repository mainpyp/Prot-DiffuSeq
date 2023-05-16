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
    parser.add_argument('-i', '--input_path', type=str, required=True, help='path to the generation folder')
    args = parser.parse_args()
    return args


def create_esm_predictions(input_path: str) -> None:
    """Creates predictions for the ESM model. By calling a different file."""

    assert os.path.isdir(input_path), f'{input_path} not found'

    # list of checkpoints (FOLDERS in dir)
    checkpoints = sorted(glob.glob(f"{input_path}/*.samples"))[::-1]
    [print(f"CKPT: {x}") for x in checkpoints]

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

            
            if not os.path.isdir(pdb_path):
                # run ESMFold if not already done
                print(f"\n#### PDB output for {seed} in:\n{pdb_path}")
                COMMAND = f'python /mnt/home/mheinzinger/deepppi1tb/ESMFold/esm/scripts/fold.py ' \
                    f' -i {fasta_path} ' \
                    f'-o {pdb_path}'
                print("\n#### Running ESM prediction:\n", COMMAND)
                os.system(COMMAND)

                # parse pLDDT
                pLDDT_path = os.path.join(checkpoint, seed + "_esmfold_pLDDT.csv")
                COMMAND = f'python /mnt/home/mheinzinger/deepppi1tb/collabfold/parse_pLDDT.py {pdb_path}    {pLDDT_path}'
                print("\n#### Parsing pLDDT:\n", COMMAND)
                os.system(COMMAND)
            else:
                print(
                    f"ESM prediction for {seed} already exists. Please refer to {pdb_path} or delete this folder to generate new predictions.")
                
                
def run_foldseek(input_path: str):
    """Runs foldseek on the generated pdb files."""
    assert os.path.isdir(input_path), f'{input_path} not found'
    # get all directories ending with pdb
    pdb_dirs = sorted(glob.glob(f"{input_path}*_pdb"))
    print(pdb_dirs)


if __name__ == "__main__":
    args = parse_arguments()
    print("#### Starting ESMFold prediction ####")
    # create_esm_predictions(args.input_path)
    run_foldseek(args.input_path)
