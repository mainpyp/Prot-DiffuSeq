"""This script analyses the generation of a model. 
It takes as input the path to the generation folder where the generations of a 
model for each checkpoint are stored.

--input_path: path to the generation folder




# extract pLDDT
python /mnt/home/mheinzinger/deepppi1tb/collabfold/parse_pLDDT.py /path/to/output/pdb    esmfold_pLDDT.csv

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

        get_seed = lambda x: x.split("/")[-1].split(".")[0]
        seeds = [get_seed(x) for x in glob.glob(f"{checkpoint}/*.fasta")]
        print(seeds)
        for seed in seeds:

            # create output pdb folder
            pdb_path = os.path.join(checkpoint, seed, "pdb_output")

            print("PDB output in: ", pdb_path)
            COMMAND = f'python /mnt/home/mheinzinger/deepppi1tb/ESMFold/esm/scripts/fold.py ' \
                    f' -i {seed}.fasta ' \
                    f'-o {pdb_path}'
            print("Running ESM prediction: ", COMMAND)
            
            # os.system(COMMAND)


if __name__ == "__main__":
    args = parse_arguments()
    create_esm_predictions(args.input_path)