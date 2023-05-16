"""This script analyses the generation of a model. 
It takes as input the path to the generation folder where the generations of a 
model for each checkpoint are stored.

--input_path: path to the generation folder
"""
import argparse
import glob
import pandas as pd
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
                
                
def run_foldseek(input_path: str, format_output: list):
    """Runs foldseek on the generated pdb files."""
    assert os.path.isdir(input_path), f'{input_path} not found'
    # get all directories ending with pdb
    print(f"input_path: {input_path}")
    ckpts = sorted(glob.glob(f"{input_path}*.samples"))
    for ckpt in ckpts:
        # get all pdb dirs
        pdb_dirs = sorted(glob.glob(f"{ckpt}/*_pdb"))
        for pdb_dir in pdb_dirs:

            """
            /mnt/home/mheinzinger/deepppi1tb/foldseek/foldseek_latest/foldseek/bin/foldseek easy-search \
            /mnt/home/mheinzinger/deepppi1tb/ProSST5/analysis_prosst5/reference_structures/AFDB/val/ \
            /path/to/output/pdb \
            /path/to/foldseek_results/aln.m8 \
            /path/to/foldseek_results/tmpFolder --format-output "query,target,pident,evalue,bits,alntmscore,lddt" \
            --exhaustive-search 1
            """
            seed_ref = "_".join(pdb_dir.split("/")[-1].split("_")[:-1])
            print("seed: ", seed_ref)
            output_aln = os.path.join(ckpt, f"{seed_ref}_aln.m8")
            output_tmp = os.path.join(ckpt, f"{seed_ref}_tmpFolder")
            format_output = ",".join(format_output)
            COMMAND = f'/mnt/home/mheinzinger/deepppi1tb/foldseek/foldseek_latest/foldseek/bin/foldseek easy-search ' \
                    f'/mnt/home/mheinzinger/deepppi1tb/ProSST5/analysis_prosst5/reference_structures/AFDB/val/ ' \
                    f'{pdb_dir} ' \
                    f'{output_aln} ' \
                    f'{output_tmp} ' \
                    f'--format-output "{format_output}" ' \
                    f'--exhaustive-search 1'
            print("\n#### Running foldseek:\n", COMMAND, "\n", "-"*50)
            os.system(COMMAND)


def parse_m8_file(input_path: str, format_output: list):

    assert os.path.isdir(input_path), f'{input_path} not found'

    import pandas as pd
    # get all aln m8 files
    print(f"input_path: {input_path}")
    ckpts = sorted(glob.glob(f"{input_path}*.samples"))
    for ckpt in ckpts:
        # get all pdb dirs
        aln_m8s = sorted(glob.glob(f"{ckpt}/*aln.m8"))
        for aln_m8 in aln_m8s:
            df = pd.read_csv(aln_m8, sep="\t", header=None)
            df.columns = format_output

            # keep rows where query and target are the same
            df = df[df["query"] == df["target"]]

            parsed_file_path = aln_m8.replace("_aln", "_aln_parsed")
            if os.path.isfile(parsed_file_path):
                print(f"WARNING: File {parsed_file_path} already exists and is overwritten. :(")
                
            print(f"Write parsed m8 file to {parsed_file_path}")
            # df.to_csv(parsed_file_path, sep="\t", header=True)
    


if __name__ == "__main__":
    # args = parse_arguments()
    print("#### Starting ESMFold prediction ####")
    # create_esm_predictions(args.input_path)
    format_output = ["query", "target", "pident", "evalue", "bits", "alntmscore", "lddt"]
    # run_foldseek(args.input_path, format_output)

    parse_m8_file("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMediumCorrect_h256_lr1e-05_t6000_sqrt_lossaware_seed123_pm-correct-new-params20230419-17:39:32/ema_0.9999_100000.pt.samples/seed103_step0_ref_aln.m8", format_output)
