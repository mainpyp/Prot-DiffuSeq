""" This script is used to evaluate the performance of the model on the validation set.
    Input: 
        - model path
        - dataset path
        - seed (optional) 
    Workflow:
    1. Sample from the model
    2. Convert the output jsonl file to a fasta file
    3. Run ESM fold on the fasta files
    4. Extract pLDDT scores
    5. Run foldseek on the generated structures exhaustive search (all vs all)
    6. Filter for same structure (same AF ID) 
    7. Compare lddt scores and plot distributions

conda activate mheinzinger_esmfold_CLI

python /mnt/home/mheinzinger/deepppi1tb/ESMFold/esm/scripts/fold.py -i /path/to/generated.fasta -o /path/to/output/pdb

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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate the model on the validation set.")
    parser.add_argument("--model_dir", type=str, help="Path to the model directory.")
    parser.add_argument("--seed", type=int, default=123, help="Seed for the random number generator.")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on.")
    return parser.parse_args()


def generate_sequences(model_dir: str, seed: int = 123, split: str = "test") -> str:
    import subprocess

    command = f"python -u run_decode.py --model_dir {model_dir} --seed {seed} --split {split}"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)

if __name__ == "__main__":
    args = parse_arguments()
    generate_sequences(args.model_dir, args.seed, args.split)