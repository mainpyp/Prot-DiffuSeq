import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

def validate_directory(path: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path {path} does not exist.")
    
    validation_fasta_path = os.path.join(path, "val_AA.fasta")
    if not os.path.exists(validation_fasta_path):
        raise FileNotFoundError(f"Path {validation_fasta_path} does not exist.")
    
    return path
    

def get_checkpoints(path_to_checkpoints: str):
    """Gets all folders with the generated outputs."""
    return sorted(glob.glob(f"{path_to_checkpoints}/*.samples"))[::-1]


def get_fastas(path_to_ckpt: str):
    """Gets all fasta files from a checkpoint."""
    return sorted(glob.glob(f"{path_to_ckpt}/*.fasta"))


def get_seeds(checkpoint: str):
    """Gets all seeds from a checkpoint."""
    return checkpoint.split("/")[-1].split(".")[0]


def create_esm_prediction(path_to_fasta: str, pdb_path: str = None, pLDDT_path: str = None):
    
    assert os.path.isfile(path_to_fasta), f'{path_to_fasta} not found'

    if not os.path.isdir(pdb_path):
        # run ESMFold if not already done
        print(f"\n#### PDB output for {path_to_fasta} in:\n{pdb_path}")
        COMMAND = f'python /mnt/home/mheinzinger/deepppi1tb/ESMFold/esm/scripts/fold.py ' \
            f' -i {path_to_fasta} ' \
            f'-o {pdb_path}'
        print("\n#### Running ESM prediction:\n", COMMAND)
        os.system(COMMAND)

        # parse pLDDT
        COMMAND = f'python /mnt/home/mheinzinger/deepppi1tb/collabfold/parse_pLDDT.py {pdb_path} {pLDDT_path}'
        print("\n#### Parsing pLDDT:\n", COMMAND)
        os.system(COMMAND)
    else:
        print(
            f"ESM prediction for {path_to_fasta} already exists. Please refer to {pdb_path} or delete this folder to generate new predictions.")



def foldseek(pdb_dir: str, out_aln:str, format_output: list):
    # check if out_aln already exists
    if os.path.exists(out_aln):
        print(f"Foldseek output {out_aln} already exists. Skipping foldseek.")
        return
    
    format_output = ','.join(format_output)
    
    output_tmp = pdb_dir + "_tmp"
    COMMAND = f'/mnt/home/mheinzinger/deepppi1tb/foldseek/foldseek_latest/foldseek/bin/foldseek easy-search ' \
            f'/mnt/home/mheinzinger/deepppi1tb/ProSST5/analysis_prosst5/reference_structures/AFDB/val/ ' \
            f'{pdb_dir} ' \
            f'{out_aln} ' \
            f'{output_tmp} ' \
            f'--format-output "{format_output}" ' \
            f'--exhaustive-search 1 ' \
            f'-e {100.}'
    print("\n#### Running foldseek:\n", COMMAND, "\n", "-"*50)
    os.system(COMMAND)
    
    
def generate_truncated_reference_fastas(path_to_reference: str, original_length: int):
    """This function was used to generate the truncated reference fastas.
    It was called like this: 
    for i in [256, 512, 1024]:
        generate_truncated_reference_fastas(reference_fasta, original_length=i)
    """
    
    truncation_length = int((original_length / 2) - 3)
    print(truncation_length)
    lines = []
    with open(path_to_reference, "r") as f:
        for line in f:
            lines.append(line[:truncation_length].strip())
    
    with open(path_to_reference.replace(".fasta", f"_trunc{original_length}.fasta"), "w") as f:
        f.write("\n".join(lines))



def parse_m8(m8_file: str, format_output: list, dry_run: bool=False):
    """Parses the m8 file from foldseek."""
    df = pd.read_csv(m8_file, sep="\t", header=None)
    df.columns = format_output

    # keep rows where query and target are the same
    df = df[df["query"] == df["target"]]

    parsed_file_path = m8_file.replace("_aln", "_aln_parsed")
    if os.path.isfile(parsed_file_path):
        print(f"WARNING: File {parsed_file_path} already exists and is overwritten. :(")
    
    if not dry_run:
        print(f"Write parsed m8 file to {parsed_file_path}")
        print(f"DF: {df.shape}")
        df.to_csv(parsed_file_path, sep="\t", header=True)
    else:
        print("DRY RUN")
        print(f"DF: {df.head()}")
        print(f"DF: {df.shape}")
        
        
def generate_references():
    """This was only done once"""
    for i in [256, 512, 1024]:
            reference_fasta = os.path.join("..", "generation_outputs", "test_sequences", f"val_AA_{i}.fasta")
            assert os.path.isfile(reference_fasta), f'{reference_fasta} not found'
            reference_pdb_path = reference_fasta.replace(".fasta", "_pdb")
            reference_pLDDT_path = reference_fasta.replace(".fasta", "_esmfold_pLDDT.csv")
            create_esm_prediction(path_to_fasta=reference_fasta, pdb_path=reference_pdb_path, pLDDT_path=reference_pLDDT_path)
            
            output_aln = reference_fasta.replace(".fasta", "_aln.m8")
            format_output = ["query", "target", "pident", "evalue", "bits", "alntmscore", "lddt"]
            foldseek(pdb_dir=reference_pdb_path, out_aln=output_aln, format_output=format_output)
            
            parse_m8(output_aln, format_output=format_output)
        
    print("DONE CREATING REFERENCE PREDICTIONS")


if __name__ == "__main__":
    input_path = "FINAL_GENERATIONS/"
    full_input_path = os.path.join("..", "generation_outputs", input_path)
    validate_directory(full_input_path)
    
    generate_references()
    
    exit()

    checkpoints = get_checkpoints(full_input_path)
    for ckpt in checkpoints:
        for fasta in get_fastas(ckpt):
            # store pdb and pLDDT in the same folder as the fasta
            pdb_path = fasta.replace(".fasta", "_pdb")
            pLDDT_path = fasta.replace(".fasta", "_esmfold_pLDDT.csv")
            create_esm_prediction(path_to_fasta=fasta, pdb_path=pdb_path, pLDDT_path=pLDDT_path)
            
            output_aln = fasta.replace(".fasta", "_aln.m8")
            output_tmp = fasta.replace(".fasta", "_tmpFolder")
            format_output = ["query", "target", "pident", "evalue", "bits", "alntmscore", "lddt"]
            foldseek(pdb_dir=pdb_path, out_aln=output_aln, format_output=format_output)