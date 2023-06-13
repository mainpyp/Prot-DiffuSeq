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


def compare_with_validation(aln_path: str, validation_path: str):
    df_rec = pd.read_csv(aln_path, sep="\t", header=0)
    df_ref = pd.read_csv(validation_path, sep="\t", header=0)
    
    # create five subplots the first 4 are equally sized and the last one is twice as big
    # 
    # 1st subplot: histogram of pident
    # 2nd subplot: histogram of evalue
    # 3rd subplot: histogram of bits
    # 4th subplot: histogram of alntmscore
    # 5th subplot: histogram of lddt        
        
    # Create the figure and axes
    fig = plt.figure(figsize=(10, 12))
    
    # add title to figure
    seed = aln_path.split("/")[-1].split("_")[0]
    fig.suptitle(f"ckpt: {aln_path.split('/')[-2].replace('.pt.samples', '')} - {seed}", fontsize=16)

    # Define the layout
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

    # Set the titles for each subplot (optional)
    ax1.set_title('pident')
    ax2.set_title('evalue')
    ax3.set_title('bits')
    ax4.set_title('alntmscore')
    ax5.set_title('lddt')
    
    bins = 15
    color_rec = "green"
    color_ref = "orange"
    text_color = "black"
    alpha = 0.5
    
    ax1_n, _, _ = ax1.hist(df_rec["pident"], bins=bins, color=color_rec, alpha=alpha, label="rec")
    ax2_n, _, _ = ax2.hist(df_rec["evalue"], bins=bins, color=color_rec, alpha=alpha, label="rec")
    ax3_n, _, _ = ax3.hist(df_rec["bits"], bins=bins, color=color_rec, alpha=alpha, label="rec")
    ax4_n, _, _ = ax4.hist(df_rec["alntmscore"], bins=bins, color=color_rec, alpha=alpha, label="rec")
    ax5_n, _, _ = ax5.hist(df_rec["lddt"], bins=bins, color=color_rec, alpha=alpha, label="rec")
    
    ax1.hist(df_ref["pident"], bins=bins, color=color_ref, alpha=alpha, label="ref")
    ax2.hist(df_ref["evalue"], bins=bins, color=color_ref, alpha=alpha, label="ref")
    ax3.hist(df_ref["bits"], bins=bins, color=color_ref, alpha=alpha, label="ref")
    ax4.hist(df_ref["alntmscore"], bins=bins, color=color_ref, alpha=alpha, label="ref")
    ax5.hist(df_ref["lddt"], bins=bins, color=color_ref, alpha=alpha, label="ref")
    
    # adds mean and std to each subplot
    ax1.axvline(df_rec["pident"].mean(), color=color_rec, linestyle='dashed', linewidth=1)
    ax1.axvline(df_ref["pident"].mean(), color=color_ref, linestyle='dashed', linewidth=1)
    ax2.axvline(df_rec["evalue"].mean(), color=color_rec, linestyle='dashed', linewidth=1)
    ax2.axvline(df_ref["evalue"].mean(), color=color_ref, linestyle='dashed', linewidth=1)
    ax3.axvline(df_rec["bits"].mean(), color=color_rec, linestyle='dashed', linewidth=1)
    ax3.axvline(df_ref["bits"].mean(), color=color_ref, linestyle='dashed', linewidth=1)
    ax4.axvline(df_rec["alntmscore"].mean(), color=color_rec, linestyle='dashed', linewidth=1)
    ax4.axvline(df_ref["alntmscore"].mean(), color=color_ref, linestyle='dashed', linewidth=1)
    ax5.axvline(df_rec["lddt"].mean(), color=color_rec, linestyle='dashed', linewidth=1)
    ax5.axvline(df_ref["lddt"].mean(), color=color_ref, linestyle='dashed', linewidth=1)
    
    # adds mean as text to each subplot next to each axvline and in the middle of the height of the histogram
    
    ax1_half = ax1_n.max() / 2
    ax2_half = ax2_n.max() / 2
    ax3_half = ax3_n.max() / 2
    ax4_half = ax4_n.max() / 2
    ax5_half = ax5_n.max() / 2
    
    ax1.text(df_rec["pident"].mean(), ax1_half, f"mean: {df_rec['pident'].mean():.2f}", rotation=90, color=text_color)
    ax1.text(df_ref["pident"].mean(), ax1_half, f"mean: {df_ref['pident'].mean():.2f}", rotation=90, color=text_color)
    ax2.text(df_rec["evalue"].mean(), ax2_half, f"mean: {df_rec['evalue'].mean():.2f}", rotation=90, color=text_color)
    ax2.text(df_ref["evalue"].mean(), ax2_half, f"mean: {df_ref['evalue'].mean():.2f}", rotation=90, color=text_color)
    ax3.text(df_rec["bits"].mean(), ax3_half, f"mean: {df_rec['bits'].mean():.2f}", rotation=90, color=text_color)
    ax3.text(df_ref["bits"].mean(), ax3_half, f"mean: {df_ref['bits'].mean():.2f}", rotation=90, color=text_color)
    ax4.text(df_rec["alntmscore"].mean(), ax4_half, f"mean: {df_rec['alntmscore'].mean():.2f}", rotation=90, color=text_color)
    ax4.text(df_ref["alntmscore"].mean(), ax4_half, f"mean: {df_ref['alntmscore'].mean():.2f}", rotation=90, color=text_color)
    ax5.text(df_rec["lddt"].mean(), ax5_half, f"mean: {df_rec['lddt'].mean():.2f}", rotation=90, color=text_color)
    ax5.text(df_ref["lddt"].mean(), ax5_half, f"mean: {df_ref['lddt'].mean():.2f}", rotation=90, color=text_color)
    
    # add legend
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()

    # Adjust the spacing between subplots
    plt.tight_layout()
    
   # export results
    plot_output = os.path.split(aln_path)[0]  # removes last part of path
    plot_output = os.path.join(plot_output, "plots")
    if not os.path.isdir(plot_output):
        os.makedirs(plot_output)
        print(f"Created {plot_output}")
    plt.savefig(os.path.join(plot_output, f"{seed}_compare_plot.png"))
    print(f"Saved {os.path.join(plot_output, f'{seed}_compare_plot.png')}")
    plt.close()
    

if __name__ == "__main__":
    input_path = "FINAL_GENERATIONS/"
    full_input_path = os.path.join("..", "generation_outputs", input_path)
    validate_directory(full_input_path)
    
    validation_file = os.path.join("..", "generation_outputs", "test_sequences", "val_AA_512_aln_parsed.m8")
    assert os.path.isfile(validation_file), f'{validation_file} not found'

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
            parse_m8(output_aln, format_output=format_output)
            
            compare_with_validation(output_aln, validation_file)