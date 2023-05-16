"""This script analyses the generation of a model. 
It takes as input the path to the generation folder where the generations of a 
model for each checkpoint are stored.

--input_path: path to the generation folder
"""
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
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
    print("#### Starting ESMFold prediction ####")

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
                
                
def run_foldseek(input_path: str, format_output: list, eval_threshold: float = 100.):
    """Runs foldseek on the generated pdb files."""
    print("#### Starting foldseek ####")
    assert os.path.isdir(input_path), f'{input_path} not found'
    # get all directories ending with pdb
    print(f"input_path: {input_path}")
    ckpts = sorted(glob.glob(f"{input_path}*.samples"))
    format_output = ",".join(format_output)
    print(f"format_output: {format_output}")
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
            COMMAND = f'/mnt/home/mheinzinger/deepppi1tb/foldseek/foldseek_latest/foldseek/bin/foldseek easy-search ' \
                    f'/mnt/home/mheinzinger/deepppi1tb/ProSST5/analysis_prosst5/reference_structures/AFDB/val/ ' \
                    f'{pdb_dir} ' \
                    f'{output_aln} ' \
                    f'{output_tmp} ' \
                    f'--format-output "{format_output}" ' \
                    f'--exhaustive-search 1 ' \
                    f'-e {eval_threshold}'
            print("\n#### Running foldseek:\n", COMMAND, "\n", "-"*50)
            os.system(COMMAND)


def parse_m8_file(input_path: str, format_output: list, dry_run: bool = False):
    print("#### Starting parsing the m8 files ####")
    assert os.path.isdir(input_path), f'{input_path} not found'

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
            
            

            if not dry_run:
                print(f"Write parsed m8 file to {parsed_file_path}")
                print(f"DF: {df.shape}")
                df.to_csv(parsed_file_path, sep="\t", header=True)
            else:
                print("DRY RUN")
                print(f"DF: {df.head()}")
                print(f"DF: {df.shape}")
    

def compare(input_path: str, save: bool = False):
    """This function gets a path for a training folder and compares for 
    each checkpoint the results of foldseek and ESMFold.
    Folder structure:
    MODEL
    |__ checkpoint1
    |   |__ seed1.json
    |   |__ seed1_ref|ref.fasta
    |   |__ seed1_rec|ref_aln.m8
    |   |__ seed1_rec|ref_aln_parsed.m8
    |   |__ seed1_rec|ref_pdb_folder
    |   |__ seed1_rec|ref_tmpFolder

    This function compares the content of seed1_rec|ref_aln_parsed.m8 which are two files.
    """
    assert os.path.isdir(input_path), f'{input_path} not found'

    # get all aln m8 files
    print(f"input_path: {input_path}")
    ckpts = sorted(glob.glob(f"{input_path}/*.samples"))
    for ckpt in ckpts:
        # get all pdb dirs
        aln_rec_m8s = sorted(glob.glob(f"{ckpt}/*rec_aln_parsed.m8"))
        
        for aln_rec in aln_rec_m8s:
            
            df_rec = pd.read_csv(aln_rec, sep="\t", header=0)
            df_ref = pd.read_csv(aln_rec.replace("_rec_", "_ref_"), sep="\t", header=0)
            print(df_rec.head())
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
            seed = aln_rec.split("/")[-1].split("_")[0]
            fig.suptitle(f"ckpt: {ckpt.split('/')[-1].replace('.pt.samples', '')} - {seed}", fontsize=16)

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
            
            # Show the plot
            if save:
                # export results
                plot_output = os.path.join(ckpt, f"{seed}_plots")
                if not os.path.isdir(plot_output):
                    os.makedirs(plot_output)
                    print(f"Created {plot_output}")
                plt.savefig(os.path.join(plot_output, f"compare_plot.png"))
                print(f"Saved {os.path.join(plot_output, f'compare_plot.png')}")
            else:
                plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    
    create_esm_predictions(args.input_path)
    
    format_output = ["query", "target", "pident", "evalue", "bits", "alntmscore", "lddt"]
    run_foldseek(args.input_path, format_output, eval_threshold=100.)
    parse_m8_file(args.input_path, format_output)
    compare(args.input_path, save=True)
