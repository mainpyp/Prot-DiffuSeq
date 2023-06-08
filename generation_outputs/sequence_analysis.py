# This file contains functions to analyze the generated SEQUENCES
# It does not check for structure quality

import json
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import pandas as pd
from collections import Counter
#from generation_outputs.dataset_utils import remove_brackets, dummy_aligner

# global settings for matplotlib
plt.rcParams.update({'font.size': 17})
plt.rc('legend', fontsize=12)


def from_path_to_step_K(paths: list) -> list:
    """ Extract step number from path and replace 3 trailing zeros with K
    path looks like this: ./ema_0.9999_001000.pt.samples/seed101_step0.json
    """
    labels = [path.split('/')[-2] for path in paths]
    labels= [label.split('_')[-1] for label in labels]
    labels = [label.split('.')[0] for label in labels]
    
    # replace all trailing zeros
    labels = [str(int(label)) for label in labels]
    # replace ending 000 with K
    labels = [label[:-3] + "K" if label.endswith("000") else label for label in labels]
    return labels


def extract_seqs_to_df(paths: str) -> dict:
    """ Extract sequences from jsonl file and return dataframe with columns ["recover", "reference", "source", "af_id"] ""
    Input: 
        path: path to jsonl file
    Output:
        df: dataframe with columns ["recover", "reference", "source", "af_id"]
    """
    # default dict. Keys is the paths list, values is the dataframe
    df_dict = dict()
    for path in paths:
        with open(path) as f:
            df = pd.DataFrame(columns=["recover", "reference", "source", "af_id"])
            for line in f:
                line = json.loads(line)
                rec, ref, src, af_id = line["recover"], line["reference"], line["source"], line["af_id"]
                # remove everything in square brackets
                rec = re.sub(r'\[.*?\]', '', rec).strip()
                ref = re.sub(r'\[.*?\]', '', ref).strip()
                src = re.sub(r'\[.*?\]', '', src).strip()
                # concatenate to dataframe with concat
                df = pd.concat([df, pd.DataFrame([[rec, ref, src, af_id]], columns=["recover", "reference", "source", "af_id"])])
        # add length column
        df["len_rec"] = df["recover"].apply(lambda x: len(x.split(" ")))
        df["len_ref"] = df["reference"].apply(lambda x: len(x.split(" ")))
        df["len_src"] = df["source"].apply(lambda x: len(x.split(" ")))
        df.reset_index(inplace=False)
        df_dict[path] = df
    
    return df_dict


def len_pearson_corr(df: pd.DataFrame) -> float:
    return df["len_rec"].corr(df["len_ref"], method="pearson")


def aa_dist_pearson_corr(df: pd.DataFrame) -> float:
    return df["recover"].corr(df["reference"], method="pearson")


def get_paths(seed: str) -> list:
    # get all json files in the directory and subdirectories
    paths = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".json"):
                if seed in file:
                    paths.append(os.path.join(root, file))
    return sorted(paths)
    
   
def plot_pearsons(dataframes: dict, step_labels: list, colors: list) -> None:
    
    # plot with two subplots neqxt to each other
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    length_correlations = []
    distribution_correlations = []
    for path in dataframes:
        df = dataframes[path]
        # length correlation
        length_correlations.append(len_pearson_corr(df))
        
        # distribution correlation
        glob_c_recover, glob_c_reference = get_global_counter(df)
        # join the two dicts in a dataframe. fill in missing values with 0
        df = pd.DataFrame([glob_c_recover, glob_c_reference], index=["recover", "reference"]).fillna(0).transpose()
        distribution_correlations.append(aa_dist_pearson_corr(df))
    
    # PLOT THE LENGTH CORRELATIONS    
    ax1.bar(range(len(length_correlations)), length_correlations, color=colors)
    ax1.set_xticks(range(len(length_correlations)), step_labels , rotation=45)
    ax1.axhline(y=0, color="k", linestyle='-')
    ax1.set_title("Sequence length correlation")
    
    # PLOT THE DISTRIBUTION CORRELATIONS
    ax2.bar(range(len(distribution_correlations)), distribution_correlations, color=colors)
    ax2.set_xticks(range(len(distribution_correlations)), step_labels , rotation=45)
    ax2.axhline(y=0, color="k", linestyle='-')
    ax2.set_title("AA distribution correlation")
    
    # add line that shows change between bars
    for i in range(len(length_correlations)-1):
        ax1.plot([i+0.5, i+0.5], [length_correlations[i], length_correlations[i+1]], color="k", linestyle="--")
        
    # share x and y label without tick overlap
    fig.autofmt_xdate()
    
    fig.text(0.5, 0.0, 'Number of training steps', ha='center')
    fig.text(0.01, 0.5, 'Pearson correlation coefficient', va='center', rotation='vertical')
    
    plt.title("Length and AA distribution correlation")
    plt.tight_layout()
    plt.show()
    

def get_global_counter(df: pd.DataFrame):
    glob_c_recover = Counter()
    glob_c_reference = Counter()
    for index, row in df.iterrows():
        c_recover = Counter(row.recover)
        c_reference = Counter(row.reference)
        del c_recover[" "], c_reference[" "]
        glob_c_recover += c_recover
        glob_c_reference += c_reference

    return glob_c_recover, glob_c_reference

# calculate the pearson correlation based of two counter dicts
def dict_correlation(c1: Counter, c2: Counter) -> float:
    keys = list(c1.keys() | c2.keys())
    import numpy as np
    corrs = np.corrcoef([c1.get(k, 0) for k in keys], [c2.get(k, 0) for k in keys])
    return corrs


def compare_aa_distribution(dataframes: dict, step_labels: list, colors: list) -> None:
    """ Compare the amino acid distribution between the reference and the recovered sequence
    """
    
    # plot with as many subfigures as there are dataframes
    fig, axs = plt.subplots(len(dataframes) // 2, 2, figsize=(10, 8))
    
    for i, (ax, title, path) in enumerate(zip(axs.reshape(-1), step_labels, dataframes)):
        df = dataframes[path]
        glob_c_recover, glob_c_reference = get_global_counter(df)
              
        glob_c_recover = sorted(glob_c_recover.items())
        glob_c_reference = sorted(glob_c_reference.items())
        
        # keep order of amino acids
        x_rec = [x[0] for x in glob_c_recover]
        y_rec = [x[1] for x in glob_c_recover]
        
        x_ref = [x[0] for x in glob_c_reference]
        y_ref = [x[1] for x in glob_c_reference]
        
        # plot as barplot the two distributions
        ax.bar(x_rec, y_rec, color="blue", alpha=0.5, label="recover")
        ax.bar(x_ref, y_ref, color="red", alpha=0.5, label="reference")
        ax.set_title(title)
    
    # set shared x and y labels
    fig.text(0.5, 0.04, 'Amino acid', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    # set global legend
    fig.legend(["recover", "reference"], loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generation_path = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/FINAL_GENERATIONS"
    # change working directory to this directory
    os.chdir(generation_path)
    
    paths = get_paths("seed101")
    print(paths)
    step_labels = from_path_to_step_K(paths[::2])
    
    dataframes = extract_seqs_to_df(paths[::2])
    print(dataframes.keys())
    
    # list of 50 different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(dataframes)))
    plot_pearsons(dataframes, step_labels, colors)
    
    compare_aa_distribution(dataframes, step_labels, colors)

    
    