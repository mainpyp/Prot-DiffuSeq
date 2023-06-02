import json
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import pandas as pd
from generation_outputs.dataset_utils import remove_brackets, dummy_aligner

# change working directory to this directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# global settings for matplotlib
plt.rcParams.update({'font.size': 20})
plt.rc('legend', fontsize=10)


def from_path_to_step_K(paths: list) -> list:
    """ Extract step number from path and replace 3 trailing zeros with K
    path looks like this: ./ema_0.9999_001000.pt.samples/seed101_step0.json
    """
    labels = [path.split('/')[1] for path in paths]
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


def get_paths(seed: str) -> list:
    # get all json files in the directory and subdirectories
    paths = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".json"):
                if seed in file:
                    paths.append(os.path.join(root, file))
    return sorted(paths)
    
   
def plot_pearson_len(dataframes: dict, step_labels: list, colors: list) -> None:
    
    person_correlations = []
    for path in dataframes:
        df = dataframes[path]
        person_correlations.append(len_pearson_corr(df))
        
    plt.bar(range(len(person_correlations)), person_correlations, color=colors)
    
    plt.xticks(range(len(person_correlations)), step_labels , rotation=45)
    # add horizontal line at 0
    plt.axhline(y=0, color="k", linestyle='-')
    
    # add line that shows change between bars
    for i in range(len(person_correlations)-1):
        plt.plot([i+0.5, i+0.5], [person_correlations[i], person_correlations[i+1]], color="k", linestyle="--")
        
    plt.xlabel("Number of training steps")
    plt.ylabel("Pearson correlation coefficient")
    plt.title("Pearson correlation coefficient between number of words in reference and recovered sequence")
    plt.tight_layout()
    plt.show()
    

def compare_aa_distribution(dataframes: dict, step_labels: list, colors: list) -> None:
    """ Compare the amino acid distribution between the reference and the recovered sequence
    """
    for path in dataframes:
        df = dataframes[path]
        
        
        

if __name__ == "__main__":
    
    paths = get_paths("seed101")
    step_labels = from_path_to_step_K(paths)
    
    dataframes = extract_seqs_to_df(paths)
    
    # list of 50 different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(dataframes)))
    #plot_pearson_len(dataframes, step_labels, colors)
    compare_aa_distribution(dataframes, step_labels, colors)

    
    