import json
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import pandas as pd
from collections import Counter
#from generation_outputs.dataset_utils import remove_brackets, dummy_aligner

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
    fig, axs = plt.subplots(len(dataframes) // 4, 2, figsize=(10, 15))
    
    for i, (ax, title, path) in enumerate(zip(axs.reshape(-1), step_labels, dataframes)):
        if i % 2 == 0:
            continue
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
        
        
        
        
        # corrs = pd.DataFrame(columns=["step", "spearman", "pearson"])
        
        # 
        # pcorr = pearson_correlation(glob_c_recover, glob_c_reference)

        
def plot_correlations(models_path):
    ...
    
        
        
if __name__ == "__main__":
    paths = get_paths("seed101")
    step_labels = from_path_to_step_K(paths)
    
    dataframes = extract_seqs_to_df(paths)
    
    # list of 50 different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(dataframes)))
    # plot_pearson_len(dataframes, step_labels, colors)
    
    compare_aa_distribution(dataframes, step_labels, colors)

    
    