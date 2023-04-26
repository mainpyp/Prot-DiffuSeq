import json
import re
import os
import blosum as bl
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# helper class of amino acids
from enum import Enum


class AminoAcids(Enum):
    polar = [*"STYNQ"]
    non_polar = [*"GAVCPLIMWF"]
    pos_charged = [*"KRH"]
    neg_charged = [*"DE"]
    aromatic = [*"FYW"]
    special = [*"X"]


# the subytitution matrix for BLOSUM62
# https://en.wikipedia.org/wiki/BLOSUM
# https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt
BLOSUM62 = bl.BLOSUM(62)


def load_jsonl(path: str) -> tuple:
    """ Loads a jsonl file into a list of dicts. 
    The json file is the output of one generation script.
    """
    recorverd = []
    reference = []
    source = []
    with open(path, 'r') as f:
        for line in f:
            rec, ref, src = json.loads(line).items()

            # we only want the sequence (key is at first place)
            recorverd.append(rec[1])
            reference.append(ref[1])
            source.append(src[1])

        result = pd.DataFrame(
            {"recovered": recorverd, "reference": reference, "source": source})

        return result


def dummy_aligner(seq1, seq2):
    """ Aligns two sequences by adding spaces where no match and | if exact match.
    """
    aligned_str = "".join(
        ["|" if s1 == s2 else " " for s1, s2 in zip(seq1, seq2)])
    blosum_score = sum([BLOSUM62[s1][s2] for s1, s2 in zip(seq1, seq2)])
    return aligned_str, aligned_str.count("|"), blosum_score


def basic_analyze(rec_ref_src: tuple):
    # removes everything in brackets and spaces
    for _, (recover, reference, source) in rec_ref_src.iterrows():
        # cleans the sequences from brackets and spaces
        clean_rec, clean_ref, clean_src = map(
            remove_brackets, (recover, reference, source))

        print(f"generated: {clean_rec}")
        dum_ali = dummy_aligner(clean_rec, clean_ref)
        print(
            f"dummy ali: {dum_ali[0]} score:  {dum_ali[1]} | {dum_ali[1] / len(dum_ali[0]):.3f} blosum: {dum_ali[2]}")
        print(f"reference: {clean_ref}")
        print(f"source   : {clean_src}\n---")

        print(
            f"generated sequence == reference sequence: {clean_rec == clean_ref}")
        print("generated has the same size as reference:",
              len(clean_rec) == len(clean_ref))
        print("generated starts with methionin: ", clean_rec.startswith("M"))
        print("generated is only AminoAcids: ", clean_rec.isupper())
        print("Amount of Different AminoAcids in generated: ", len(set(clean_rec)))
        print("Amount of Different AminoAcids in reference: ", len(set(clean_ref)))
        print("\n")

        # print(f"{len(generated) {len(reference)} {len(source)-1}")


def plot_aa_dist_pairwise(rec_ref_src: pd.DataFrame):
    # fig ax with 5 subplots
    fig, ax = plt.subplots(rec_ref_src.shape[0], figsize=(
        10, rec_ref_src.shape[0] * 1.25), sharex=True)
    fig.suptitle(
        "Token distribution of generated and reference sequences", fontsize=16)

    glob_c_recover = Counter()
    glob_c_reference = Counter()

    for row, (index, (recover, reference, source)) in zip(ax, rec_ref_src.iterrows()):
        c_recover = Counter(remove_brackets(recover))
        c_reference = Counter(remove_brackets(reference))
        del c_recover[" "]
        del c_reference[" "]
        glob_c_recover += c_recover
        glob_c_reference += c_reference

        # plot counter dict as a bar plots with the bars next to another
        row.bar(c_recover.keys(), c_recover.values(), color='orange')
        row.bar(c_reference.keys(), c_reference.values(),
                color='black', alpha=0.5)
    fig.legend(["generated", "reference"])

    plt.tight_layout()
    plt.show()

    # plot the global counter dict
    sp_correlation = spearman_correlation(glob_c_recover, glob_c_reference)
    p_correlation = pearson_correlation(glob_c_recover, glob_c_reference)
    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.suptitle(
        f"Global token distribution sp: {sp_correlation:.3f} p: {p_correlation:.3f}", fontsize=16)
    ax.bar(glob_c_recover.keys(), glob_c_recover.values(), color='orange')
    ax.bar(glob_c_reference.keys(), glob_c_reference.values(),
           color='black', alpha=0.5)
    fig.legend(["generated", "reference"])
    plt.tight_layout()
    plt.show()


def plot_aa_dist_global(rec_ref_src: pd.DataFrame, step: int, save_path: str = None):

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    glob_c_recover = Counter()
    glob_c_reference = Counter()

    for index, (recover, reference, source) in rec_ref_src.iterrows():
        c_recover = Counter(remove_brackets(recover))
        c_reference = Counter(remove_brackets(reference))
        del c_recover[" "], c_reference[" "]
        glob_c_recover += c_recover
        glob_c_reference += c_reference

    # plot the global counter dict
    sp_correlation = spearman_correlation(glob_c_recover, glob_c_reference)
    p_correlation = pearson_correlation(glob_c_recover, glob_c_reference)
    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.suptitle(
        f"Global token distribution sp: {sp_correlation:.3f} p: {p_correlation:.3f}\nstep {step}", fontsize=16)
    ax.bar(glob_c_recover.keys(), glob_c_recover.values(), color='orange')
    ax.bar(glob_c_reference.keys(), glob_c_reference.values(),
           color='black', alpha=0.5)
    fig.legend(["generated", "reference"])
    plt.tight_layout()

    if not save_path: plt.show() 
    else: plt.savefig(os.path.join(save_path, f"step_{step}.png"))

    return sp_correlation, p_correlation

# calculate the spearman correlation based of two counter dicts
def spearman_correlation(c1: Counter, c2: Counter) -> float:
    from scipy.stats import spearmanr
    # get the intersection of the keys
    keys = set(c1.keys()).intersection(set(c2.keys()))
    # get the values of the intersection
    v1 = [c1[k] for k in keys]
    v2 = [c2[k] for k in keys]
    # calculate the spearman correlation
    return spearmanr(v1, v2).correlation

# calculate the pearson correlation based of two counter dicts
def pearson_correlation(c1: Counter, c2: Counter) -> float:
    from scipy.stats import pearsonr
    # get the intersection of the keys
    keys = set(c1.keys()).intersection(set(c2.keys()))
    # get the values of the intersection
    v1 = [c1[k] for k in keys]
    v2 = [c2[k] for k in keys]
    # calculate the spearman correlation
    return pearsonr(v1, v2)[0]

# get all json files in a directory and subdirectories


def get_json_files(path: str):
    import os
    json_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return sorted(json_files)

# regex that gets the number between the 999 and .pt


def get_step_from_path(path: str) -> int:
    import re
    return int(re.search(r'999_(.*?)\.pt', path).group(1))


if __name__ == '__main__':
    global remove_brackets
    def remove_brackets(x): return re.sub(r'\[(.*?)\]', '', x).replace(" ", "")

    path = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMediumCorrect_h256_lr1e-05_t6000_sqrt_lossaware_seed123_pm-correct-new-params20230419-17:39:32/ema_0.9999_040000.pt.samples/seed123_step0.json"
    l = load_jsonl(path)
    model_path = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMediumCorrect_h256_lr1e-05_t6000_sqrt_lossaware_seed123_pm-correct-new-params20230419-17:39:32/"

    files = get_json_files(model_path)
    for path in files:
        step = get_step_from_path(path)
        l = load_jsonl(path)
        plot_aa_dist_global(l, step, save_path=model_path + "plots/")

    # basic_analyze(l)
    # plot_aa_dist_global(l)
