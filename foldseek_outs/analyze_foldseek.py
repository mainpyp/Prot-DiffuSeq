import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/foldseek_outs")
plt.rcParams.update({'font.size': 22})


def analyze_foldseek(file_list: list) -> None:
    
    # create 4 subplots in one column
    fig, axs = plt.subplots(4, figsize=(10, 12))
    for ax, c, file in zip(axs[:-1], ["g", "b", "orange"],file_list):
        header = "query,target,pident,evalue,bits,alntmscore,lddt".split(",")
        df = pd.read_csv(file, sep="\t", header=None)
        df.columns = header
        df = df.drop_duplicates(subset=["query", "target"])
        df = df.sort_values(by=["query", "lddt"], ascending=False)

        ax.hist(df.lddt, bins=250, color=c, alpha=1)
        mean = df.lddt.mean()
        # adds mean as vertical line
        ax.axvline(mean, color="k", linestyle="dashed", linewidth=1)
        # adds value of mean as text
        ax.text(mean + 0.1, 400, f"mean: {mean:.2f}")
        axs[-1].hist(df.lddt, bins=250, color=c, alpha=0.5)
        ax.set_title(f"lddt distribution of {file.split('/')[0]}")
    plt.legend(["ref", "130krec", "100krec", "all"])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filename = "aln.m8"
    files = ["ref", "130krec", "100krec"]

    files = list(map(lambda x: f"{x}/{filename}", files))
    analyze_foldseek(files)