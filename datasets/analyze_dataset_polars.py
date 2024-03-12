import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from collections import Counter
import argparse


def timer(func):
    def inner(*args, **kwargs):
        import time

        start = time.time()
        result = func(*args, **kwargs)
        print(f"Elapsed time: {time.time() - start:.5f} Function: {func.__name__}")
        return result

    return inner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=str, required=False, default="ProtMinimalSeqToStruc/train.jsonl"
    )
    parser.add_argument(
        "--valid", type=str, required=False, default="ProtMinimalSeqToStruc/valid.jsonl"
    )
    parser.add_argument(
        "--test", type=str, required=False, default="ProtMinimalSeqToStruc/test.jsonl"
    )
    return parser.parse_args()


def setup_plot_params():
    plt.rcParams["figure.figsize"] = (25, 20)
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.labelsize"] = 350
    plt.rcParams["axes.labelweight"] = "semibold"
    plt.rcParams["xtick.labelsize"] = 25
    plt.rcParams["ytick.labelsize"] = 25
    plt.rcParams["legend.fontsize"] = 20

    sns.set_palette("colorblind", 5)
    sns.set_style("whitegrid")
    # set seaborn font sizes to something readable
    sns.set_context("paper", font_scale=2.3)


def get_subdirs_options():
    """Return the names of subdirectores that start with Prot
    and contain the files, train.jsonl, test.jsonl, valid.jsonl"""
    must_contain = ["train.jsonl", "test.jsonl", "valid.jsonl"]
    dirs = [d for d in os.listdir() if os.path.isdir(d) and d.startswith("Prot")]
    possible_dirs = [d for d in dirs if all([f in os.listdir(d) for f in must_contain])]
    return possible_dirs


def get_paths_to_files(dataset_path):
    """Return the paths to the files in the dataset directory"""
    must_contain = ["train.jsonl", "test.jsonl", "valid.jsonl"]
    paths = {
        f[:-6]: os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f in must_contain
    }
    paths["output"] = dataset_path + "/"
    return paths


@timer
def load_data(path: str, verbose: bool = False) -> pl.DataFrame:
    df = pl.read_ndjson(path)
    df = df.with_columns(pl.col("*").str.replace_all(" ", "").str.strip_chars())
    df = df.with_columns(df["src"].str.len_chars().alias("length"))

    if verbose:
        print(f"Loaded {path}\n{df.describe()}")
    return df


@timer
def get_pdb(path: str) -> pl.DataFrame:
    pdb = pl.read_csv(path)
    pdb = pdb.with_columns(pdb["sequence"].str.len_chars().alias("length"))
    pdb = pdb.filter(pdb["length"] > 30)
    return pdb


@timer
def get_statistics(df: pl.DataFrame, column: str) -> pl.DataFrame:
    statistics_as_dict = df.select(column).describe().to_dict()
    stats, values = statistics_as_dict.values()
    values = [round(v, 3) for v in values]
    return dict(zip(stats, values))


@timer
def create_length_coparisions(stats: dict, length: dict) -> str:
    pass  # TODO: implement the length comparison


def create_facedgrid_plot():
    pass  # TODO: https://seaborn.pydata.org/examples/kde_ridgeplot.html


@timer
def count_letters(df: pl.DataFrame, column: str) -> Counter:
    df = (
        df.select(pl.col(column).str.split("").flatten().alias("char"))
        .groupby("char")
        .agg(pl.count())
        .filter(pl.col("char").str.lengths() > 0)
        .sort("char")
    )

    # normalize the counts
    df = df.with_columns(pl.col("count") / pl.col("count").sum())

    return df


def main(files) -> None:
    setup_plot_params()
    pl.Config.set_tbl_rows(100)

    train, test, valid, pdb = (
        load_data(files["train"]),
        load_data(files["test"]),
        load_data(files["valid"]),
        get_pdb("../3DI_stats/seq_ss_3Di.csv"),
    )

    s_train, s_test, s_valid, s_pdb = (
        get_statistics(train, "length"),
        get_statistics(test, "length"),
        get_statistics(valid, "length"),
        get_statistics(pdb, "length"),
    )

    train_aa_counter, test_aa_counter, valid_aa_counter, pdb_aa_counter = (
        count_letters(train, "trg"),
        count_letters(test, "trg"),
        count_letters(valid, "trg"),
        count_letters(pdb, "sequence"),
    )
    # drop B and O from pdb
    pdb_aa_counter = pdb_aa_counter.filter(pdb_aa_counter["char"] != "B")
    pdb_aa_counter = pdb_aa_counter.filter(pdb_aa_counter["char"] != "O")
    pdb_aa_counter = pdb_aa_counter.filter(pdb_aa_counter["char"] != "X")
    pdb_aa_counter = pdb_aa_counter.filter(pdb_aa_counter["char"] != "U")

    aa_counter = pd.DataFrame(
        {
            "char": train_aa_counter["char"],
            "train": train_aa_counter["count"],
            "test": test_aa_counter["count"],
            "valid": valid_aa_counter["count"],
            "pdb": pdb_aa_counter["count"],
        }
    )

    train_3di_counter, test_3di_counter, valid_3di_counter, pdb_3di_counter = (
        count_letters(train, "src"),
        count_letters(test, "src"),
        count_letters(valid, "src"),
        count_letters(pdb, "3DI"),
    )
    tdi_counter = pd.DataFrame(
        {
            "char": train_3di_counter["char"],
            "train": train_3di_counter["count"],
            "test": test_3di_counter["count"],
            "valid": valid_3di_counter["count"],
            "pdb": pdb_3di_counter["count"],
        }
    )

    # Select the 'length' column and add a 'Dataset' column for each DataFrame
    train = train.select([pl.col("length"), pl.lit("train").alias("Dataset")])
    test = test.select([pl.col("length"), pl.lit("test").alias("Dataset")])
    valid = valid.select([pl.col("length"), pl.lit("valid").alias("Dataset")])
    pdb = pdb.select([pl.col("length"), pl.lit("pdb").alias("Dataset")])

    # Combine into a single DataFrame
    combined_data = pl.concat([train, test, valid, pdb])

    # Since plotting directly from Polars is not supported, for plotting, convert to Pandas DataFrame
    combined_data_pd = combined_data.to_pandas()

    axs = (
        plt.subplot2grid((3, 2), (0, 0), colspan=1),
        plt.subplot2grid((3, 2), (1, 0)),
        plt.subplot2grid((3, 2), (1, 1)),
    )

    dataset_colors = {
        "train": "darkred",
        "test": "orange",
        "valid": "green",
        "pdb": "#325880",
    }

    violin = True
    if violin:
        sns.violinplot(
            combined_data_pd,
            x="Dataset",
            y="length",
            linewidth=2,
            alpha=0.8,
            cut=0,
            # hue="length",
            split=True,
            fill=True,
            palette=dataset_colors,
            ax=axs[0],
        )
        axs[0].set_ylim(0, 850)

        axs[0].set_xticklabels(["Train", "Test", "Valid", "PDB"])
        axs[0].set_ylabel("Sequence length")
        axs[0].set_xlabel("")
    else:
        sns.kdeplot(train["length"], ax=axs[0], linewidth=2, fill=True, alpha=0.1)
        sns.kdeplot(test["length"], ax=axs[0], linewidth=2, fill=True, alpha=0.1)
        sns.kdeplot(valid["length"], ax=axs[0], linewidth=2, fill=True, alpha=0.1)
        sns.kdeplot(
            pdb["length"],
            ax=axs[0],
            linewidth=2,
            color="#325880",
            label="PDB",
            fill=True,
            alpha=0.1,
        )
        axs[0].set_xlabel("Length of source sequence")
        axs[0].set_xlim(-10, 850)
        axs[0].legend(["Train", "Test", "Valid", "PDB"])

    aa_counter.plot(
        kind="bar",
        x="char",
        y=["train", "test", "valid", "pdb"],
        ax=axs[1],
        width=0.8,
        color=list(dataset_colors.values()),
        ylabel="Percentage",
    )
    axs[1].tick_params(axis="x", rotation=0)
    axs[1].set_xlabel("Amino acid residue")
    axs[1].legend(["Train", "Test", "Valid", "PDB"], loc="upper right")

    tdi_counter.plot(
        kind="bar",
        x="char",
        y=["train", "test", "valid", "pdb"],
        ax=axs[2],
        width=0.8,
        color=list(dataset_colors.values()),
    )
    axs[2].legend(["Train", "Test", "Valid", "PDB"], loc="upper left")
    axs[2].tick_params(axis="x", rotation=0)
    axs[2].set_xlabel("3DI token")

    axs[1].legend(["Train", "Test", "Valid", "PDB"])
    axs[2].legend(["Train", "Test", "Valid", "PDB"])

    plt.tight_layout()
    # plt.show()

    output_name = f"{files['output']}analysis_plot_{violin=}.png"
    plt.savefig(output_name)
    print(f"Saving plot to {output_name}")



if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = input(
        f"Enter the path to the dataset. \n Possible choices:\n {get_subdirs_options()}: "
    )
    if dataset_path == "":
        dataset_path = "ProtMinimalSeqToStruc"
        print(f"Using the default path: {dataset_path}")

    files = get_paths_to_files(dataset_path)

    main(files)
