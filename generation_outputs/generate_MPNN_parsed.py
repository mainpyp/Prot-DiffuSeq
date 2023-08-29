import pandas as pd
import matplotlib.pyplot as plt
format_output = ["query", "target", "pident", "evalue", "bits", "alntmscore", "lddt"]
df = pd.read_csv("generation_outputs/val_EFvsAFDB_aln_PMPNN.m8", sep="\t", header=None)
df.columns = format_output

# only rows where query == target
df = df[df["query"] == df["target"]]

# export df without header
df.to_csv("generation_outputs/val_EFvsAFDB_aln_PMPNN_parsed.csv", sep="\t")