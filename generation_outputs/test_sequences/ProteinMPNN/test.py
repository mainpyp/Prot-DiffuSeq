import pandas as pd
import sys, os

# set working directory to vurrent dir
abspath = os.path.abspath(sys.argv[0])
format_output = ["query", "target", "pident", "evalue", "bits", "alntmscore", "lddt"]

df = pd.read_csv("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/test_sequences/semi_random_generation/semi_random_aln.m8", sep="\t", header=None)
print(df)
df.columns = format_output
# parse where target == query
df = df[df["query"] == df["target"]]
df.to_csv("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/test_sequences/semi_random_generation/semi_random_aln_parsed.m8", sep="\t")
print(df)
