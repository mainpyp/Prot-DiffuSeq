import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

LENGTHS = [128, 256, 512]
MIN_LENGTH = np.inf
PATH = "val_AA.fasta"

# read the file from PATH
# iterate so that two subsequent lines are read in at once
# the first line is the header, the second line is the sequence
with open(PATH, "r") as f:
    val_aa = dict()
    for line in f:
        if line.startswith(">"):
            head = line.strip()
            val_aa[head] = ""   
        else:
            val_aa[head] = line.strip()
            if len(line.strip()) < MIN_LENGTH:
                MIN_LENGTH = len(line.strip())


total = ""
for key, value in val_aa.items():
    total += value

unique_aa = set(total)


for length in LENGTHS:
    outpath = f"random_{length*2}_AA.jsonl"
    
    # if the file exists, delete it
    if os.path.exists(outpath):
        os.remove(outpath)
    
    for head, seq in val_aa.items():
        
        # generate a random sequence of length length
        # -2 because of (length - 3 //2) when tokenizing
        random_seq = np.random.choice(list(unique_aa), size=min(len(seq), length-3))
        random_seq = "\n" + "".join(random_seq) + "\n"
        
        # write to file
        with open(outpath, "a") as f:
            f.writelines([head, random_seq])
    
