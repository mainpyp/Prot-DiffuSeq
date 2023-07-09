import json
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter

LENGTHS = [128, 256, 512]
MIN_LENGTH = np.inf
PATH = "val_AA.fasta"

# read the test fasta file from PATH
with open(PATH, "r") as f:
    
    total_counter = Counter()
    val_aa = dict()
    for line in f:
        if line.startswith(">"):
            head = line.strip()
            val_aa[head] = ""   
        else:
            val_aa[head] = line.strip()
            total_counter.update(Counter(line.strip()))
            if len(line.strip()) < MIN_LENGTH:
                MIN_LENGTH = len(line.strip())

# create a distribution dictionary out of total counter
aa_distribution = {key: value/sum(total_counter.values()) for key, value in total_counter.items()}
sample_aa = lambda : np.random.choice(list(aa_distribution.keys()), p=list(aa_distribution.values()))

for key, val in val_aa.items():
    length = len(val)
    random_seq = "".join([sample_aa() for _ in range(length)])
    outpath = f"semi_random_AA.jsonl"
    with open(outpath, "a") as f:
        f.writelines([key, "\n", random_seq, "\n"])


exit()
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
    
