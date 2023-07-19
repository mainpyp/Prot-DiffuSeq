import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def get_counter_df(path, val_path, generated_path):
    with open(path) as fasta:
        ids_mpnn = []
        seqs_mpnn = []
        for line in fasta: 
            if line.startswith(">"):
                ids_mpnn.append(line[1:].strip())
                seqs_mpnn.append(next(fasta).strip())
            
    with open(val_path) as fasta:    
        ids_val = []
        seqs_val = []
        for line in fasta:
            if line.startswith(">"):
                ids_val.append(line[1:].strip())
                seqs_val.append(next(fasta).strip())
                
    with open(generated_path) as fasta:
        ids_gen = []
        seqs_gen = []
        for line in fasta:
            if line.startswith(">"):
                ids_gen.append(line[1:].strip())
                seqs_gen.append(next(fasta).strip())  
    
    df_mpnn = pd.DataFrame({"ID": ids_mpnn, "sequence": seqs_mpnn})
    df_val = pd.DataFrame({"ID": ids_val, "sequence": seqs_val})
    df_gen = pd.DataFrame({"ID": ids_gen, "sequence": seqs_gen})
    
    df_mpnn["counter"] = df_mpnn["sequence"].apply(Counter)
    df_val["counter"] = df_val["sequence"].apply(Counter)
    df_gen["counter"] = df_gen["sequence"].apply(Counter)
    print(df_val)
    
    total_counter_mpnn = Counter()
    for counter in df_mpnn["counter"]:
        total_counter_mpnn += counter
    
    total_counter_val = Counter()
    for counter in df_val["counter"]:
        total_counter_val += counter
    
    total_counter_gen = Counter()
    for counter in df_gen["counter"]:
        total_counter_gen += counter
    
    counters = pd.DataFrame({"MPNN": total_counter_mpnn, "Validation": total_counter_val, "Generated": total_counter_gen})
    return counters

def compare(counters):
    
    print("PEARSON")
    # only one value please
    print(counters.corr(method="pearson"))
    print("SPEARMAN")
    print(counters.corr(method="spearman"))
    print("KENDALL")
    print(counters.corr(method="kendall"))
    
    counters.MPNN = counters.MPNN.apply(lambda x: x / sum(counters.MPNN))
    counters.Validation = counters.Validation.apply(lambda x: x / sum(counters.Validation))
    counters.Generated = counters.Generated.apply(lambda x: x / sum(counters.Generated))
    
    # sort ocunters by index
    counters.sort_index(inplace=True)
    
    # create plot with bars next to another for each index
    counters.plot.bar(rot=0)
    plt.rcParams["figure.figsize"] = (16, 8)
    plt.rcParams['font.size'] = 16
    plt.title("Amino acid distribution")
    plt.ylabel("Frequency")
    plt.show()
    

def calculate_entropy(counters):
    import numpy as np
    for col in counters.columns:
        background_aa_freq = get_uniprot_background()
        c = Counter(set(background_aa_freq.keys()))
        # col either MPNN, Validation or Generated
        c.update({aa: val for aa, val in counters[col].items()})
        rel_freqs_aa_gen = {aa : count/(counters[col].sum()+20) for aa, count in c.items()}
        entropy = sum( [ (rel_freqs_aa_gen[aa] * np.log(rel_freqs_aa_gen[aa]/b) ) 
                        for aa, b in background_aa_freq.items() 
                        ]
                        )
        print(f"{col} -> {entropy:.4f}")
    
def get_uniprot_background():
    # from Release 2023_02 of 03-May-2023 of UniProtKB/TrEMBL 
    # contains 249308459 sequence entries, comprising 86853323495 amino acids.
    # https://www.ebi.ac.uk/uniprot/TrEMBLstats
    probs = {"A" : 9.06,
            "Q" : 3.79,
            "L" : 9.87,
            "S" : 6.78,
            "R" : 5.84,
            "E" : 6.24,
            "K" : 4.92,
            "T" : 5.54,
            "N" : 3.79,
            "G" : 7.29,
            "M" : 2.33,
            "W" : 1.30,
            "D" : 5.47,
            "H" : 2.21,
            "F" : 3.89,
            "Y" : 2.88,
            "C" : 1.28,
            "I" : 5.55,
            "P" : 4.97,
            "V" : 6.88,
            }
    return { aa : prob/100 for aa, prob in probs.items() }
       
        

    
if __name__ == "__main__":
    path = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/test_sequences/ProteinMPNN/generated_sequences.fasta"
    validation_path = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/test_sequences/Validation/val_AA_1024.fasta"
    generated_path = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/BIG1_diffuseq_ProtMedium_h1024_lr0.0001_t2000_sqrt_lossaware_seed123_ProtMedium1MLsfRoFormerDebug20230610-18:32:34/ema_0.9999_080000.pt.samples/seed102_step0_rec.fasta"
    print(get_uniprot_background())
    counters = get_counter_df(path, validation_path, generated_path)
    compare(counters)
    #calculate_entropy(counters)