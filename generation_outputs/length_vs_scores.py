import pandas as pd

def compare(path):
    lddt = "seed102_step0_rec_aln_parsed.m8"
    rmsd = "seed102_step0_rec_val_EFvsAFDB_RMSDs.log"
    fasta = "seed102_step0_rec.fasta"
    
    lddt_df = pd.read_csv(path + lddt, sep="\t")
    rmsd_df = pd.read_csv(path + rmsd, sep=",")
    lddt_df.drop(["Unnamed: 0", "query"], axis=1, inplace=True)
    # rename target to ID
    lddt_df.rename(columns={"target": "ID"}, inplace=True)
    lddt_df["ID"] = lddt_df["ID"].apply(lambda x: x.replace(".pdb", ""))
    full = pd.merge(lddt_df, rmsd_df, on="ID", how="right")
    
    ids = []
    seqs = []
    with open(path + fasta) as fasta:
        for line in fasta: 
            if line.startswith(">"):
                ids.append(line[1:].strip())
                seqs.append(next(fasta).strip())
    fasta_df = pd.DataFrame({"ID": ids, "sequence": seqs})
    full = pd.merge(full, fasta_df, on="ID", how="right")
    
    full["sequence_length"] = full["sequence"].apply(len)
    full["LDDT > 0.7"] = full["lddt"].apply(lambda x: "yes" if x > 0.7 else "no")
    
 
    # plot sequence length vs lddt
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    #sns.set_theme(style="whitegrid", font_scale=1.5)
    
    plt.rcParams["figure.figsize"] = (16, 8)
    # increase fontsize
    plt.rcParams['font.size'] = 20
    
    # create subplots with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.scatterplot(data=full, x="SeqID", y="lddt", hue="TM-score", palette="coolwarm", alpha=0.8, ax=ax1, legend="brief")
    sns.regplot(data=full, x="SeqID", y="lddt", scatter=False, color="orange", ax=ax1)
    ax1.set_xlabel("SeqID")
    ax1.set_ylabel("LDDT")
    ax1.set_title("Sequence length vs TM-Score")
    ax1.legend(title="TM Score", loc="upper right", bbox_to_anchor=(0.95, 0.98), ncol=2, fancybox=True, fontsize=14)
    
    sns.scatterplot(data=full, x="lddt", y="TM-score", alpha=0.5, hue="sequence_length", palette="coolwarm", color="#0169B2", ax=ax2)    
    sns.regplot(data=full, x="lddt", y="TM-score", scatter=False, color="orange", ax=ax2)
    ax2.legend(title="Length", loc="upper right", bbox_to_anchor=(0.4, 0.98), ncol=2, fancybox=True, fontsize=14)
    ax2.set_xlabel("LDDT")
    ax2.set_ylabel("TM-score")
    ax2.set_title("LDDT vs TM-score")

    # add id where the TM-score > 0.8 and the text should not overlap
    for i, row in full.iterrows():
        if row["sequence_length"] > 100 and row["lddt"] > 0.7 and row["TM-score"] > 0.5 or row["ID"] == "AF-A0A5A9PN70-F1":
            if row["ID"] == "AF-A0A5A9PN70-F1":
                ax2.text(row["lddt"], row["TM-score"], row["ID"] + " g.o.a.t.", fontsize=10)
            else:
                ax2.text(row["lddt"], row["TM-score"], row["ID"], fontsize=8)
    full["Length Category"] = full["sequence_length"].apply(lambda x: "< 94" if x < 94 else "> 94")
    
    # calculate pearson correlation between lddt and SeqID
    print("Correlation between SeqID and lddt")
    print(full[["SeqID", "lddt"]].corr(method="pearson"))
    print(full[["SeqID", "lddt"]].corr(method="spearman"))
    print(full[["SeqID", "lddt"]].corr(method="kendall"))
    
    
    print("Correlation between sequence length and lddt")
    print(full[["sequence_length", "lddt"]].corr(method="pearson"))
    print(full[["sequence_length", "lddt"]].corr(method="spearman"))
    print(full[["sequence_length", "lddt"]].corr(method="kendall"))
    print("Correlation between sequence length and SeqID")
    print(full[["sequence_length", "SeqID"]].corr(method="pearson"))
    print(full[["sequence_length", "SeqID"]].corr(method="spearman"))
    print(full[["sequence_length", "SeqID"]].corr(method="kendall"))
    print(full.sequence_length.median())
    # Define the color palette for the categories
    category_colors = {"< 94": "#0169B2", "> 94": "orange"}
    sns.jointplot(data=full, x="SeqID", y="lddt", hue="Length Category", 
                xlim=(full.SeqID.min(), full.SeqID.max()),
                alpha=0.95,legend="brief", 
                # set hue color palette
                hue_order=["< 94", "> 94"],
                # set color of points
                palette=category_colors)
    plt.ylabel("LDDT")
    plt.xlabel("Sequence Identity")
    # set color of points
    #plt.scatter(full.SeqID, full.lddt, c=full["Length Category"].apply(lambda x: "red" if x == "< 94" else "blue"))
    
    # change the legend of ax1
    plt.tight_layout()
    
    
    
    plt.show()
   
                
    


if __name__ == "__main__":
    path_jsonl = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/BIG1_diffuseq_ProtMedium_h1024_lr0.0001_t2000_sqrt_lossaware_seed123_ProtMedium1MLsfRoFormerDebug20230610-18:32:34/ema_0.9999_080000.pt.samples/"
    
    # Model with highest lddt
    # pident  evalue   alntmscore  lddt    steps  RMSD     TM-score  aligned_length SeqID
    # 15.1    32.27     0.31805    0.4275  11000  4.18711  0.368439  67.270042      0.06597
    #path_jsonl = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/FINAL12x12MODEL/ema_0.9999_011000.pt.samples/"
    
    compare(path_jsonl)