import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import numpy as np

bert = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMedium_h128_lr0.0001_t2000_sqrt_lossaware_seed123_ProtMediumBERTCompare20230627-12:21:12"
roformer_small = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMedium_h128_lr0.0001_t2000_sqrt_lossaware_seed123_ProtMediumRoCompare20230706-10:27:24"
roformer_large = "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMedium_h1024_lr0.0001_t2000_sqrt_lossaware_seed123_ProtMediumRoFinalCompare20230708-19:39:52"


def parse_RMSD(full_input_path: str, cut_at: int=None, model_name: str=""):
    log_paths = sorted(glob.glob(full_input_path + "/**/*.log"))
        
    total = pd.DataFrame(columns=["file", "RMSD", "TM-score", "aligned_length", "SeqID"])
    
    for log in log_paths:
        df = pd.read_csv(log)
        RMSD = df.RMSD.median()
        TM = df["TM-score"].median()
        aligned_length = df["aligned-length"].median()
        id = df["SeqID"].median()
        total = total.append({"file": log, "model": model_name, "RMSD": RMSD, "TM-score": TM, "aligned_length": aligned_length, "SeqID": id}, ignore_index=True)
        
    steps = total.file.map(lambda x: int(x.split("/")[-2].split("_")[-1].split(".")[0]))
    total.drop(columns=["file"], inplace=True)
    total["steps"] = steps
    
    if cut_at is not None:
        total = total[cut_at:]
        
    return total

def parse_plDDT(full_input_path: str, cut_at: int=None, model_name: str=""):
    log_paths = sorted(glob.glob(full_input_path + "/**/*.csv"))
        
    total = pd.DataFrame(columns=["file", "pLDDT"])
    
    for log in log_paths:
        df = pd.read_csv(log, sep=",")
        plDDT = df["pLDDT"].median()
        total = total.append({"file": log, "model": model_name, "pLDDT": plDDT}, ignore_index=True)
        
    steps = total.file.map(lambda x: int(x.split("/")[-2].split("_")[-1].split(".")[0]))
    total.drop(columns=["file"], inplace=True)
    total["steps"] = steps
    total["pLDDT"] = total["pLDDT"] / 100.
    
    if cut_at is not None:
        total = total[cut_at:]
        
    return total

def parse_m8(full_input_path: str, cut_at: int=None, model_name: str=""):
    m8_paths = sorted(glob.glob(full_input_path + "/**/*parsed.m8"))
    
    total = pd.DataFrame(columns=["file", "pident", "evalue", "alntmscore", "lddt"])
    
    for m8 in m8_paths:
        df = pd.read_csv(m8, sep="\t")
        
        pident = df.pident.median()
        evalue = df.evalue.median()
        alntmscore = df.alntmscore.median()
        lddt = df.lddt.median()
        total = total.append({"file": m8, "model": model_name, "pident": pident, "evalue": evalue, "alntmscore": alntmscore, "lddt": lddt}, ignore_index=True)
    
    steps = total.file.map(lambda x: int(x.split("/")[-2].split("_")[-1].split(".")[0]))
    total.drop(columns=["file"], inplace=True)
    total["steps"] = steps
    
    if cut_at is not None:
        total = total[cut_at:]
    
    return total

bert_rmsd = parse_RMSD(bert, cut_at=0, model_name="BERT")
bert_m8 = parse_m8(bert, cut_at=0, model_name="BERT")
bert_plddt = parse_plDDT(bert, cut_at=0, model_name="BERT")
# join all three on model_name wihout nan values
bert = pd.merge(bert_rmsd, bert_m8, on=["steps", "model"], how="inner")
bert = pd.merge(bert, bert_plddt, on=["steps", "model"], how="inner")


roformer_small_rmsd = parse_RMSD(roformer_small, cut_at=0, model_name="RoFormerH128")
roformer_small_m8 = parse_m8(roformer_small, cut_at=0, model_name="RoFormerH128")
roformer_small_plddt = parse_plDDT(roformer_small, cut_at=0, model_name="RoFormerH128")
roformer_small = pd.merge(roformer_small_rmsd, roformer_small_m8, on=["steps", "model"], how="inner")
roformer_small = pd.merge(roformer_small, roformer_small_plddt, on=["steps", "model"], how="inner")

roformer_large_rmsd = parse_RMSD(roformer_large, cut_at=0, model_name="RoFormerH1024")
roformer_large_m8 = parse_m8(roformer_large, cut_at=0, model_name="RoFormerH1024")
roformer_large_plddt = parse_plDDT(roformer_large, cut_at=0, model_name="RoFormerH1024")
roformer_large = pd.merge(roformer_large_rmsd, roformer_large_m8, on=["steps", "model"], how="inner")
roformer_large = pd.merge(roformer_large, roformer_large_plddt, on=["steps", "model"], how="inner")

parse_step = lambda x: str(x)[:-3] + "K"

bert["parsed_step"] = bert["steps"].apply(parse_step)
roformer_small["parsed_step"] = roformer_small["steps"].apply(parse_step)
roformer_large["parsed_step"] = roformer_large["steps"].apply(parse_step)

# plot all dataframes
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

# increase font size
plt.rcParams.update({'font.size': 16, 'legend.fontsize': 16, 'axes.labelsize': 160, 'axes.titlesize': 16})

sns.lineplot(data=bert, x="parsed_step", y="RMSD", ax=axs[0, 0], label="BERT", linewidth=2.5, color="grey")
sns.lineplot(data=roformer_small, x="parsed_step", y="RMSD", ax=axs[0, 0], label="RoFormerH128", linewidth=2.5, color="orange")
sns.lineplot(data=roformer_large, x="parsed_step", y="RMSD", ax=axs[0, 0], label="RoFormerH1024", linewidth=2.5, color="#0169B2")
axs[0, 0].set_ylabel("RMSD", fontsize=16)
axs[0, 0].tick_params(axis='both', which='major', labelsize=14)

sns.lineplot(data=bert, x="parsed_step", y="pLDDT", ax=axs[1, 0], label="BERT", linewidth=2.5, color="grey")
sns.lineplot(data=roformer_small, x="parsed_step", y="pLDDT", ax=axs[1, 0], label="RoFormerH128", linewidth=2.5, color="orange")
sns.lineplot(data=roformer_large, x="parsed_step", y="pLDDT", ax=axs[1, 0], label="RoFormerH1024", linewidth=2.5, color="#0169B2")
axs[1, 0].set_ylabel("pLDDT", fontsize=16)
axs[1, 0].tick_params(axis='both', which='major', labelsize=14)

sns.lineplot(data=bert, x="parsed_step", y="lddt", ax=axs[2, 0], label="BERT", linewidth=2.5, color="grey")
sns.lineplot(data=roformer_small, x="parsed_step", y="lddt", ax=axs[2, 0], label="RoFormerH128", linewidth=2.5, color="orange")
sns.lineplot(data=roformer_large, x="parsed_step", y="lddt", ax=axs[2, 0], label="RoFormerH1024", linewidth=2.5, color="#0169B2")
axs[2, 0].set_ylabel("lDDT", fontsize=16)
axs[2, 0].tick_params(axis='both', which='major', labelsize=14) 

sns.lineplot(data=bert, x="parsed_step", y="TM-score", ax=axs[0, 1], label="BERT", linewidth=2.5, color="grey")
sns.lineplot(data=roformer_small, x="parsed_step", y="TM-score", ax=axs[0, 1], label="RoFormerH128", linewidth=2.5, color="orange")
sns.lineplot(data=roformer_large, x="parsed_step", y="TM-score", ax=axs[0, 1], label="RoFormerH1024", linewidth=2.5, color="#0169B2")
axs[0, 1].set_ylabel("TM-Score", fontsize=16)
axs[0, 1].tick_params(axis='both', which='major', labelsize=14)

sns.lineplot(data=bert, x="parsed_step", y="aligned_length", ax=axs[1, 1], label="BERT", linewidth=2.5, color="grey")
sns.lineplot(data=roformer_small, x="parsed_step", y="aligned_length", ax=axs[1, 1], label="RoFormerH128", linewidth=2.5, color="orange")
sns.lineplot(data=roformer_large, x="parsed_step", y="aligned_length", ax=axs[1, 1], label="RoFormerH1024", linewidth=2.5, color="#0169B2")
axs[1, 1].set_ylabel("Aligned Length", fontsize=16)
axs[1, 1].tick_params(axis='both', which='major', labelsize=14)

sns.lineplot(data=bert, x="parsed_step", y="evalue", ax=axs[2, 1], label="BERT", linewidth=2.5, color="grey")
sns.lineplot(data=roformer_small, x="parsed_step", y="evalue", ax=axs[2, 1], label="RoFormerH128", linewidth=2.5, color="orange")
sns.lineplot(data=roformer_large, x="parsed_step", y="evalue", ax=axs[2, 1], label="RoFormerH1024", linewidth=2.5, color="#0169B2")
axs[2, 1].set_ylabel("E-value", fontsize=16)
axs[2, 1].tick_params(axis='both', which='major', labelsize=14)

axs[0, 0].set_ylabel("RMSD")
axs[1, 0].set_ylabel("pLDDT")   
axs[2, 0].set_ylabel("lDDT")
axs[0, 1].set_ylabel("TM-Score")
axs[1, 1].set_ylabel("Aligned Length")
axs[2, 1].set_ylabel("E-value")

# remove x labels
for ax in axs.flatten():
    ax.set_xlabel("")
    
axs[2, 0].set_xlabel("Steps", fontsize=16)
axs[2, 1].set_xlabel("Steps", fontsize=16)

fig.suptitle("Evolution of Evaluation Metrics during Training")
plt.tight_layout()
plt.show()


# compare the three models at step 50000
bert_50k = bert[bert["steps"] == 50000]
roformer_large_50k = roformer_large[roformer_large["steps"] == 50000]
roformer_small_50k = roformer_small[roformer_small["steps"] == 50000]

import matplotlib.pyplot as plt
import numpy as np

# get values for each model at step 50000
bert_50k = bert[bert["steps"] == 50000]
roformer_large_50k = roformer_large[roformer_large["steps"] == 50000]
roformer_small_50k = roformer_small[roformer_small["steps"] == 50000]

# concatentate all three dataframes
concat = pd.concat([bert_50k, roformer_small_50k, roformer_large_50k])
concat.index = concat["model"]
concat.drop(columns=["model"], inplace=True)

print(concat)
concat["evalue"] = concat["evalue"] / 10
concat["Scaled RMSD"] = concat["RMSD"] / concat["RMSD"].max()
# plot all values in one plot
concat[["lddt", "pLDDT", "TM-score", "evalue", "Scaled RMSD"]].T.plot.bar(rot=0, color=["grey", "orange", "#0169B2"])

plt.xticks(range(5), ["lDDT", "plDDT / 100", "TM-Score", "E-value", "Scaled RMSD"])
plt.title("Comparison of median values")
plt.show()