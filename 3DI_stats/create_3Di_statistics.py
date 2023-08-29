import os
import sys
import pandas as pd
from tqdm import tqdm

THREE_DI_PATH = "3DI_stats/PDB_ss.fasta"
PDB_PATH = "3DI_stats/ss_dis.txt"



def create_df_from_pdb():
    # ✅ DONE
    # create a new dataframe with the columns id, seq, ss, dis
    df = pd.DataFrame(columns=["id", "seq", "ss", "dis"])

    with open(PDB_PATH) as f:
        # get the id of the first sequence
        current_id, annot = f.readline()[1:].split(":")[0:2] # get the id and annotation of the first sequence
        sequence = ""
        secstr = ""
        disorder = ""

        current_entry = "sequence"
        # entry lists
        ids = []
        annots = []
        sequences = []
        secstrs = []
        disorders = []
        for i, line in tqdm(enumerate(f), total=7_246_177):
            if line.startswith(">"):
                id, annot = line[1:].split(":")[0:2]
                if id != current_id: # append to df when all information was gathered

                    
                    ids.append(current_id)
                    annots.append(annot)
                    sequences.append(sequence)
                    secstrs.append(secstr)
                    disorders.append(disorder)
                    
                    
                    current_id = id
                    sequence = secstr = disorder = ""
                    current_entry = "sequence"
                else: # find out which sequence type is next
                    if "sequence" in line:
                        current_entry = "sequence"
                    elif "secstr" in line:
                        current_entry = "secstr"
                    elif "disorder" in line:
                        current_entry = "disorder"
            # if no header was detected, we append the line to the current entry type   
            elif current_entry == "sequence":
                sequence += line.strip()
            elif current_entry == "secstr":
                # other is represented by a whitespace, so we replace it with a dash
                secstr += line.replace(" ", "-").strip()
            elif current_entry == "disorder":
                disorder += line.strip()
    df = pd.DataFrame({"id": ids, "annot": annots, "sequence": sequences, "secstr": secstrs , "disorder": disorders})
    df.to_csv("3DI_stats/PDB_ss_dis.csv")
    
    
def pre_process_pdb():
    # ✅ DONE
    print("Loading PDB_ss_dis.csv...")
    df = pd.read_csv("3DI_stats/PDB_ss_dis.csv")
    print("Removing whitespaces...")

    remove_residues = lambda first, second: ''.join([res for res, mask in zip(first, second) if mask != 'X'])
    
    # I want to apply remove_residues to the columns sequence and secstr and disorder is the second argument
    print("Process sequence...")
    df["sequence"] = df.apply(lambda x: remove_residues(x.sequence, x.disorder), axis=1)
    print("Process secstr...")
    df["secstr"] = df.apply(lambda x: remove_residues(x.secstr, x.disorder), axis=1)
    print("Process disorder...")
    df["disorder"] = df.apply(lambda x: remove_residues(x.disorder, x.disorder), axis=1)
    
    print("Drop disorder...")
    df.drop(columns=["Unnamed: 0", "disorder"], inplace=True)

    print("Save to csv...")
    df.to_csv("3DI_stats/PDB_ss_dis_clean.csv")
    

    
                    
def add_3DI_to_df():
    print("Loading PDB_ss_dis_clean.csv...")
    df = pd.read_csv("3DI_stats/PDB_ss_dis_clean.csv")
    
    three_di_df = pd.DataFrame(columns=["id", "annot", "3DI"])
    ids = []
    annots =[]
    threedis = []
    with open(THREE_DI_PATH) as f:
        for i, line in tqdm(enumerate(f), total=548_573//2):
            if line.startswith(">"):
                id, annot = line[1:].strip().split("_")[0:2]
                id, annot = id.upper(), annot.upper().split(" ")[0]
                three_di = f.readline().strip()
                ids.append(id)
                annots.append(annot)
                threedis.append(three_di.lower())
    
    
    three_di_df = pd.DataFrame({"id": ids, "annot": annots, "3DI": threedis})  
        
    # merge df and three_di_df on id and annot
    df = pd.merge(df, three_di_df, on=["id", "annot"])
    
    # remove columns with NaN
    print("Drop NaN...")
    df.dropna(inplace=True)
    
    # remove columns where len(3Di) != len(sequence)
    print("Drop rows where len(3DI) != len(sequence)...")
    df = df[df.apply(lambda x: len(x["3DI"]) == len(x["sequence"]), axis=1)]
    
    
    print("drop columns clean...")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    
    print(f"Save to csv... Shape: {df.shape}")
    df.to_csv("3DI_stats/seq_ss_3Di.csv", index=False)
    
                
    
def stats_secondary(state: int=3, seq_or_3Di: str="3Di"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pprint
    
    # set font size
    plt.rcParams.update({'font.size': 18, 'legend.fontsize': 12})
    
    df = pd.read_csv("3DI_stats/seq_ss_3Di.csv")

    # join all sequences to a string
    all_secstr = "".join(df.secstr.values)
    if seq_or_3Di == "3Di":
        all_3Di = "".join(df["3DI"].values)
    elif seq_or_3Di == "seq":
        all_3Di = "".join(df.sequence.values)
    else:
        raise ValueError("seq_or_3Di must be either '3Di' or 'seq'")
    
    # i want the correlation between the 3DI and the secondary structure
    print("loop")
    amino_acid_frequency = {}
    for aa, ss in zip(all_secstr, all_3Di):
        if ss not in amino_acid_frequency:
            amino_acid_frequency[ss] = {}
        amino_acid_frequency[ss][aa] = amino_acid_frequency[ss].get(aa, 0) + 1

    df = pd.DataFrame(amino_acid_frequency).fillna(0).astype(float)
    
    if seq_or_3Di == "seq":
        df.drop(columns=["O", "B", "U"], inplace=True)
    
    
    if state == 3:
        # I have a mapping  
        # [G,H,I] → H (helix), [B,E] → E (strand), all others to O (other)
        # merge all indeces based on that mapping
        df = df.groupby(by={"G": "Helix", "H": "Helix", "I": "Helix", # Helix
                            "B": "Ex. strand", "E": "Ex. strand", # Strand
                            "-": "Other", "T": "Other", "S": "Other"}).sum() # Other
    
    # transpose df
    df = df.T
    # sort by index
    df = df.sort_index()
    
    df.plot(kind='bar', figsize=(10, 6))
    string_name = "3Di" if seq_or_3Di == "3Di" else "Amino Acid"
    plt.ylabel("Absolute Frequency")
    plt.xlabel(f"{string_name} Residue")
    plt.show()
    
    
    # normalize by 3Di
    df_norm_3di = df.div(df.sum(axis=1), axis=0)
    # normalize by SS
    df_norm_ss = df.div(df.sum(axis=0), axis=1)
    # transpose so that SS is the index (y-axis)
    df_norm_ss = df_norm_ss.T

    # heatmap of df
    
    plt.figure(figsize=(10, 10))
    #sns.heatmap(df_norm_3di, annot=True, fmt="0.4f", cmap="Blues", annot_kws={"size": 12})
    # add red border to max value in each row
    sns.heatmap(df_norm_3di, annot=True, fmt="0.3f", cmap="coolwarm", annot_kws={"size": 12})
    # Add red dot to highlight the highest value in each row
    for i in range(df_norm_3di.shape[0]):
        max_index = np.argmax(df_norm_3di.iloc[i, :])
        plt.plot(max_index + .15, i + 0.5, 'ro', markersize=6)
    plt.title(f"Normalized by {string_name}")
    plt.xlabel(f"{state}-State Secondary Structure")
    plt.ylabel(f"{string_name} Residue")
    plt.show()

    plt.figure(figsize=(16, 6))
    sns.heatmap(df_norm_ss, annot=True, fmt="0.2f", cmap="coolwarm", annot_kws={"size": 12})
    for i in range(df_norm_ss.shape[0]):
        max_index = np.argmax(df_norm_ss.iloc[i, :])
        plt.plot(max_index + 0.5, i + 0.75, 'ro', markersize=6)
    plt.title(f"Normalized by Secondary Structure")
    plt.ylabel(f"{state}-State Secondary Structure")
    plt.xlabel(f"{string_name} residue")
    plt.show()
    
    
def stats_3di_aa():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pprint
    
    # set font size
    plt.rcParams.update({'font.size': 18, 'legend.fontsize': 12})
    
    df = pd.read_csv("3DI_stats/seq_ss_3Di.csv")

    # join all sequences to a string
    all_3di = "".join(df["3DI"].values)
    all_aa = "".join(df.sequence.values)

    # i want the correlation between the 3DI and the secondary structure
    print("loop")
    amino_acid_frequency = {}
    for aa, tdi in zip(all_aa, all_3di):
        if tdi not in amino_acid_frequency:
            amino_acid_frequency[tdi] = {}
        amino_acid_frequency[tdi][aa] = amino_acid_frequency[tdi].get(aa, 0) + 1

    df = pd.DataFrame(amino_acid_frequency).fillna(0).astype(float)
    
    print(df)
    
    # transpose df
    df = df.T
    # sort by index
    df = df.sort_index()
    
    df.drop(columns=["O", "B", "U"], inplace=True)
        
    
    df_norm_3di = df.div(df.sum(axis=1), axis=0)
    # normalize by SS
    df_norm_aa = df.div(df.sum(axis=0), axis=1)
    # transpose so that SS is the index (y-axis)
    df_norm_aa = df_norm_aa.T

    # heatmap of df
    
    plt.figure(figsize=(10, 10))
    #sns.heatmap(df_norm_3di, annot=True, fmt="0.4f", cmap="Blues", annot_kws={"size": 12})
    # add red border to max value in each row
    sns.heatmap(df_norm_3di, annot=True, fmt="0.3f", cmap="coolwarm", annot_kws={"size": 12})
    # Add red dot to highlight the highest value in each row
    for i in range(df_norm_3di.shape[0]):
        max_index = np.argmax(df_norm_3di.iloc[i, :])
        plt.plot(max_index + .15, i + 0.5, 'ro', markersize=6)
    plt.title(f"Normalized by 3Di")
    plt.xlabel(f"Amino Acid")
    plt.ylabel(f"3Di residue")
    plt.show()
    
    plt.figure(figsize=(10, 10))
    #sns.heatmap(df_norm_3di, annot=True, fmt="0.4f", cmap="Blues", annot_kws={"size": 12})
    # add red border to max value in each row
    sns.heatmap(df_norm_aa, annot=True, fmt="0.3f", cmap="coolwarm", annot_kws={"size": 12})
    # Add red dot to highlight the highest value in each row
    for i in range(df_norm_aa.shape[0]):
        max_index = np.argmax(df_norm_aa.iloc[i, :])
        plt.plot(max_index + .15, i + 0.5, 'ro', markersize=6)
    plt.title(f"Normalized by Amino Acid")
    plt.xlabel(f"Amino Acid")
    plt.ylabel(f"3Di residue")
    plt.show()
    
    
def create_sankey(state: int=3, norm_by: str=None, seq_or_3Di: str="3Di"):
    
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import numpy as np
    df = pd.read_csv("3DI_stats/seq_ss_3Di.csv")

    # join all sequences to a string
    all_secstr = "".join(df.secstr.values)
    if seq_or_3Di == "3Di":
        all_3Di = "".join(df["3DI"].values)
    elif seq_or_3Di == "seq":
        all_3Di = "".join(df.sequence.values)
    else:
        raise ValueError("seq_or_3Di must be either '3Di' or 'seq'")
    
    # i want the correlation between the 3DI and the secondary structure
    print("loop")
    amino_acid_frequency = {}
    for aa, ss in zip(all_secstr, all_3Di):
        if ss not in amino_acid_frequency:
            amino_acid_frequency[ss] = {}
        amino_acid_frequency[ss][aa] = amino_acid_frequency[ss].get(aa, 0) + 1

    df = pd.DataFrame(amino_acid_frequency).fillna(0).astype(float)
    
    
    if state == 3:
        # [G,H,I] → H (helix), [B,E] → E (strand), all others to O (other)
        # merge all indeces based on that mapping
        df = df.groupby(by={"G": "Helix", "H": "Helix", "I": "Helix", # Helix
                            "B": "Strand", "E": "Strand", # Strand
                            "-": "Other", "T": "Other", "S": "Other"}).sum() # Other
    
    # transpose df
    df = df.T
    # sort by index
    df = df.sort_index()
    
    if norm_by == "3Di":
        df = df.div(df.sum(axis=1), axis=0)
        # multiply by 100 to get percentage
        df = df * 100
    elif norm_by == "SS":
        df = df.div(df.sum(axis=0), axis=1)
        df = df * 100
    else: 
        ...
    
    #labels = df.index.values.tolist() + df.columns.values.tolist()
    labels = df.index.tolist() + df.columns.values.tolist()

    values = df.values.flatten().tolist()

    def get_index(label):
        return labels.index(label)
    
    source = []
    target = []
    values = []
    
    for i in df.itertuples():
        print(i)
        for attribute, value in i._asdict().items():
            if attribute == "Index":
                source += [get_index(value) for _ in range(len(i) -1)]
                continue
            if attribute == "_1":
                target.append(get_index("-"))
            else:
                target.append(get_index(attribute))
            values.append(value)
            
    assert len(source) == len(target) == len(values), "Lengths of source, target and values are not equal"
    # Generate a color palette with 20 colors for color-blind people
    tmp_colors = plt.cm.turbo(np.linspace(1, 0, len(df.columns) + len(df.index)))
    tmp_colors = list(reversed([plt.cm.colors.to_hex(color) for color in tmp_colors]))
        
    color_map = {get_index(label): color for label, color in zip(labels, tmp_colors)}
    
    def hex_to_rgba(hex_color, alpha):
        # Remove the '#' character if present
        hex_color = hex_color.lstrip('#')

        # Convert the hexadecimal color to RGB
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Create the RGBA color with the specified alpha value
        rgba = rgb + (alpha,)

        return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"

    def rgba_to_hex(rgba_color):
        # Extract the red, green, blue, and alpha values
        red, green, blue, alpha = rgba_color
        
        # Convert each value to its hexadecimal representation
        red_hex = format(red, '02x')
        green_hex = format(green, '02x')
        blue_hex = format(blue, '02x')
        alpha_hex = format(int(alpha * 255), '02x')  # Scale alpha to 0-255 range and convert to hexadecimal
        
        # Combine the hexadecimal values
        hex_color = '#' + red_hex + green_hex + blue_hex + alpha_hex
        
        return hex_color
    
    colors = [color_map[label] for label in source]
    print(colors)
    colors = [hex_to_rgba(color, 0.8) for color in colors]
    print(colors)
    plt.bar(x=labels, height=1, color=tmp_colors)
    plt.show()
    
    import plotly.express as px
    
    fig = go.Figure(data=[go.Sankey(
        arrangement = "snap",
        node = dict(
            pad = 10,
            thickness = 10,
            line = dict(color = "black", width = 0.5),
            label = labels, 
            x = [0.01 for _ in range(len(df.index))] + [0.99 for _ in range(len(df.columns))],
            y = [0.01] + [i/len(df.index) for i in range(1, len(df.index))] + [i/len(df.columns) for i in range(len(df.columns))],
            color = tmp_colors
        ),
        link = dict(
            source = source,
            target = target,
            value = values, 
            color = colors
        )
    )])

    # put title in the middle
    fig.update_layout(title_x=0.5)
    string_name = "3Di" if seq_or_3Di == "3Di" else "Amino acid"
    fig.update_layout(title_text=f"{string_name} to Structure Mapping", font_size=35)
    fig.show()
    

    

if __name__ == "__main__":
    # create_df_from_pdb()
    # pre_process_pdb()
    #add_3DI_to_df()
    # Example usage
    #stats_secondary(state=3, seq_or_3Di="3Di")
    #stats_3di_aa()
    create_sankey(state=3, norm_by=None, seq_or_3Di="3Di")
