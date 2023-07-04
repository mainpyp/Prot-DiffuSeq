import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from collections import Counter
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--valid', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    return parser.parse_args()


def load_data(path:str) -> pd.DataFrame:
    with open(path, 'r') as f:
        srcs, targets = [], []
        print('Loading data...')
        i = 0
        for line in f:
            i += 1
            if i % 1_000_000 == 0:
                print(f'Loaded {i} lines...')
                
            l = json.loads(line)
            src, target = l['src'], l['trg']
            srcs.append(src.replace(' ', ''))
            targets.append(target.replace(' ', ''))
        print('Done!')
    df = pd.DataFrame(columns=['src', 'trg'])
    df['src'] = srcs
    df['trg'] = targets
    df['length'] = df.src.apply(len)
    return df


def get_column_counter(series:pd.Series) -> Counter:
    """Get the Counter of a pandas Series"""
    c = Counter()
    for s in series:
        c.update(s[:512])
    return c


def main(args):
    train = load_data(args.train)
    test = load_data(args.test)
    valid = load_data(args.valid)
    
    # create three subplots
    # order axis so that ax1 spans the whole width and ax2 and ax3 are next to each other below ax1
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax5 = plt.subplot2grid((3, 2), (2, 1))


    ######################################
    ### 1 PLOT THE LENGTH DISTRIBUTION ###
    ######################################
    # normalize the counts
    # plot the normalized distribution
    print('Plotting length distribution...')
    sns.histplot(train.length, bins=50, ax=ax1, alpha=0.8, stat='percent')
    sns.histplot(test.length, bins=50, ax=ax1, alpha=0.8, stat='percent')
    sns.histplot(valid.length, bins=50, ax=ax1, alpha=0.8, stat='percent')
    
    
    
    # add vertical line at 512 to show the truncation
    ax1.axvline(512, color='orange', linestyle='--', label='cutoff')
    # add annotation to that line with the title cutoff and rotate it 90 degrees
    ax1.annotate('cutoff (512)', xy=(530, 0.5), xytext=(530, 10), rotation=0, color='orange', fontsize=8)
    
    from matplotlib.patches import Patch
    a = Patch([], [], color='#0673B2', label='train')
    b = Patch([], [], color='#DE8E03', label='test')
    c = Patch([], [], color='#089E73', label='valid')
    d = Patch([], [], color='#325880', label='PDB')
    
    ax1.set_xlim(0, 1500)
    ax1.set_xlabel('Length of source sequence')
    ax1.set_title('Distribution of sequence lengths')
    ax1.legend(handles=[a, b, c, d])
    
    
    ###########################################
    ### 2 PLOT THE 3DI RESIDUE DISTRIBUTION ###
    ###########################################
    print('Plotting 3DI residue distribution...')
    scr_counter_train = get_column_counter(train.src)
    scr_counter_test = get_column_counter(test.src)
    scr_counter_valid = get_column_counter(valid.src)
    # put all counters in a pandas DataFrame
    scr_counter = pd.DataFrame(columns=['train', 'test', 'valid'])
    scr_counter['train'] = scr_counter_train
    scr_counter['test'] = scr_counter_test
    scr_counter['valid'] = scr_counter_valid
    
    # normalize the counts
    scr_counter['train_norm'] = scr_counter['train'] / sum(scr_counter['train'])
    scr_counter['test_norm'] = scr_counter['test'] / sum(scr_counter['test'])
    scr_counter['valid_norm'] = scr_counter['valid'] / sum(scr_counter['valid'])
    
    # plot the normalized distribution, the bars should be next to each other and sort by x axis name
    scr_counter = scr_counter.sort_index()
    # add horizontal grid in 0.05 repetition
    ax2.grid(axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    scr_counter[['train_norm', 'test_norm', 'valid_norm']].plot.bar(ax=ax2)
    
    print(f"3Di source counter train: {scr_counter['train']}")
    print(f"3Di source counter test: {scr_counter['test']}")
    print(f"3Di source counter valid: {scr_counter['valid']}")
    
    ax2.set_xlabel('3Di')
    ax2.set_ylabel('Percentage')
    ax2.get_legend().remove()

    # turn the xticks
    ax2.tick_params(axis='x', rotation=0)
   
    
    ###########################################
    ### 3 PLOT THE AAs RESIDUE DISTRIBUTION ###
    ###########################################
    print('Plotting amino acid distribution...')
    trg_counter_train = get_column_counter(train.trg)
    trg_counter_test = get_column_counter(test.trg)
    trg_counter_valid = get_column_counter(valid.trg)
    # put all counters in a pandas DataFrame
    trg_counter = pd.DataFrame(columns=['train', 'test', 'valid'])
    trg_counter['train'] = trg_counter_train
    trg_counter['test'] = trg_counter_test
    trg_counter['valid'] = trg_counter_valid
    
    # normalize the counts
    trg_counter['train_norm'] = trg_counter['train'] / sum(trg_counter['train'])
    trg_counter['test_norm'] = trg_counter['test'] / sum(trg_counter['test'])
    trg_counter['valid_norm'] = trg_counter['valid'] / sum(trg_counter['valid'])
    
    # plot the normalized distribution, the bars should be next to each other and sort by x axis name
    trg_counter = trg_counter.sort_index()
    trg_counter[['train_norm', 'test_norm', 'valid_norm']].plot.bar(ax=ax3)
    
    print(f"AA target counter train: {trg_counter['train']}")
    print(f"AA target counter test: {trg_counter['test']}")
    print(f"AA target counter valid: {trg_counter['valid']}")
    
    ax3.set_xlabel('Amino Acid')
    # turn the xticks
    ax3.tick_params(axis='x', rotation=0)
    
    ax3.get_legend().remove()
    
    
    #############################################
    ### PLOT THE PDB 3DI RESIDUE DISTRIBUTION ###
    #############################################
    df = pd.read_csv("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/3DI_stats/seq_ss_3Di.csv")
    
    # join all sequences to a string
    all_secstr = "".join(df.secstr.values)
    all_3Di = "".join(df["3DI"].values)
    
    # i want the correlation between the 3DI and the secondary structure
    print("loop")
    amino_acid_frequency = {}
    for aa, ss in zip(all_secstr, all_3Di):
        if ss not in amino_acid_frequency:
            amino_acid_frequency[ss] = {}
        amino_acid_frequency[ss][aa] = amino_acid_frequency[ss].get(aa, 0) + 1

    df = pd.DataFrame(amino_acid_frequency).fillna(0).astype(float)
    
    df = df.groupby(by={"G": "Helix", "H": "Helix", "I": "Helix", # Helix
                        "B": "Ex. strand", "E": "Ex. strand", # Strand
                        "-": "Other", "T": "Other", "S": "Other"}).sum() # Other
    
    # transpose df
    df = df.T
    # sort by index
    df = df.sort_index()
    
    # normalize
    df = df / df.sum()
    print("#"*50)
    print(df["Helix"].sum())
    
    df.plot(kind='bar', figsize=(10, 6), ax=ax4)
    ax4.set_title("PDB 3Di")
    ax4.set_ylabel("Percentage")
    ax4.tick_params(axis='x', rotation=0)
    ax4.get_legend().remove()
    
    # add very small legend to the middle of the plot and the legend should be flat
    ax4.legend(loc='upper center', ncol=3, fancybox=True, shadow=True, fontsize=8)
    
    
    #############################################
    ### PLOT THE PDB AA RESIDUE DISTRIBUTION ###
    #############################################
    df = pd.read_csv("../3DI_stats/seq_ss_3Di.csv")
    c = get_column_counter(df["sequence"])
    
    # create to dataframe, noramlize and sort by index
    df = pd.DataFrame(c, index=["Amino acid"]).T
    df = df / sum(df["Amino acid"])
    df = df.sort_index()
    df.drop(["X", "U"], inplace=True)
    # plot
    df.plot(kind='bar', figsize=(10, 6), color="#325880", ax=ax5)
    
    # rotate xticks
    ax5.tick_params(axis='x', rotation=0)
    ax5.get_legend().remove()   
    ax5.set_title("PDB")
    # set y lim to 0.1
    ax5.set_ylim(0, 0.1)

    
    plt.tight_layout()
    print(f'Saving plot to {args.output_path}')
    # limit x axis of ax3 to 1500
    
    plt.show()
    #plt.savefig(args.output_path)



if __name__ == '__main__':
    # set current dir as working dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    PALETTE = sns.color_palette("colorblind", 5)
    sns.set_palette(PALETTE)
    sns.set_style("whitegrid")
    # remove vertical grid lines
    sns.set_style({"axes.grid": False})

    
    # set seaborn font sizes to something readable
    sns.set_context("paper", font_scale=2)
    
    args = parse_args()
    
    # output path is the input path without the last extension file
    output_path = args.train.split('.')[0] + "_dataset_distributions.png"
    args.output_path = output_path
    
    main(args)
    
    
    