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
            srcs.append(src.replace(' ', '').strip())
            targets.append(target.replace(' ', '').strip())
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


def main(args, save=False):
    train = load_data(args.train)
    test = load_data(args.test)
    valid = load_data(args.valid)
    pdb_df = pd.read_csv("../3DI_stats/seq_ss_3Di.csv")
    print("VALID")
    print(valid.length.describe())
    print("TEST")
    print(test.length.describe())
    print("TRAIN")
    print(train.length.describe())
    
    pdb_df['length'] = pdb_df['sequence'].apply(len)
    print(pdb_df)
    # calculate the mean, mediad, std, 25 percentile and 75 percentile of the train length
    
    
    stats_dict = dict()
    for name, d in zip(["train", "test", "valid", "pdb"], [train, test, valid, pdb_df]):
        mean = d.length.mean()
        median = d.length.median()
        std = d.length.std()
        p25 = d.length.quantile(0.25)
        p75 = d.length.quantile(0.75)
        
        # check percentile 125
        train_125 = len(d[d.length <= 125]) / len(d)
        # check which percentile the 255 cutoff is
        train_253 = len(d[d.length <= 255]) / len(d)
        # check which percentile the 508 cutoff is
        train_508 = len(d[d.length <= 508]) / len(d)
        stats_dict[name] = [mean, median, std, p25, p75, train_125*100, train_253*100, train_508*100]
    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['mean', 'median', 'std', 'p25', 'p75', '125', '253', '508'])
    # round everything to 2 decimals
    stats_df = stats_df.round(2)
    stats_df = stats_df.transpose()
    print(stats_df)
        
    print("TRAIN")
    print(f"Mean: {mean:.2f} Median: {median:.2f} Std: {std:.2f} 25%: {p25:.2f} 75%: {p75:.2f} 253: {train_253*100:.2f}")

    # create three boxplots that demonstrate the length distribution
    plt.boxplot([train.length, test.length, valid.length, pdb_df.length], labels=['train', 'test', 'valid', 'pdb'], whis=3)
    # limit y axis ti 1500
    plt.ylim(0, 1500)
    plt.title('Length distribution')
    plt.ylabel("Sequence Length")
    
    plt.tight_layout()
    
    if save:
        print(f'Saving plot to {args.output_path + "_box.png"}')
        plt.savefig(args.output_path + '_box.png')
    else:
        plt.show()
    
    
    
    # create three subplots
    # order axis so that ax1 spans the whole width and ax2 and ax3 are next to each other below ax1
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax3 = plt.subplot2grid((3, 2), (1, 1))


    ######################################
    ### 1 PLOT THE LENGTH DISTRIBUTION ###
    ######################################
    # normalize the counts
    # plot the normalized distribution
    print('Plotting length distribution...')
    sns.kdeplot(train.length, ax=ax1, linewidth=2, fill=True, alpha=0.1)
    sns.kdeplot(test.length, ax=ax1, linewidth=2, fill=True, alpha=0.1)
    sns.kdeplot(valid.length, ax=ax1, linewidth=2, fill=True, alpha=0.1)
    # add sns displot to ax1
    sns.kdeplot(pdb_df.length, ax=ax1, linewidth=2, color='#325880', label='PDB', fill=True, alpha=0.1)
    #sns.displot(pdb_df.length, ax=ax1, kind="kde")
    
    
    
    
    
    # add vertical line at 512 to show the truncation
    #ax1.axvline(512, color='orange', linestyle='--', label='cutoff')
    # add annotation to that line with the title cutoff and rotate it 90 degrees
    ax1.annotate('cutoff (512)', xy=(530, 0.0025), xytext=(530, 0.0025), rotation=0, color='orange', fontsize=8)
    
    from matplotlib.patches import Patch
    a = Patch([], [], color='#0673B2', label='train')
    b = Patch([], [], color='#DE8E03', label='test')
    c = Patch([], [], color='#089E73', label='valid')
    d = Patch([], [], color='#325880', label='PDB')
    
    ax1.set_xlim(0, 512)
    ax1.set_xlabel('Length of source sequence')
    ax1.legend(handles=[a, b, c, d])
    
    
    ###########################################
    ### 2 PLOT THE 3DI RESIDUE DISTRIBUTION ###
    ###########################################
    print('Plotting 3DI residue distribution...')
    print(pdb_df)
    scr_counter_train = get_column_counter(train.src)
    scr_counter_test = get_column_counter(test.src)
    scr_counter_valid = get_column_counter(valid.src)
    pdb_counter_3di = get_column_counter(pdb_df["3DI"])
    # put all counters in a pandas DataFrame
    scr_counter = pd.DataFrame(columns=['train', 'test', 'valid'])
    scr_counter['train'] = scr_counter_train
    scr_counter['test'] = scr_counter_test
    scr_counter['valid'] = scr_counter_valid
    scr_counter['pdb'] = pdb_counter_3di
    
    # normalize the counts
    scr_counter['train_norm'] = scr_counter['train'] / sum(scr_counter['train'])
    scr_counter['test_norm'] = scr_counter['test'] / sum(scr_counter['test'])
    scr_counter['valid_norm'] = scr_counter['valid'] / sum(scr_counter['valid'])
    scr_counter['pdb_norm'] = scr_counter['pdb'] / sum(scr_counter['pdb'])
    
    # plot the normalized distribution, the bars should be next to each other and sort by x axis name
    scr_counter = scr_counter.sort_index()
    # add horizontal grid in 0.05 repetition
    ax2.grid(axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    scr_counter[['train_norm', 'test_norm', 'valid_norm', 'pdb_norm']].plot.bar(ax=ax2)
    
    # color the bars of pdb
    for i in range(60, 80):
        ax2.patches[i].set_facecolor('#325880')
    
    print(f"3Di source counter train: {scr_counter['train']}")
    print(f"3Di source counter test: {scr_counter['test']}")
    print(f"3Di source counter valid: {scr_counter['valid']}")
    
    ax2.set_xlabel('3Di Token')
    ax2.set_ylabel('Percentage')
    ax2.get_legend().remove()

    # turn the xticks
    ax2.tick_params(axis='x', rotation=0)
   
    
    ###########################################
    ### 3 PLOT THE AAs RESIDUE DISTRIBUTION ###
    ###########################################
     
    # create to dataframe, noramlize and sort by index    
    print('Plotting amino acid distribution...')
    trg_counter_train = get_column_counter(train.trg)
    trg_counter_test = get_column_counter(test.trg)
    trg_counter_valid = get_column_counter(valid.trg)
    pdb_counter = get_column_counter(pdb_df["sequence"])
    # put all counters in a pandas DataFrame
    trg_counter = pd.DataFrame(columns=['train', 'test', 'valid', 'pdb'])
    trg_counter['train'] = trg_counter_train
    trg_counter['test'] = trg_counter_test
    trg_counter['valid'] = trg_counter_valid
    trg_counter['pdb'] = pdb_counter
    
    print(trg_counter)
    
    # normalize the counts
    trg_counter['train_norm'] = trg_counter['train'] / sum(trg_counter['train'])
    trg_counter['test_norm'] = trg_counter['test'] / sum(trg_counter['test'])
    trg_counter['valid_norm'] = trg_counter['valid'] / sum(trg_counter['valid'])
    trg_counter['pdb_norm'] = trg_counter['pdb'] / sum(trg_counter['pdb'])
    
    # plot the normalized distribution, the bars should be next to each other and sort by x axis name
    trg_counter = trg_counter.sort_index()
    trg_counter[['train_norm', 'test_norm', 'valid_norm', 'pdb_norm']].plot.bar(ax=ax3)
    
    # color the bars of all pdb_norm bars
    for i in range(60, 80):
        ax3.patches[i].set_facecolor('#325880')
    
    print(f"AA target counter train: {trg_counter['train']}")
    print(f"AA target counter test: {trg_counter['test']}")
    print(f"AA target counter valid: {trg_counter['valid']}")
    
    ax3.set_xlabel('Amino Acid Residue')
    # turn the xticks
    ax3.tick_params(axis='x', rotation=0)
    ax3.get_legend().remove()
    
    plt.tight_layout()
    
    if save:
        print(f'Saving plot to {args.output_path + "_distributions.png"}')
        plt.savefig(args.output_path + '_distributions.png')
    else:
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
    output_path = args.train.split('.')[0] + "_dataset_distributions_squeezed"
    args.output_path = output_path
    
    main(args, save=True)
    
    
    