import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter


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
    return df

def plot_length_distribution(df:pd.DataFrame, save_path:str=None):
    fig = df.src.apply(len).plot(kind='hist', bins=100, colormap='viridis', alpha=0.8)
    fig.set_xlabel('Length of source sequence')
    fig.set_title('Distribution of sequence lengths')

    if save_path:
        fig.figure.savefig(save_path)
    else:
        plt.show()



if __name__ == '__main__':
    d = load_data('datasets/ProtTotalCorrect/train.jsonl')
    plot_length_distribution(df=d, save_path="datasets/plots/length_distribution_ProtTotalCorrect.png")
    