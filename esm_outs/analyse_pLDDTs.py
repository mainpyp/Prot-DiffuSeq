import pandas as pd
import matplotlib.pyplot as plt
import os

# set this folder as root 
os.chdir("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/esm_outs")
plt.rcParams.update({'font.size': 22})


def analyze_plddt(file_list: list) -> None:
    """ This function takes a list of paths containing csv files in the same format [id, pLDDT] and 
    first creates one big dataframe containing all the scores per index.
    It then plots the pLDDT distribution of the files in the same plot as a bar plot with the bars being next 
    to another.
    """
    # create a list of dataframes
    names = list(map(lambda x: x.split("_")[1].replace(".csv", ""), file_list))
    df_list = []
    for name, file in zip(names, file_list):
        df = pd.read_csv(file, header=None)
        df.columns = ["id", name]
        df_list.append(df)

    # merge the dataframes
    
    merged_df = pd.concat(df_list, axis=1, join="inner")
    # drop duplicated columns
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    # removes first row
    merged_df = merged_df.iloc[1:]
    # converts every column to numeric except for the id column
    merged_df = merged_df.apply(pd.to_numeric, errors="ignore")
    print(merged_df)
    # plot the dataframes
    merged_df.plot("id", names, bins=40, kind="hist", figsize=(20, 10), alpha=0.5)
    # add the mean of each column as a vertical line
    for i, name in enumerate(names):
        plt.axvline(merged_df[name].mean(), color="k", linestyle="dashed", linewidth=1)
        # adds value of mean as text and name
        plt.text(merged_df[name].mean() + 0.1, 8 + 8* (i / 10), f"{name}: {merged_df[name].mean():.2f}")
    
    

    # adds title
    plt.title("pLDDT distribution of different models")
    # adds x axis label
    plt.xlabel("pLDDT score")
    # increases font size globally
    
    plt.show()


if __name__ == "__main__":
    files = ["esmfold_100recpLDDT.csv", "esmfold_130recpLDDT.csv", "esmfold_refpLDDT.csv"]
    analyze_plddt(files)