# Load merged_scaore_ringdata.csv and plot score distribution.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
# ------------------------
# Load Data
# ------------------------
def load_data(file_path):
    return pd.read_csv(file_path)
# ------------------------
# Main
# ------------------------
def main(args):
    # Load data
    df = load_data(args.data_path)
    print(df["stress_level"])
    # Plot total_score distribution
    
    
    counts = df['stress_level'].value_counts()
    
    print(counts)
    exit()
    # Put the counts as text on top of the plt
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title('Stress Level Distribution')
    plt.xlabel('Stress Level')
    plt.ylabel('Counts')
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 3,
                '{:1.0f}'.format(height), ha="center")
    plt.savefig('stress_level_distribution.png')

    print(counts)
    plt.show()
# ------------------------
# Argparse
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Score Distributions')
    parser.add_argument('--data_path', type=str, default='data/merged_score_ringdata.csv',
                        help='Path to the merged score and ringdata CSV file')
    args = parser.parse_args()
    main(args)