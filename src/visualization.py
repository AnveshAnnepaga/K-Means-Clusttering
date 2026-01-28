import matplotlib.pyplot as plt
import pandas as pd

def plot_clusters(df, centers):
    plt.figure(figsize=(8,6))
    for c in df['Cluster'].unique():
        subset = df[df['Cluster'] == c]
        plt.scatter(subset['Grocery'], subset['Milk'], label=f'Cluster {c}')

    plt.scatter(
        centers[:,2], centers[:,1],
        s=300, c='black', marker='X', label='Centers'
    )
    plt.xlabel("Grocery Spend")
    plt.ylabel("Milk Spend")
    plt.legend()
    return plt