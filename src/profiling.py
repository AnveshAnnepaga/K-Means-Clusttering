def profile_clusters(df):
    return df.groupby('Cluster').mean()