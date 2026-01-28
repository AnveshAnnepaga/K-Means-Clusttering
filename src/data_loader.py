import pandas as pd

def load_data(file_path):
    """
    Loads the wholesale customer dataset and returns a DataFrame
    """
    df = pd.read_csv(file_path)
    return df
