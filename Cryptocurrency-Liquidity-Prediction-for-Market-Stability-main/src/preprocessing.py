import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(filepaths):
    dfs = [pd.read_csv(fp) for fp in filepaths]
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

def scale_data(df, columns_to_scale):
    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df
