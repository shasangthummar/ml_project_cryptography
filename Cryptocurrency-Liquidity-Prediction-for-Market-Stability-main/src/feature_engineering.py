import numpy as np

def add_features(df):
    # Moving average of price
    df['price_ma'] = df.groupby('coin')['price'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # Liquidity Ratio = 24h Volume / Market Cap
    df['liquidity_ratio'] = df['24h_volume'] / df['mkt_cap']

    # Replace infinity and NaN with 0
    df['liquidity_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['liquidity_ratio'].fillna(0, inplace=True)

    return df
