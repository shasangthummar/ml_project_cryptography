from src.preprocessing import load_and_clean_data, scale_data
from src.feature_engineering import add_features

filepaths = ['data/coin_gecko_2022-03-16.csv', 'data/coin_gecko_2022-03-17.csv']

# Load and clean data
df = load_and_clean_data(filepaths)

# Scale selected columns
columns_to_scale = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap']
df = scale_data(df, columns_to_scale)

# Add engineered features
df = add_features(df)

# Show the new columns
print(df[['coin', 'price', 'price_ma', 'liquidity_ratio']].head())
from src.model import train_model

# Train the model to predict liquidity_ratio
train_model(df, target_col='liquidity_ratio')

