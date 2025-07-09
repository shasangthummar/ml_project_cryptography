import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model/liquidity_model.pkl", "rb"))

st.title("🔮 Crypto Liquidity Predictor")

# Input fields
price = st.number_input("Price", value=1.0)
h1 = st.number_input("1h Change", value=0.0)
h24 = st.number_input("24h Change", value=0.0)
d7 = st.number_input("7d Change", value=0.0)
volume = st.number_input("24h Volume", value=1.0)
cap = st.number_input("Market Cap", value=1.0)
price_ma = st.number_input("3-day Avg Price", value=1.0)

if st.button("Predict Liquidity"):
    input_df = pd.DataFrame([[price, h1, h24, d7, volume, cap, price_ma]],
                            columns=['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap', 'price_ma'])

    prediction = model.predict(input_df)[0]
    st.success(f"💧 Predicted Liquidity Ratio: {prediction:.4f}")
