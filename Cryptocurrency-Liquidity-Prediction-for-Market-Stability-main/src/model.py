from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

def train_model(df, target_col):
    # Remove unnecessary columns
    X = df.drop(columns=[target_col, 'coin', 'symbol', 'date'])
    y = df[target_col]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("R² Score:", r2_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    # Save the trained model
    with open("model/liquidity_model.pkl", "wb") as f:
        pickle.dump(model, f)
