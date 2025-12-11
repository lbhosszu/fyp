import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "../ml_outputs/ml_features.csv"
MODEL_OUT = "../ml_outputs/rf_model.pkl"
PLOT_OUT = "../ml_outputs/feature_importance.png"


# Load and preprocess dataset
def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"ML dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Drop rows with missing target variable
    df = df.dropna(subset=["resultposition"])

    return df


def encode_categorical(df):
    """One-hot encode team, driver_code, and location."""
    categorical_cols = ["team", "driver_code", "location"]

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = enc.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out(categorical_cols))

    # Drop old columns and merge new encoded columns
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    return df, enc


# Train Random Forest model 
def train_model(df):
    X = df.drop(columns=["resultposition", "driver_id"])  # drop target + ID columns   
    y = df["resultposition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Evaluation metrics 
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("=== Random Forest Results ===")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")

    return model, X_train.columns.tolist()


# Plot feature importances 
def save_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_n = 20  # only plot most important features 
    plt.figure(figsize=(10, 6))
    plt.title("Top Feature Importances (Random Forest)")
    plt.bar(range(top_n), importances[indices][:top_n])
    plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=90)
    plt.tight_layout()

    os.makedirs(os.path.dirname(PLOT_OUT), exist_ok=True)
    plt.savefig(PLOT_OUT, dpi=200)
    print(f"Saved feature importance plot: {PLOT_OUT}")


def main():
    print("Loading dataset...")
    df = load_dataset()

    print("Encoding categorical columns...")
    df_encoded, encoder = encode_categorical(df)

    print("Training Random Forest model...")
    model, feature_names = train_model(df_encoded)

    print("Saving model...")
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"Model saved to: {MODEL_OUT}")

    print("Saving feature importance plot...")
    save_feature_importance(model, feature_names)

    print("Done.")


if __name__ == "__main__":
    main()
