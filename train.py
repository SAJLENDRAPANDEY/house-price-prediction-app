"""
Training script for the King County House Price Prediction model.

Dataset: King County House Sales (kc_house_data.csv)
Download from: https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv

Usage:
    python train.py                          # auto-downloads dataset
    python train.py --data kc_house_data.csv # use local file
"""

import argparse
import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_URL = (
    "https://raw.githubusercontent.com/Shreyas3108/"
    "house-price-prediction/master/kc_house_data.csv"
)

# King County, WA — zipcode to city name mapping
ZIPCODE_TO_CITY = {
    98001: "Auburn",       98002: "Auburn",       98003: "Federal Way",
    98004: "Bellevue",     98005: "Bellevue",     98006: "Bellevue",
    98007: "Bellevue",     98008: "Bellevue",     98009: "Bellevue",
    98010: "Black Diamond", 98011: "Bothell",     98014: "Carnation",
    98019: "Duvall",       98022: "Enumclaw",     98023: "Federal Way",
    98024: "Fall City",    98027: "Issaquah",     98028: "Kenmore",
    98029: "Issaquah",     98030: "Kent",         98031: "Kent",
    98032: "Kent",         98033: "Kirkland",     98034: "Kirkland",
    98038: "Maple Valley", 98039: "Medina",       98040: "Mercer Island",
    98042: "Kent",         98045: "North Bend",   98047: "Pacific",
    98050: "Preston",      98051: "Ravensdale",   98052: "Redmond",
    98053: "Redmond",      98055: "Renton",       98056: "Renton",
    98057: "Renton",       98058: "Renton",       98059: "Renton",
    98065: "Snoqualmie",   98068: "Snoqualmie Pass", 98070: "Vashon",
    98072: "Woodinville",  98074: "Sammamish",    98075: "Sammamish",
    98077: "Woodinville",  98092: "Auburn",
    98102: "Seattle",      98103: "Seattle",      98105: "Seattle",
    98106: "Seattle",      98107: "Seattle",      98108: "Seattle",
    98109: "Seattle",      98112: "Seattle",      98115: "Seattle",
    98116: "Seattle",      98117: "Seattle",      98118: "Seattle",
    98119: "Seattle",      98121: "Seattle",      98122: "Seattle",
    98125: "Seattle",      98126: "Seattle",      98133: "Seattle",
    98136: "Seattle",      98144: "Seattle",      98146: "Burien",
    98148: "Burien",       98155: "Shoreline",    98158: "SeaTac",
    98166: "Burien",       98168: "Burien",       98177: "Shoreline",
    98178: "Seattle",      98188: "SeaTac",       98198: "Des Moines",
    98199: "Seattle",
}

NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "condition", "yr_built",
]


def download_dataset(path):
    print(f"Downloading dataset from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, path)
    print(f"Saved to {path}")


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=["price", "sqft_living", "bedrooms", "bathrooms"])
    df = df[(df["price"] > 0) & (df["sqft_living"] > 0)]
    df = df[(df["bedrooms"] > 0) & (df["bedrooms"] < 20)]

    # Map zipcode → city; unknown zips default to Seattle (largest city)
    df["city"] = df["zipcode"].map(ZIPCODE_TO_CITY).fillna("Seattle")

    # Remove price outliers (keep 5th–95th percentile ± 1.5 IQR)
    q_low = df["price"].quantile(0.05)
    q_high = df["price"].quantile(0.95)
    iqr = q_high - q_low
    df = df[(df["price"] >= q_low - 1.5 * iqr) & (df["price"] <= q_high + 1.5 * iqr)]

    return df


def build_features(df):
    city_dummies = pd.get_dummies(df["city"], prefix="city")
    X = pd.concat([df[NUMERIC_FEATURES], city_dummies], axis=1)
    y = df["price"]
    return X, y


def train(csv_path, output_dir="."):
    df = load_and_clean(csv_path)
    print(f"Clean dataset: {len(df):,} rows")
    print(f"Price range : ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")
    print(f"Cities      : {sorted(df['city'].unique())}")

    X, y = build_features(df)
    columns = list(X.columns)
    print(f"Features    : {len(columns)}  ({len([c for c in columns if c.startswith('city_')])} city OHE)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\nTraining GradientBoostingRegressor …")
    model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R²   (test) : {r2:.4f}")
    print(f"RMSE (test) : ${rmse:,.0f}")

    # Quick sanity-check predictions
    print("\n--- Sanity-check predictions ---")
    cases = [
        dict(bedrooms=3, bathrooms=2.0, sqft_living=1200, sqft_lot=5000,
             floors=1, waterfront=0, condition=3, yr_built=2000, city=None),
        dict(bedrooms=3, bathrooms=2.0, sqft_living=1200, sqft_lot=5000,
             floors=1, waterfront=0, condition=3, yr_built=2000, city="Seattle"),
        dict(bedrooms=3, bathrooms=2.0, sqft_living=2000, sqft_lot=5000,
             floors=2, waterfront=0, condition=4, yr_built=2005, city="Seattle"),
        dict(bedrooms=4, bathrooms=3.0, sqft_living=2500, sqft_lot=7000,
             floors=2, waterfront=0, condition=4, yr_built=2010, city="Bellevue"),
    ]
    for case in cases:
        city = case.pop("city")
        row = pd.DataFrame([[0] * len(columns)], columns=columns)
        for feat, val in case.items():
            row[feat] = val
        city_col = f"city_{city}" if city else None
        if city_col and city_col in row.columns:
            row[city_col] = 1
        pred = model.predict(scaler.transform(row))[0]
        print(f"  {case}  city={city!r:12s} → ${pred:,.0f}")

    # Persist artifacts
    joblib.dump(model,   os.path.join(output_dir, "model.pkl"))
    joblib.dump(scaler,  os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(columns, os.path.join(output_dir, "columns.pkl"))
    print(f"\nSaved model.pkl, scaler.pkl, columns.pkl → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train the house-price prediction model.")
    parser.add_argument("--data", default="kc_house_data.csv",
                        help="Path to the KC Houses CSV file (downloaded if missing).")
    parser.add_argument("--output", default=".",
                        help="Directory to save model artifacts (default: current directory).")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        download_dataset(args.data)

    train(args.data, output_dir=args.output)


if __name__ == "__main__":
    main()
