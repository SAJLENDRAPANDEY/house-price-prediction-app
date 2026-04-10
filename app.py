from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder=".")

# Load model files
model   = joblib.load("model.pkl")
scaler  = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Serve UI
@app.route("/")
def index():
    return app.send_static_file("house_predictor.html")  # ✅ better than send_from_directory

# Predict API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Create empty input
        new_house = pd.DataFrame([[0] * len(columns)], columns=columns)

        # Fill values — use realistic defaults for optional fields so that
        # omitted inputs don't produce extreme out-of-range predictions.
        new_house["bedrooms"]    = int(data.get("bedrooms", 3))
        new_house["bathrooms"]   = float(data.get("bathrooms", 2.0))
        new_house["sqft_living"] = int(data.get("sqft_living", 1800))
        new_house["sqft_lot"]    = int(data.get("sqft_lot", 5000))
        new_house["floors"]      = int(data.get("floors", 1))
        new_house["waterfront"]  = int(data.get("waterfront", 0))
        new_house["condition"]   = int(data.get("condition", 3))
        new_house["yr_built"]    = int(data.get("yr_built", 1990))

        # City encoding (safe)
        city = str(data.get("city", "")).strip()
        city_col = "city_" + city

        if city_col in new_house.columns:
            new_house[city_col] = 1

        # Scale & predict
        scaled = scaler.transform(new_house)
        price  = model.predict(scaled)[0]

        return jsonify({
            "success": True,
            "price": round(float(price), 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)