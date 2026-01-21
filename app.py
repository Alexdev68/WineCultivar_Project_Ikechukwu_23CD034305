import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model, scaler = joblib.load("model/wine_cultivar_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["alcalinity_of_ash"]),
            float(request.form["total_phenols"]),
            float(request.form["color_intensity"]),
            float(request.form["proline"])
        ]

        scaled = scaler.transform([features])
        result = model.predict(scaled)[0]
        prediction = f"Cultivar {result + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
