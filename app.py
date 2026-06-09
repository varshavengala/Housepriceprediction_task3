from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])

    input_data = pd.DataFrame(
        [[area, bedrooms]],
        columns=["Area", "Bedrooms"]
    )

    prediction = model.predict(input_data)

    result = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ₹{result}"
    )

if __name__ == "__main__":
    app.run(debug=True)