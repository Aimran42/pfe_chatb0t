from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
CORS(app)

# Charger et entraîner le modèle
df = pd.read_csv("dataset_phosphate.csv")
X = df.drop("Qualite_Phosphate", axis=1)
y = df["Qualite_Phosphate"]

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X, y)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[ 
        data["p2o5"],
        data["cao"],
        data["sio2"],
        data["mgo"],
        data["humidite"]
    ]])
    features = scaler.transform(features)
    pred = model.predict(features)
    result = le.inverse_transform(pred)[0]
    return jsonify({"qualite": result})

if __name__ == "__main__":
    app.run(debug=True)