from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 事前に保存したモデルをロード
model = joblib.load("catboost_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 入力データを受け取る
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # 予測
        prediction = int(model.predict(features)[0])

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
