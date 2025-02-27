import streamlit as st
import joblib
import numpy as np
import os

# モデルのロード（フルパスを取得）
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "catboost_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ユーザーが入力するためのスライダーや選択ボックス
C2_7_angle = st.number_input("C2-7 angle", value=0.0)
C4_5_affect_foramen = st.number_input("C4-5 affect foramen", value=0.0)
C4_5_not_affect_foramen = st.number_input("C4-5 not affect foramen", value=0.0)
Duration = st.number_input("Duration (months)", value=0.0)
Female = st.radio("Gender", ["Male", "Female"])
Female = 1 if Female == "Female" else 0  # Female: 1, Male: 0
MMT_Biceps = st.number_input("MMT Biceps", value=0.0)

# 予測ボタン
if st.button("予測を実行"):
    # 予測を実行
    features = np.array([[C2_7_angle, C4_5_affect_foramen, C4_5_not_affect_foramen, Duration, Female, MMT_Biceps]])
    prediction = int(model.predict(features)[0])

    # 結果を表示
    st.success(f"予測結果: {prediction}")
