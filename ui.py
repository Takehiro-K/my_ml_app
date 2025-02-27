import streamlit as st
import requests

st.title("CatBoost 二項分類モデル")

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
    # APIにリクエストを送る
    response = requests.post(
        "http://127.0.0.1:5000/predict", 
        json={"features": [C2_7_angle, C4_5_affect_foramen, C4_5_not_affect_foramen, Duration, Female, MMT_Biceps]}
    )
    
    # 結果を表示
    if response.status_code == 200:
        st.success(f"予測結果: {response.json()['prediction']}")
    else:
        st.error("エラーが発生しました")
