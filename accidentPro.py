import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import streamlit as st

# 컬럼 이름 설정
time_col = "Time (s)"
x_col = "Linear Acceleration x (m/s^2)"
y_col = "Linear Acceleration y (m/s^2)"
z_col = "Linear Acceleration z (m/s^2)"
abs_col = "Absolute acceleration (m/s^2)"

st.write("현재 작업 디렉터리:", os.getcwd())
st.write("현재 파일 목록:", os.listdir('.'))
st.write("walking 폴더 내 파일 목록:", os.listdir('walking'))

# 위험 탐지 함수
def detect_risk_spikes(df, abs_threshold=20, axis_threshold=15):
    abs_risk = df[abs_col] > abs_threshold
    x_risk = df[x_col].abs() > axis_threshold
    y_risk = df[y_col].abs() > axis_threshold
    z_risk = df[z_col].abs() > axis_threshold

    df["risk_flag"] = abs_risk | x_risk | y_risk | z_risk
    return df


# Streamlit 앱 시작
st.title("운동 데이터 분류 및 위험 상태 탐지")


# 1. 학습용 데이터 로드 (로컬 csv파일 경로 수정 필요)
@st.cache_data
def load_training_data():
    walking_file = "walk\\walk.csv.csv"
    running_file = "run\\run.csv.csv"
    stairs_file = "stair\\stair.csv.csv"

    df_walking = pd.read_csv(walking_file)
    df_running = pd.read_csv(running_file)
    df_stairs = pd.read_csv(stairs_file)

    df_walking["label"] = 0
    df_running["label"] = 1
    df_stairs["label"] = 2

    df_all = pd.concat([df_walking, df_running, df_stairs], ignore_index=True)
    return df_all


df_all = load_training_data()


# 2. 모델 학습 (캐시로 속도 향상)
@st.cache_data
def train_model(df):
    X = df[[x_col, y_col, z_col, abs_col]]
    y = df["label"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


model = train_model(df_all)

st.markdown("### 새로운 운동 데이터 CSV 파일 업로드")
uploaded_file = st.file_uploader("파일 선택 (CSV)", type=["csv"])

if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터 미리보기:", new_df.head())

    # 위험 탐지 실행
    new_df = detect_risk_spikes(new_df)

    # 위험 샘플 보여주기
    risk_samples = new_df[new_df["risk_flag"]]
    if len(risk_samples) > 0:
        st.warning(f"⚠️ 위험 상태가 감지된 샘플이 {len(risk_samples)}개 있습니다!")
        st.dataframe(risk_samples[[time_col, x_col, y_col, z_col, abs_col, "risk_flag"]].head(10))
    else:
        st.success("✅ 위험 상태가 감지되지 않았습니다.")

    # 모델 예측 실행 및 결과 표시
    X_new = new_df[[x_col, y_col, z_col, abs_col]]
    pred_labels = model.predict(X_new)

    label_map = {0: "걷기", 1: "달리기", 2: "계단"}
    pred_names = [label_map[label] for label in pred_labels]

    st.markdown("### 예측 결과 (처음 10개 샘플)")
    result_df = new_df.copy()
    result_df["predicted_label"] = pred_names
    st.dataframe(result_df[[time_col, x_col, y_col, z_col, abs_col, "predicted_label"]].head(10))
