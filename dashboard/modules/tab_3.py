# modules/tab_3.py
import streamlit as st
import pandas as pd
import plotly.express as px
from shared import *

def show_tab_prediction(train):
    st.title("📊 전력 예측 결과")
    st.write("AI 모델을 기반으로 향후 전력 수요를 예측한 결과를 보여줍니다.")

    # 예시: 하루 단위 평균 전력사용량
    df = train.copy()
    df["date"] = df[COL_TIME].dt.date
    daily_avg = df.groupby("date", as_index=False)[COL_USAGE].mean()
    daily_avg["예측값(kWh)"] = daily_avg[COL_USAGE] * 1.05  # 단순 예측 예시

    fig = px.line(
        daily_avg,
        x="date",
        y=["전력사용량(kWh)", "예측값(kWh)"],
        title="실제 vs 예측 전력 사용량 추이",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(daily_avg.tail(10), use_container_width=True)
