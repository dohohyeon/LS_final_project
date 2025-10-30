# app.py
import streamlit as st
from shared import load_train
from modules.tab_1 import show_tab_realtime
from modules.tab_2 import show_tab_analysis

st.set_page_config(page_title="전력 모니터링 대시보드", layout="wide")

train = load_train()
if train.empty:
    st.stop()

tab_rt, tab_viz = st.tabs(["📡 실시간 모니터링", "📈 통계 분석"])

with tab_rt:
    show_tab_realtime(train)

with tab_viz:
    show_tab_analysis(train)
