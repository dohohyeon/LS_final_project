import streamlit as st
from shared import load_train
from modules.tab_1 import show_tab_realtime
from modules.tab_2 import show_tab_analysis
from modules.tab_3 import show_tab_prediction
from modules.tab_4 import show_tab_appendix  # ✅ 부록 탭 추가

st.set_page_config(page_title="전력 모니터링 대시보드", layout="wide")

train = load_train()
if train.empty:
    st.stop()

# ✅ 탭 순서 지정 (홈 → 실시간 → 통계 → 부록)
tab_pred, tab_rt, tab_viz, tab_appendix = st.tabs([
    " HOME",
    " 실시간 모니터링",
    " 통계 분석",
    " 부록"
])

# ✅ 탭별 연결
with tab_pred:
    show_tab_prediction(train)

with tab_rt:
    show_tab_realtime(train)

with tab_viz:
    show_tab_analysis(train)

with tab_appendix:
    show_tab_appendix(train)
