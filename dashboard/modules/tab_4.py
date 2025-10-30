# modules/tab_4.py
import streamlit as st
import pandas as pd
from shared import *

def show_tab_appendix(train):
    st.title("📘 부록 (Appendix)")
    st.write("이 탭에서는 참고자료, 용어 설명, 추가 통계 등을 제공합니다.")

    st.markdown("""
    ### 📄 포함 내용 예시
    - 데이터 컬럼 설명서
    - 단위 변환표
    - 전력 용어 정리
    - 참고 링크
    """)

    # 예시 테이블
    desc = pd.DataFrame({
        "컬럼명": [COL_TIME, COL_USAGE, COL_COST, COL_JOB, COL_DEMAND, COL_PF],
        "설명": [
            "측정된 일시",
            "해당 시점의 전력 사용량 (kWh)",
            "전기요금 (원)",
            "작업유형 (예: 주간/야간 등)",
            "수요전력 (kW)",
            "지상역률 (%)"
        ]
    })
    st.dataframe(desc, use_container_width=True, hide_index=True)

    st.markdown("#### 🔗 참고 링크")
    st.markdown("[📘 전력 수요 예측 개요 - KEPCO](https://home.kepco.co.kr/)")
