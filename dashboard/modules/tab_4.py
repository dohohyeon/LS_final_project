# modules/tab_4.py
import streamlit as st
import pandas as pd
from shared import * # 컬럼명 상수를 가져오기 위해 import

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

    # --- [수정] 예시 테이블 ---
    desc_data = {
        "컬럼명": [
            COL_TIME, COL_USAGE, COL_COST, COL_JOB, 
            COL_DEMAND, COL_LAG_PF, COL_LEAD_PF
        ],
        "설명": [
            "측정된 일시 (15분 간격)",
            "15분간 사용된 전력의 총량 (kWh)",
            "전기요금 (원) - 추정치 또는 실제",
            "작업유형 (예: Light_Load, Medium_Load 등)",
            "15분간의 평균 수요전력 (kW) [= 전력사용량(kWh) * 4]",
            "지상역률 (%). 90% 미만 시 요금 할증.",
            "진상역률 (%). 95% 미만 시 요금 할증."
        ]
    }
    
    # train 데이터에 실제 있는 컬럼만 필터링해서 보여주기
    valid_cols = [col for col in desc_data["컬럼명"] if col in train.columns]
    valid_indices = [desc_data["컬럼명"].index(col) for col in valid_cols]
    
    final_desc = {
        "컬럼명": [desc_data["컬럼명"][i] for i in valid_indices],
        "설명": [desc_data["설명"][i] for i in valid_indices]
    }
    
    st.dataframe(pd.DataFrame(final_desc), use_container_width=True, hide_index=True)

    st.markdown("#### 🔗 참고 링크")
    st.markdown("[📘 KEPCO (한국전력공사) - 전기요금표](https://cyber.kepco.co.kr/ckepco/front/jsp/CY/E/E/CYEEHP00101.jsp)")
