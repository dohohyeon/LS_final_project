# ---------------------------------------------------
# 전력 데이터 통합 분석 보고서 생성 (tab_2 동기화 완성 버전)
# ---------------------------------------------------

import os, io
import pandas as pd
import numpy as np
from datetime import datetime
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ===== ✅ 한글 폰트 강제 등록 (Kaleido 인식용) =====
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
font_path = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic-Regular.ttf")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Malgun Gothic"

# ===== Plotly 폰트 설정 =====
pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.font.family = "Malgun Gothic"
pio.defaults.font = dict(family="Malgun Gothic", size=12, color="#222")

# ============ 공통 표 스타일 ============
def format_table_uniform(table):
    table.style = "Table Grid"
    for ridx, row in enumerate(table.rows):
        for cell in row.cells:
            cell._tc.get_or_add_tcPr().append(
                parse_xml(r'<w:shd {} w:fill="FFFFFF"/>'.format(nsdecls('w')))
            )
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.name = "Malgun Gothic"
                    r.font.size = Pt(10)
                    r.font.color.rgb = RGBColor(0, 0, 0)
                    if ridx == 0:
                        r.font.bold = True


# ============ 보고서 생성 함수 ============
def generate_analysis_report(df, filtered_df, output_path="./reports/tab2_report.docx"):
    df = df.copy()
    filtered_df = filtered_df.copy()

    # 타입 보정
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    filtered_df["측정일시"] = pd.to_datetime(filtered_df["측정일시"], errors="coerce")

    if "시간" not in filtered_df.columns:
        filtered_df["시간"] = filtered_df["측정일시"].dt.hour

    for col in ["전력사용량(kWh)", "전기요금(원)", "수요전력(kW)", "지상역률(%)", "진상역률(%)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = Document()
    doc.styles["Normal"].font.name = "Malgun Gothic"

    # 페이지 테두리
    sectPr = doc.sections[0]._sectPr
    sectPr.append(parse_xml('''
        <w:pgBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:offsetFrom="page">
            <w:top w:val="single" w:sz="12" w:space="24" w:color="auto"/>
            <w:left w:val="single" w:sz="12" w:space="24" w:color="auto"/>
            <w:bottom w:val="single" w:sz="12" w:space="24" w:color="auto"/>
            <w:right w:val="single" w:sz="12" w:space="24" w:color="auto"/>
        </w:pgBorders>
    '''))

    # 0) 제목
    p = doc.add_paragraph("전력 데이터 통합 분석 보고서")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].font.size = Pt(18)
    p.runs[0].bold = True

    sd, ed = filtered_df["측정일시"].min(), filtered_df["측정일시"].max()
    doc.add_paragraph(f"분석 기간: {sd:%Y-%m-%d} ~ {ed:%Y-%m-%d}").alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"생성일: {datetime.now():%Y-%m-%d %H:%M}  |  작성자: 관리자").alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # 1️⃣ 주요 지표
    doc.add_heading("1. 주요 지표 요약", level=1)
    total_usage = filtered_df["전력사용량(kWh)"].sum()
    total_cost = filtered_df["전기요금(원)"].sum()
    avg_price = (total_cost / total_usage) if total_usage > 0 else 0
    carbon = total_usage * 0.000331

    cm = filtered_df["측정일시"].dt.month.mode()[0]
    pm = cm - 1 if cm > 1 else 12
    prev_df = df[df["측정일시"].dt.month == pm]

    def pct(curr, prev):
        return "-" if (prev == 0 or pd.isna(prev)) else f"{(curr - prev) / prev * 100:+.1f}%"

    if not prev_df.empty:
        pu = prev_df["전력사용량(kWh)"].sum()
        pc = prev_df["전기요금(원)"].sum()
        pp = (pc / pu) if pu > 0 else 0
        pco = pu * 0.000331
    else:
        pu = pc = pp = pco = np.nan

    t = doc.add_table(rows=2, cols=4)
    headers = ["전력사용량(kWh)", "전기요금(원)", "평균단가(원/kWh)", "탄소배출량(tCO₂)"]
    vals = [f"{total_usage:,.1f}", f"{total_cost:,.0f}", f"{avg_price:,.2f}", f"{carbon:,.3f}"]
    changes = [pct(total_usage, pu), pct(total_cost, pc), pct(avg_price, pp), pct(carbon, pco)]
    for i, h in enumerate(headers):
        t.cell(0, i).text = h
        t.cell(1, i).text = f"{vals[i]}\n({changes[i]})"
    format_table_uniform(t)
    doc.add_paragraph(f"{cm}월 주요 지표 및 전월({pm}월) 대비 증감률을 나타냅니다.")
    doc.add_paragraph(
        "전반적으로 선택한 기간 동안의 전력 사용량과 요금 수준을 요약합니다. "
        "전월 대비 증감률을 함께 제시하여 에너지 사용 추세를 파악할 수 있습니다. "
        "평균 단가의 상승은 피크 부하 시간대 사용량 증가 또는 요금제 변경의 영향을 시사할 수 있습니다."
    )
    doc.add_paragraph()

    # 2️⃣ 요일·시간대별 패턴
    doc.add_heading("2. 요일·시간대별 전력 사용 패턴", level=1)
    weekday_map = {"Monday": "월요일", "Tuesday": "화요일", "Wednesday": "수요일",
                   "Thursday": "목요일", "Friday": "금요일", "Saturday": "토요일", "Sunday": "일요일"}
    filtered_df["요일"] = filtered_df["측정일시"].dt.day_name().map(weekday_map)

    wd = (filtered_df.groupby("요일")["전력사용량(kWh)"].mean()
          .reindex(["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]).reset_index())
    hr = (filtered_df.groupby("시간")["전력사용량(kWh)"].mean().reset_index().sort_values("시간"))

    fig_pattern = make_subplots(rows=2, cols=1,
                                subplot_titles=("요일별 평균 전력사용량", "시간대별 평균 전력사용량"),
                                vertical_spacing=0.18)
    fig_pattern.add_bar(x=wd["요일"], y=wd["전력사용량(kWh)"], marker_color="#3A86FF", row=1, col=1)
    fig_pattern.add_trace(
        go.Scatter(x=hr["시간"].values, y=hr["전력사용량(kWh)"].values, mode="lines+markers",
                   line=dict(width=2), marker=dict(size=6)),
        row=2, col=1
    )
    fig_pattern.update_layout(showlegend=False, hovermode="x unified", template="plotly_white",
                              height=700, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
    buf = io.BytesIO()
    fig_pattern.write_image(buf, format="png")
    buf.seek(0)
    doc.add_picture(buf, width=Inches(6))
    doc.add_paragraph(
        "요일별 패턴을 보면 평일과 주말 간 전력 사용량의 차이를 확인할 수 있습니다. "
        "시간대별 그래프에서는 주간과 야간의 사용 패턴이 명확하게 구분되며, "
        "특정 시간대(예: 오전 9시~11시, 오후 2시~5시)에 전력 수요가 집중되는 경향이 나타납니다. "
        "이는 생산 공정 또는 설비 가동 스케줄과 밀접한 관련이 있습니다."
    )
    doc.add_paragraph()

    # 3️⃣ 피크 수요 및 역률
    doc.add_heading("3. 피크 수요 및 역률 분석", level=1)
    peak_kw = filtered_df["수요전력(kW)"].max() if "수요전력(kW)" in filtered_df else np.nan
    if "수요전력(kW)" in filtered_df and not np.isnan(peak_kw):
        pidx = filtered_df["수요전력(kW)"].idxmax()
        peak_time_str = filtered_df.loc[pidx, "측정일시"].strftime("%Y-%m-%d %H:%M")
    else:
        peak_time_str = "-"

    avg_lag = filtered_df["지상역률(%)"].mean() if "지상역률(%)" in filtered_df else np.nan
    avg_lead = filtered_df["진상역률(%)"].mean() if "진상역률(%)" in filtered_df else np.nan
    mean_pf = np.nanmean([avg_lag, avg_lead]) if not (np.isnan(avg_lag) and np.isnan(avg_lead)) else np.nan

    tp = doc.add_table(rows=2, cols=3)
    for i, (h, v) in enumerate([
        ("피크전력(kW)", f"{peak_kw:.1f}" if not np.isnan(peak_kw) else "-"),
        ("피크발생시각", peak_time_str),
        ("평균역률(%)", f"{mean_pf:.1f}" if not np.isnan(mean_pf) else "-"),
    ]):
        tp.cell(0, i).text, tp.cell(1, i).text = h, v
    format_table_uniform(tp)
    doc.add_paragraph()

    if "수요전력(kW)" in filtered_df.columns:
        dfp = (filtered_df[["측정일시", "수요전력(kW)"]]
               .dropna().drop_duplicates(subset=["측정일시"]).sort_values("측정일시"))
        xvals = dfp["측정일시"].dt.to_pydatetime()
        fig_peak = go.Figure()
        fig_peak.add_trace(go.Scatter(x=xvals, y=dfp["수요전력(kW)"], mode="lines",
                                      line=dict(width=2, color="#0066CC")))
        top3 = dfp.nlargest(3, "수요전력(kW)")
        fig_peak.add_trace(go.Scatter(x=top3["측정일시"].dt.to_pydatetime(),
                                      y=top3["수요전력(kW)"],
                                      mode="markers+text",
                                      text=[f"피크{i+1}" for i in range(len(top3))],
                                      textposition="top center",
                                      marker=dict(size=10, color="red"),
                                      showlegend=False))
        fig_peak.update_layout(title="기간 내 전력 사용량 추이 (상위 피크 3개 강조)",
                               template="plotly_white", hovermode="x unified",
                               showlegend=False, plot_bgcolor="#fff", paper_bgcolor="#fff")
        fig_peak.update_xaxes(type="date", tickformat="%m-%d %H:%M", tickangle=45)
        buf = io.BytesIO()
        fig_peak.write_image(buf, format="png")
        buf.seek(0)
        doc.add_picture(buf, width=Inches(6))
    doc.add_paragraph(
        "피크 수요 시간은 설비 가동률이 가장 높거나, 냉난방·조명 등의 부하가 동시에 작동할 때 발생합니다. "
        "이 시간대를 중심으로 부하 분산이나 피크 억제 대책을 검토하는 것이 중요합니다. "
        "역률 평균이 95% 이하로 나타나는 경우, 역률 보상장치(콘덴서 등) 점검을 권장합니다."
    )
    doc.add_paragraph()

    # 4️⃣ 작업유형별 전기요금
    doc.add_heading("4. 시간대별 작업유형별 전기요금 현황", level=1)
    if "작업유형" in filtered_df.columns:
        g = (filtered_df.groupby(["시간", "작업유형"])["전기요금(원)"].sum().reset_index())
        color_map = {"Light_Load": "#4CAF50", "Medium_Load": "#FF9800", "Maximum_Load": "#F44336"}
        fig_cost = px.bar(g, x="시간", y="전기요금(원)", color="작업유형",
                          title="시간대별 작업유형별 전기요금 현황 (누적 막대)",
                          labels={"전기요금(원)": "전기요금(원)", "시간": "시간대"},
                          color_discrete_map=color_map, text_auto=".2s")
        fig_cost.update_layout(barmode="stack", template="plotly_white", hovermode="x unified",
                               plot_bgcolor="#fff", paper_bgcolor="#fff",
                               legend_title="작업유형", xaxis=dict(dtick=1), height=500)
        buf = io.BytesIO()
        fig_cost.write_image(buf, format="png")
        buf.seek(0)
        doc.add_picture(buf, width=Inches(6))
    doc.add_paragraph(
        "작업 유형별로 전기요금이 집중되는 시간대를 확인할 수 있습니다. "
        "‘Maximum_Load’ 구간이 특정 시간대에 편중되어 있다면, "
        "생산 일정 조정이나 부하 분산을 통해 요금 절감 효과를 기대할 수 있습니다. "
        "반대로 ‘Light_Load’ 시간대가 장시간 지속되면 설비 효율 저하의 가능성도 있습니다."
    )
    doc.add_paragraph()

    # 5️⃣ 시계열 추이
    doc.add_heading("5. 전력사용량 시계열 추이", level=1)
    ts = (filtered_df[["측정일시", "전력사용량(kWh)"]]
          .dropna().drop_duplicates(subset=["측정일시"]).sort_values("측정일시"))
    xvals = ts["측정일시"].dt.to_pydatetime()
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=xvals, y=ts["전력사용량(kWh)"],
                                mode="lines", line=dict(width=1.8, color="#007bff")))
    fig_ts.update_layout(title="전력사용량 시계열 추이", template="plotly_white",
                         hovermode="x unified", plot_bgcolor="#fff", paper_bgcolor="#fff")
    fig_ts.update_xaxes(type="date", tickformat="%m-%d", dtick=86400000*3,
                        tickangle=45, showgrid=True, gridcolor="#eee")
    buf = io.BytesIO()
    fig_ts.write_image(buf, format="png")
    buf.seek(0)
    doc.add_picture(buf, width=Inches(6))
    doc.add_paragraph(
        "시계열 추이를 통해 전력 사용의 계절적 변동이나 공정별 특이 패턴을 파악할 수 있습니다. "
        "특정 기간에 급격한 사용량 증가는 설비 점검, 신규 라인 가동, 또는 계절적 요인에 의한 것일 수 있습니다. "
        "이를 바탕으로 향후 수요 예측 및 전력 계약전력 조정에 활용할 수 있습니다."
    )
    doc.add_paragraph()

    # 6️⃣ 개선사항
    doc.add_heading("6. 개선사항 및 제안", level=1)
    t2 = doc.add_table(rows=1, cols=2)
    t2.rows[0].cells[0].text, t2.rows[0].cells[1].text = "개선 항목", "내용"
    for k, v in [
        ("피크 부하 관리", "피크 시간대 설비 분산 및 가동 최적화"),
        ("역률 개선", "역률 보상장치 점검 및 유지보수"),
        ("에너지 절감", "야간 불필요 설비 자동 차단 시스템 도입"),
        ("요금 최적화", "시간대별 요금제 및 계약전력 검토"),
    ]:
        r = t2.add_row().cells
        r[0].text, r[1].text = k, v
    format_table_uniform(t2)
    doc.add_paragraph(
        "위의 분석을 종합하면, 피크 부하 시간대의 관리와 역률 개선이 가장 주요 과제입니다. "
        "또한 시간대별 요금제 선택과 설비 효율화는 에너지 비용 절감에 직접적인 영향을 줄 수 있습니다. "
        "지속적인 모니터링 체계를 구축하면 월별 변화 추이를 정량적으로 평가할 수 있습니다."
    )

    doc.save(output_path)
    return output_path
