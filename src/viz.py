"""2단계 · 시각화 — Seaborn 정적 1개↑ + Plotly 인터랙티브 1개↑ (제목·축 레이블 필수!)
담당: 김단빈 / 브랜치: feat/viz

리턴 dict 키(변경 금지):
  seaborn_path : 저장된 정적 차트 경로 (str, outputs/ 아래)
  plotly_path  : 저장된 인터랙티브 HTML 경로 (str, outputs/ 아래)
  desc         : 두 차트가 보여주는 내용 요약 한 단락 (str)
"""

import logging

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


def run(df) -> dict:
    """시각화 전체 실행 — 컬럼 검증 → 정적 박스플롯 → 인터랙티브 산점도."""

    # ── 0. 컬럼 존재 검증 (②가정 방어: 없는 컬럼이면 즉시 명확한 에러) ──
    required = {"sex", "hours-per-week", "age", "income"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필요 컬럼 누락: {sorted(missing)}")

    # ── 1. Seaborn 정적 차트: 성별 주당 근무시간 분포 → outputs/seaborn_chart.png ──
    plt.figure(figsize=(10, 6))  # 도화지 크기 설정

    # 그래프 그리기 (hue 지정으로 palette 경고 회피)
    # order로 저소득→고소득 순서를 고정해 "고소득이 더 오래 일한다"가 한눈에 읽히게 함
    sns.boxplot(data=df, x='income', y='hours-per-week', hue='income',
                order=['<=50K', '>50K'], legend=False)

    # 제목 및 라벨 달기 (영문 — 한글 폰트 깨짐 회피)
    plt.title("Weekly Working Hours by Income Group")
    plt.xlabel("Income Group")
    plt.ylabel("Hours Per Week")

    # 저장 및 닫기
    plt.tight_layout()
    plt.savefig('outputs/seaborn_chart.png')
    plt.close()
    logging.info("Seaborn 박스플롯 저장 완료: outputs/seaborn_chart.png")

    # ── 2. Plotly 인터랙티브 차트: 나이-근무시간 산점도(소득 그룹 색상) ──
    # 렌더링 성능을 위해 표본 추출 (행 수가 1,000 미만이어도 안전하게)
    n_sample = min(1000, len(df))
    df_sampled = df.sample(n_sample, random_state=42)

    fig = px.scatter(
        df_sampled,
        x='age',
        y='hours-per-week',
        color='income',
        title="Age vs Hours Per Week by Income",
        labels={
            'age': 'Age',
            'hours-per-week': 'Hours Per Week',
            'income': 'Income Group'
        }
    )
    fig.write_html('outputs/plotly_chart.html')
    logging.info(f"Plotly 산점도 저장 완료: outputs/plotly_chart.html (표본 {n_sample:,}건)")

    # ── 3. 리턴 (키 이름 고정 — report가 그대로 소비) ──
    desc = (
        "박스플롯은 소득 그룹별 주당 근무시간 분포를 비교한 것으로, 고소득(>50K) 그룹의 "
        "중앙값과 사분위 범위가 저소득 그룹보다 뚜렷하게 위에 있어, 통계 파트의 t-test 결과"
        "(고소득 그룹이 주당 약 6.4시간 더 근무, p<0.05로 유의)를 시각적으로 그대로 보여준다. "
        "산점도(무작위 1,000건 표본)는 나이-근무시간 평면에서 소득 그룹(>50K vs <=50K)을 색으로 "
        "구분한 것으로, 고소득 그룹이 30~50대·주 40시간 이상 구간에 상대적으로 밀집하는 경향을 보여 "
        "변수 중요도 결과(age 1위, hours-per-week 3위)와도 일관된다."
    )
    return {
        "seaborn_path": "outputs/seaborn_chart.png",
        "plotly_path": "outputs/plotly_chart.html",
        "desc": desc,
    }