"""2단계 · 시각화 — Seaborn 정적 1개↑ + Plotly 인터랙티브 1개↑ (제목·축 레이블 필수!)
담당: ______ / 브랜치: feat/viz

리턴 dict 키(변경 금지):
  seaborn_path : 저장된 정적 차트 경로 (str, outputs/ 아래)
  plotly_path  : 저장된 인터랙티브 HTML 경로 (str, outputs/ 아래)
  desc         : 두 차트가 보여주는 내용 요약 한 단락 (str)
"""

import logging


def run(df) -> dict:
    """시각화 전체 실행. 아래 TODO 순서대로 구현하세요."""

    # ── TODO 1. Seaborn 정적 차트 → outputs/seaborn_chart.png ──
    #   예시 주제: 성별(또는 소득그룹)에 따른 주당 근무시간 분포 boxplot
    #   ※ palette를 쓰려면 hue를 함께 지정할 것 (안 그러면 FutureWarning이 캡처에 찍힘):
    #     sns.boxplot(data=df, x='sex', y='hours-per-week', hue='sex', legend=False)
    #   ※ plt.title / plt.xlabel / plt.ylabel 필수 (채점 기준 명시)
    #   ※ 라벨은 영문 권장 — 한글 폰트 깨짐 회피. 한글 필요 시 폰트 설정 먼저.
    #   plt.tight_layout(); plt.savefig('outputs/seaborn_chart.png'); plt.close()

    # ── TODO 2. Plotly 인터랙티브 차트 → outputs/plotly_chart.html ──
    #   예시 주제: age vs hours-per-week 산점도, color='income'
    #   ※ 대량 산점도는 df.sample(1000, random_state=42)로 샘플링 (렌더 성능)
    #   ※ title=, labels= 로 제목·축 레이블 지정 필수
    #   fig.write_html('outputs/plotly_chart.html')

    # ── TODO 3. 리턴 ──
    #   return {"seaborn_path": "outputs/seaborn_chart.png",
    #           "plotly_path": "outputs/plotly_chart.html",
    #           "desc": "..."}

    raise NotImplementedError("feat/viz 브랜치에서 구현 후 이 줄을 삭제하세요")
