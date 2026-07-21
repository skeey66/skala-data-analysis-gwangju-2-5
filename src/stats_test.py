"""3단계 · 통계 분석 — 기술통계 + 상관계수 + scipy.stats.ttest_ind + p-value 해석
담당: ______ / 브랜치: feat/stats

리턴 dict 키(변경 금지):
  describe_md : 기술통계(평균·표준편차·분위수) markdown 표 (str)
  corr_md     : 수치형 변수 상관계수 markdown 표 (str)
  group_label : t-test 비교 그룹 설명 예: "고소득(>50K) vs 저소득(<=50K)의 주당 근무시간" (str)
  mean_a      : 그룹 A 평균 (float)
  mean_b      : 그룹 B 평균 (float)
  t_stat      : t 통계량 (float)
  p_value     : p-value (float)
  interp      : p-value 해석 문장 — 귀무가설 기각 여부 명시 (str)
"""

import logging

NUMERIC_COLS = ["age", "fnlwgt", "education-num",
                "capital-gain", "capital-loss", "hours-per-week"]


def run(df) -> dict:
    """통계 분석 전체 실행. 아래 TODO 순서대로 구현하세요."""

    # ── TODO 1. 기술통계 → markdown 표 문자열 ──
    #   describe_md = df[NUMERIC_COLS].describe().round(2).to_markdown()

    # ── TODO 2. 상관계수 → markdown 표 문자열 ──
    #   corr_md = df[NUMERIC_COLS].corr().round(3).to_markdown()

    # ── TODO 3. t-test — 가설을 먼저 명시하고 검정 (채점 포인트: 해석 필수) ──
    #   H0(귀무): 두 그룹의 주당 평균 근무시간에 차이가 없다
    #   H1(대립): 두 그룹의 주당 평균 근무시간에 차이가 있다
    #   from scipy import stats
    #   g_a = df[df['income'] == '>50K']['hours-per-week']
    #   g_b = df[df['income'] == '<=50K']['hours-per-week']
    #   t_stat, p_value = stats.ttest_ind(g_a, g_b, equal_var=False)  # Welch
    #   ※ 그룹 평균(mean_a, mean_b)도 함께 리턴 — "몇 시간 차이"까지 말해야 해석이 완성됨

    # ── TODO 4. 해석 문장 (p<0.05 기준 귀무가설 기각 여부를 문장으로) ──
    #   interp = f"p-value({p_value:.3e}) < 0.05 이므로 귀무가설을 기각한다. ..."

    # ── TODO 5. 리턴 (키 이름 그대로!) ──

    raise NotImplementedError("feat/stats 브랜치에서 구현 후 이 줄을 삭제하세요")
