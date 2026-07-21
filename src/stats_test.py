"""3단계 · 통계 분석 — 기술통계 + 상관계수 + scipy.stats.ttest_ind + p-value 해석
담당: 박연주 / 브랜치: feat/stats  (팀 결정: docs/decisions.md 참조)

리턴 dict 키(변경 금지):
  describe_md / corr_md / group_label / mean_a / mean_b / t_stat / p_value / interp
"""

import logging

from scipy import stats

# ── 상관계수 대상: 수치형 5개 (0721 팀 결정 — fnlwgt 제외, education-num은 통계용으로 사용) ──
NUMERIC_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
TARGET_COL = "hours-per-week"   # t-test 비교 변수


def run(df) -> dict:
    """통계 분석 전체 실행 — 기술통계 → 상관 → 이상치 탐지(보고만) → t-test."""

    # ── 1. 컬럼 존재 검증 (②가정 방어: 없는 컬럼이면 즉시 명확한 에러) ──
    missing = set(NUMERIC_COLS + ["income"]) - set(df.columns)
    if missing:
        raise ValueError(f"필요 컬럼 누락: {sorted(missing)}")

    # ── 2. 기술통계 (평균·표준편차·분위수) ──
    describe_md = df[NUMERIC_COLS].describe().round(2).to_markdown()
    logging.info("기술통계 산출 완료 (평균·표준편차·분위수)")

    # ── 3. 상관계수 매트릭스 (수치형 변수 간) ──
    corr_md = df[NUMERIC_COLS].corr().round(3).to_markdown()
    logging.info("상관계수 계산 완료 (수치형 5개 변수)")

    # ── 4. 이상치 탐지 — 건수 '보고'만 하고 제거하지 않음 (팀 방침: docs/decisions.md) ──
    # age는 우측 편포, capital-gain 99999는 top-coding(구조적 특성)이라 IQR 기계 제거 금지
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        logging.info(f"[이상치 탐지] {col}: IQR 기준 {n_out:,}건 — 제거하지 않고 보존")

    # ── 5. 독립표본 t-test (Welch) ──
    # H0(귀무가설): 고소득(>50K)과 저소득(<=50K) 그룹의 주당 평균 근무시간에 차이가 없다
    # H1(대립가설): 두 그룹의 주당 평균 근무시간에 차이가 있다
    g_high = df.loc[df["income"] == ">50K", TARGET_COL]
    g_low = df.loc[df["income"] == "<=50K", TARGET_COL]
    if len(g_high) == 0 or len(g_low) == 0:   # 타겟 형식이 깨졌으면 즉시 중단
        raise ValueError("t-test 그룹이 비어 있음 — income 값 형식을 재확인할 것")

    t_stat, p_value = stats.ttest_ind(g_high, g_low, equal_var=False)  # Welch t-test
    mean_a, mean_b = float(g_high.mean()), float(g_low.mean())
    diff = mean_a - mean_b

    # ── 6. p-value 해석 (채점 기준: 귀무가설 기각 여부를 문장으로 명시) ──
    if p_value < 0.05:
        interp = (
            f"p-value({p_value:.3e}) < 0.05 이므로 귀무가설을 기각한다. "
            f"고소득(>50K) 그룹의 주당 근무시간 평균({mean_a:.2f}h)이 저소득 그룹({mean_b:.2f}h)보다 "
            f"약 {diff:.2f}시간 길며, 이 차이는 통계적으로 유의하다 (Welch t-test)."
        )
    else:
        interp = (
            f"p-value({p_value:.3e}) >= 0.05 이므로 귀무가설을 기각할 수 없다. "
            f"두 그룹의 주당 근무시간 차이({diff:.2f}h)는 통계적으로 유의하지 않다."
        )
    logging.info(f"[t-test] t={t_stat:.4f}, p={p_value:.3e} → 해석 완료")

    # ── 7. 리턴 (키 이름 고정 — report가 그대로 소비) ──
    return {
        "describe_md": describe_md,
        "corr_md": corr_md,
        "group_label": "고소득(>50K) vs 저소득(<=50K)의 주당 근무시간",
        "mean_a": mean_a,
        "mean_b": mean_b,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "interp": interp,
    }