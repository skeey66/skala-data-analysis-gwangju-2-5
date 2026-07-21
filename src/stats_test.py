"""3단계 · 통계 분석 — 기술통계 → 상관계수 → IQR 탐지 → Welch t-test.

리턴 키(변경 금지):
  describe_md / corr_md / group_label / mean_a / mean_b / t_stat / p_value / interp
"""

import logging

import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy import stats


NUMERIC_COLS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
GROUP_COL = "income"
TARGET_COL = "hours-per-week"
HIGH_INCOME = ">50K"
LOW_INCOME = "<=50K"
ALPHA = 0.05


def _validate_input(df) -> None:
    """분석에 필요한 데이터 구조와 값을 실행 전에 검증한다."""
    if df is None or not hasattr(df, "columns"):
        raise TypeError("Pandas DataFrame을 전달해야 합니다.")
    if df.empty:
        raise ValueError("분석할 데이터가 비어 있습니다.")

    required = NUMERIC_COLS + [GROUP_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필요 컬럼 누락: {missing}")

    non_numeric = [col for col in NUMERIC_COLS if not is_numeric_dtype(df[col])]
    if non_numeric:
        raise TypeError(f"수치형이어야 하는 컬럼: {non_numeric}")

    null_cols = [col for col in required if df[col].isna().any()]
    if null_cols:
        raise ValueError(f"결측값이 포함된 컬럼: {null_cols}")

    # 열 단위로 검사해 전체 크기의 임시 Boolean DataFrame 생성을 피한다.
    infinite_cols = [
        col
        for col in NUMERIC_COLS
        if not np.isfinite(df[col].to_numpy(copy=False)).all()
    ]
    if infinite_cols:
        raise ValueError(f"무한대 값이 포함된 컬럼: {infinite_cols}")

    labels = set(df[GROUP_COL].unique())
    expected = {HIGH_INCOME, LOW_INCOME}
    unexpected = labels - expected
    if unexpected:
        raise ValueError(f"income에 예상하지 못한 값이 있습니다: {sorted(unexpected)}")

    missing_groups = expected - labels
    if missing_groups:
        raise ValueError(f"t-test 비교 그룹이 없습니다: {sorted(missing_groups)}")


def _format_p_value(p_value: float) -> str:
    """극도로 작은 p-value가 정확히 0으로 오해되지 않게 표시한다."""
    return "< 1e-300" if p_value == 0 else f"{p_value:.3e}"


def _describe_difference(mean_high: float, mean_low: float) -> str:
    """두 그룹 평균 차이의 크기와 방향을 자연어로 표현한다."""
    difference = abs(mean_high - mean_low)
    if mean_high > mean_low:
        direction = "길다"
    elif mean_high < mean_low:
        direction = "짧다"
    else:
        return "두 그룹의 주당 근무시간 평균이 같다"

    return (
        f"고소득 그룹의 주당 근무시간 평균({mean_high:.2f}h)이 "
        f"저소득 그룹({mean_low:.2f}h)보다 약 {difference:.2f}시간 {direction}"
    )


def run(df) -> dict:
    """입력 검증부터 Welch t-test 해석까지 순차적으로 수행한다."""
    _validate_input(df)

    # 필요한 열을 한 번만 선택해 이후 모든 통계 계산에서 재사용한다.
    numeric = df.loc[:, NUMERIC_COLS]

    # ── 1. 기술통계 ──────────────────────────────────────
    describe_md = numeric.describe().round(2).to_markdown()
    logging.info(
        "[통계 1/4] 기술통계 완료\n"
        f"- 관측치: {len(df):,}건\n"
        f"- 대상 변수: {', '.join(NUMERIC_COLS)}"
    )

    # ── 2. Pearson 상관계수 ──────────────────────────────
    corr_md = numeric.corr().round(3).to_markdown()
    constant_cols = [col for col in NUMERIC_COLS if numeric[col].nunique() < 2]
    if constant_cols:
        logging.warning(
            "[통계 2/4] 상관계수 계산 완료\n"
            f"- 값이 일정해 상관계수를 계산할 수 없는 변수: {constant_cols}"
        )
    else:
        logging.info(
            "[통계 2/4] 상관계수 계산 완료\n"
            f"- 수치형 변수 {len(NUMERIC_COLS)}개의 Pearson 상관계수"
        )

    # ── 3. IQR 경계 밖 관측값 탐지(보고만 하고 제거하지 않음) ──
    outlier_lines = ["[통계 3/4] IQR 경계 밖 관측값 확인"]
    for col in NUMERIC_COLS:
        series = numeric[col]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        outlier_lines.append(f"- {col}: {count:,}건")

    outlier_lines.append("- 처리 방침: 분포와 top-coding 특성을 고려해 제거하지 않음")
    logging.info("\n".join(outlier_lines))

    # ── 4. 소득 그룹별 주당 근무시간 Welch 독립표본 t-test ──
    high_group = df.loc[df[GROUP_COL] == HIGH_INCOME, TARGET_COL]
    low_group = df.loc[df[GROUP_COL] == LOW_INCOME, TARGET_COL]
    if len(high_group) < 2 or len(low_group) < 2:
        raise ValueError("Welch t-test에는 각 그룹별로 2개 이상의 관측값이 필요합니다.")

    t_stat, p_value = stats.ttest_ind(
        high_group.to_numpy(copy=False),
        low_group.to_numpy(copy=False),
        equal_var=False,
        nan_policy="raise",
    )
    t_stat, p_value = float(t_stat), float(p_value)
    if not np.isfinite(t_stat) or not np.isfinite(p_value):
        raise ValueError("t-test 결과를 계산할 수 없습니다. 그룹별 표본 수와 분산을 확인하세요.")

    mean_high = float(high_group.mean())
    mean_low = float(low_group.mean())
    p_text = _format_p_value(p_value)
    comparison = _describe_difference(mean_high, mean_low)

    if p_value < ALPHA:
        decision = "귀무가설을 기각한다"
        interp = (
            f"p-value({p_text})가 유의수준 {ALPHA:.2f}보다 작으므로 {decision}. "
            f"{comparison}. 따라서 두 소득 그룹의 평균 차이는 통계적으로 유의하다 "
            "(Welch t-test)."
        )
    else:
        decision = "귀무가설을 기각할 수 없다"
        interp = (
            f"p-value({p_text})가 유의수준 {ALPHA:.2f} 이상이므로 {decision}. "
            f"{comparison}. 다만 관측된 평균 차이는 통계적으로 유의하지 않다 "
            "(Welch t-test)."
        )

    logging.info(
        "[통계 4/4] Welch 독립표본 t-test 완료\n"
        "- H0: 두 소득 그룹의 평균 주당 근무시간에 차이가 없다\n"
        "- H1: 두 소득 그룹의 평균 주당 근무시간에 차이가 있다\n"
        f"- 고소득 그룹: n={len(high_group):,}, 평균={mean_high:.2f}h\n"
        f"- 저소득 그룹: n={len(low_group):,}, 평균={mean_low:.2f}h\n"
        f"- t-statistic: {t_stat:.4f}\n"
        f"- p-value: {p_text}\n"
        f"- 결론: {decision}"
    )

    return {
        "describe_md": describe_md,
        "corr_md": corr_md,
        "group_label": "고소득(>50K) vs 저소득(<=50K)의 주당 근무시간",
        "mean_a": mean_high,
        "mean_b": mean_low,
        "t_stat": t_stat,
        "p_value": p_value,
        "interp": interp,
    }
