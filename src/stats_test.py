"""3단계 · 통계 분석 — 기술통계 → 상관계수 → IQR 탐지 → Welch t-test.
담당: 박연주 / 브랜치: feat/stats

리턴 키(변경 금지):
  describe_md / corr_md / group_label / mean_a / mean_b / t_stat / p_value / interp
"""

import logging

import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy import stats


# 기술통계와 Pearson 상관계수에 사용할 수치형 변수다.
# fnlwgt는 개인 특성이 아닌 표본 가중치이므로 팀 결정에 따라 제외했다.
# education-num은 모델에서는 education과 중복되어 제외하지만 통계 분석에는 사용한다.
NUMERIC_COLS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# t-test는 income에 따라 두 그룹을 만들고 hours-per-week의 평균을 비교한다.
GROUP_COL = "income"
TARGET_COL = "hours-per-week"
HIGH_INCOME = ">50K"
LOW_INCOME = "<=50K"

# 유의수준 5%: p-value가 이 값보다 작으면 귀무가설을 기각한다.
ALPHA = 0.05


def _validate_input(df) -> None:
    """분석에 필요한 데이터 구조와 값을 실행 전에 검증한다."""
    # 1) 입력 객체와 데이터 행 존재 여부를 먼저 확인한다.
    # 잘못된 입력을 뒤쪽 통계 함수까지 보내면 원인을 찾기 어려운 오류가 발생한다.
    if df is None or not hasattr(df, "columns"):
        raise TypeError("Pandas DataFrame을 전달해야 합니다.")
    if df.empty:
        raise ValueError("분석할 데이터가 비어 있습니다.")

    # 2) 실습에 필요한 컬럼이 모두 있는지 확인한다.
    # 누락된 컬럼을 한 번에 알려주어 입력 데이터 문제를 쉽게 찾도록 한다.
    required = NUMERIC_COLS + [GROUP_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필요 컬럼 누락: {missing}")

    # 3) 평균, 분위수, 상관계수를 계산할 컬럼은 반드시 수치형이어야 한다.
    non_numeric = [col for col in NUMERIC_COLS if not is_numeric_dtype(df[col])]
    if non_numeric:
        raise TypeError(f"수치형이어야 하는 컬럼: {non_numeric}")

    # 4) 이 파이프라인은 load.py에서 결측 제거를 마친 df_clean을 입력으로 받는다.
    # 결측이 남아 있으면 단계별 표본 수가 달라질 수 있으므로 조용히 제외하지 않고 중단한다.
    null_cols = [col for col in required if df[col].isna().any()]
    if null_cols:
        raise ValueError(f"결측값이 포함된 컬럼: {null_cols}")

    # 5) NaN과 별도로 양·음의 무한대도 통계량을 망가뜨릴 수 있어 검사한다.
    # 열 단위로 확인해 전체 크기의 임시 Boolean DataFrame 생성을 피한다.
    infinite_cols = [
        col
        for col in NUMERIC_COLS
        if not np.isfinite(df[col].to_numpy(copy=False)).all()
    ]
    if infinite_cols:
        raise ValueError(f"무한대 값이 포함된 컬럼: {infinite_cols}")

    # 6) income 라벨의 오탈자나 선행 공백이 있으면 일부 행이 그룹에서 빠질 수 있다.
    # 예상하지 못한 라벨과 누락된 비교 그룹을 각각 확인해 이를 방지한다.
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
    # 부동소수점 표현 한계보다 작은 값은 계산 결과가 0.0이 될 수 있다.
    # 이는 실제 확률이 정확히 0이라는 뜻이 아니므로 상한 형태로 표현한다.
    return "< 1e-300" if p_value == 0 else f"{p_value:.3e}"


def _describe_difference(mean_high: float, mean_low: float) -> str:
    """두 그룹 평균 차이의 크기와 방향을 자연어로 표현한다."""
    # 절댓값은 차이의 크기에 사용하고, 원래 평균 비교로 길다/짧다를 결정한다.
    # 이렇게 하면 음수 차이를 "-2시간 길다"라고 잘못 출력하지 않는다.
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
    # ── 0. 입력 검증 ──────────────────────────────────────
    # 분석 전에 실패 원인을 차단하여 잘못된 통계 결과가 리포트에 들어가지 않게 한다.
    _validate_input(df)

    # 필요한 열을 한 번만 선택하고 모든 통계 계산에서 재사용한다.
    # 같은 DataFrame 부분집합을 반복 생성하지 않아 코드와 메모리 사용이 간결해진다.
    numeric = df.loc[:, NUMERIC_COLS]

    # ── 1. 기술통계 ──────────────────────────────────────
    # describe(): 개수, 평균, 표준편차, 최솟값, 사분위수, 최댓값을 계산한다.
    # 결과는 보고서에 바로 넣을 수 있도록 Markdown 표 문자열로 변환한다.
    describe_md = numeric.describe().round(2).to_markdown()
    logging.info(
        "[통계 1/4] 기술통계 완료\n"
        f"- 관측치: {len(df):,}건\n"
        f"- 대상 변수: {', '.join(NUMERIC_COLS)}"
    )

    # ── 2. Pearson 상관계수 ──────────────────────────────
    # corr(): 두 수치형 변수 사이의 선형 관계를 -1~1 범위로 계산한다.
    # 1에 가까우면 양의 관계, -1에 가까우면 음의 관계, 0에 가까우면 선형 관계가 약하다.
    # 상관관계는 두 변수가 함께 변하는 정도이며 인과관계를 의미하지 않는다.
    corr_md = numeric.corr().round(3).to_markdown()

    # 값이 하나뿐인 변수는 분산이 0이므로 상관계수를 정의할 수 없다.
    # 보고서 생성은 유지하되 사용자가 결과를 오해하지 않도록 경고한다.
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
    # IQR = Q3 - Q1이며, [Q1 - 1.5×IQR, Q3 + 1.5×IQR] 밖의 값을 센다.
    # age의 편포, capital-gain의 top-coding처럼 실제 데이터 특성일 수 있으므로
    # 경계 밖 관측값의 건수만 확인하고 원본 데이터에서는 제거하지 않는다.
    outlier_lines = ["[통계 3/4] IQR 경계 밖 관측값 확인"]
    for col in NUMERIC_COLS:
        # 한 열씩 처리하면 모든 열에 대한 Boolean 표를 한꺼번에 만들지 않아 메모리에 유리하다.
        series = numeric[col]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        outlier_lines.append(f"- {col}: {count:,}건")

    outlier_lines.append("- 처리 방침: 분포와 top-coding 특성을 고려해 제거하지 않음")
    logging.info("\n".join(outlier_lines))

    # ── 4. 소득 그룹별 주당 근무시간 Welch 독립표본 t-test ──
    # H0(귀무가설): 두 소득 그룹의 평균 주당 근무시간에 차이가 없다.
    # H1(대립가설): 두 소득 그룹의 평균 주당 근무시간에 차이가 있다.
    # 두 독립 그룹의 분산이 같다고 단정할 수 없으므로 Student 방식이 아닌
    # equal_var=False인 Welch t-test를 사용한다.
    high_group = df.loc[df[GROUP_COL] == HIGH_INCOME, TARGET_COL]
    low_group = df.loc[df[GROUP_COL] == LOW_INCOME, TARGET_COL]

    # 평균과 표본분산을 이용하는 검정이므로 각 그룹에 최소 2개 관측값이 필요하다.
    if len(high_group) < 2 or len(low_group) < 2:
        raise ValueError("Welch t-test에는 각 그룹별로 2개 이상의 관측값이 필요합니다.")

    # Pandas Series의 내부 배열을 불필요하게 복사하지 않고 SciPy에 전달한다.
    # nan_policy="raise"는 예상하지 못한 결측이 있을 때 조용히 계산하지 않게 한다.
    t_stat, p_value = stats.ttest_ind(
        high_group.to_numpy(copy=False),
        low_group.to_numpy(copy=False),
        equal_var=False,
        nan_policy="raise",
    )

    # NumPy 스칼라를 일반 float로 바꾸면 Jinja2 템플릿과 직렬화에서 다루기 쉽다.
    # 표본분산이 0인 특수 상황 등으로 유효하지 않은 결과가 나오면 해석 전에 중단한다.
    t_stat, p_value = float(t_stat), float(p_value)
    if not np.isfinite(t_stat) or not np.isfinite(p_value):
        raise ValueError("t-test 결과를 계산할 수 없습니다. 그룹별 표본 수와 분산을 확인하세요.")

    # p-value뿐 아니라 실제 평균과 차이의 방향도 함께 제시해야 결과 해석이 완성된다.
    mean_high = float(high_group.mean())
    mean_low = float(low_group.mean())
    p_text = _format_p_value(p_value)
    comparison = _describe_difference(mean_high, mean_low)

    # p-value가 유의수준 0.05보다 작으면 H0를 기각하고 H1을 지지한다.
    # 크거나 같으면 H0가 참이라고 확정하는 것이 아니라 "기각할 근거가 부족하다"고 표현한다.
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

    # ── 5. 결과 반환 ──────────────────────────────────────
    # 고정된 키는 report.py와 report_template.md가 그대로 사용하므로 변경하지 않는다.
    # 표는 Markdown 문자열로, 계산 결과는 숫자로, 결론은 해석 문장으로 전달한다.
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
