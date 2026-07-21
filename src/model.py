"""4단계 · ML Pipeline — RandomForest 메인 + LogisticRegression 비교, joblib 저장
담당: 김승현 / 브랜치: feat/model  (팀 결정: docs/decisions.md 참조)

리턴 dict 키(변경 금지):
  model_name / accuracy / f1 / clf_report / model_path
  top_features_md : 변수 중요도 Top5 markdown 표 — 주제 질문의 답!
  comparison_md   : RF vs LogReg 비교 markdown 표
"""

import logging

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── 피처 정의 (0721 팀 결정 반영: 연주 제안 채택) ──
# 제외: fnlwgt(표본 가중치) / education-num(education과 중복, 통계 파트용으로 df엔 유지)
#       / native-country(원핫 차원 폭증·정보량 낮음)
NUM_FEATURES = ["age", "capital-gain", "capital-loss", "hours-per-week"]
CAT_FEATURES = ["workclass", "education", "marital-status",
                "occupation", "relationship", "race", "sex"]
TARGET_COL = "income"
POSITIVE_LABEL = ">50K"      # 이진 타겟의 양성 클래스 (소수 클래스, 약 25%)
NEGATIVE_LABEL = "<=50K"
TEST_SIZE = 0.2              # 학습:평가 = 8:2
RANDOM_STATE = 42            # 분할·모델 재현성 고정
MODEL_PATH = "outputs/model_pipeline.joblib"


def _validate_input(df) -> None:
    """학습에 필요한 데이터 구조와 값을 실행 전에 검증한다."""
    # 1) 입력 객체와 데이터 행 존재 여부를 먼저 확인한다.
    if df is None or not hasattr(df, "columns"):
        raise TypeError("Pandas DataFrame을 전달해야 합니다.")
    if df.empty:
        raise ValueError("학습할 데이터가 비어 있습니다.")

    # 2) 학습에 필요한 컬럼이 모두 있는지 한 번에 확인한다.
    required = NUM_FEATURES + CAT_FEATURES + [TARGET_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필요 컬럼 누락: {missing}")

    # 3) StandardScaler를 적용할 컬럼은 반드시 수치형이어야 한다.
    non_numeric = [col for col in NUM_FEATURES if not is_numeric_dtype(df[col])]
    if non_numeric:
        raise TypeError(f"수치형이어야 하는 컬럼: {non_numeric}")

    # 4) load.py에서 결측 제거를 마친 df_clean을 받으므로, 결측이 남아 있으면
    # 파이프라인 앞 단계가 깨진 것이다 — 조용히 진행하지 않고 중단한다.
    null_cols = [col for col in required if df[col].isna().any()]
    if null_cols:
        raise ValueError(f"결측값이 포함된 컬럼: {null_cols}")

    # 5) income 라벨에 오탈자·선행 공백·다른 파일 형식(">50K." 등)이 섞이면
    # y가 한 클래스로 쏠려도 겉보기엔 정상 학습되므로 라벨 집합을 직접 검증한다.
    labels = set(df[TARGET_COL].unique())
    expected = {POSITIVE_LABEL, NEGATIVE_LABEL}
    unexpected = labels - expected
    if unexpected:
        raise ValueError(f"income에 예상하지 못한 값이 있습니다: {sorted(unexpected)}")

    missing_labels = expected - labels
    if missing_labels:
        raise ValueError(f"학습에 필요한 클래스가 없습니다: {sorted(missing_labels)}")


def _build_preprocessor() -> ColumnTransformer:
    """수치형=표준화, 범주형=원핫(빈도 1% 미만 희귀 카테고리 자동 통합 — 승현 조사 반영)."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), CAT_FEATURES),
    ])


def _fit_and_eval(clf, X_train, y_train, X_test, y_test):
    """전처리+분류기 파이프라인을 학습·평가해 (pipe, y_pred, accuracy, f1)을 반환한다.

    RF·LogReg가 같은 전처리 조건에서 공정하게 비교되도록 구성을 한 곳에 모음.
    """
    pipe = Pipeline(steps=[
        ("preprocessor", _build_preprocessor()),
        ("classifier", clf),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return pipe, y_pred, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)


def run(df) -> dict:
    """ML 파이프라인 전체 실행 — 검증 → 분할 → 학습 → 평가 → 중요도 → 저장 → 비교."""

    # ── 1. 입력 검증 + X, y 준비 ──
    _validate_input(df)
    y = (df[TARGET_COL] == POSITIVE_LABEL).astype(int)   # 이진 타겟: >50K=1, <=50K=0
    X = df[NUM_FEATURES + CAT_FEATURES]

    # ── 2. 분할 — 클래스 불균형(약 75:25) 대응으로 stratify 필수 ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    logging.info(
        "[모델 1/5] 데이터 분할 완료\n"
        f"- train {X_train.shape} / test {X_test.shape} (stratify 적용)\n"
        f"- 고소득(>50K) 비율: train {y_train.mean():.1%} / test {y_test.mean():.1%}"
    )

    # ── 3. 메인 모델 학습·평가 — 불균형 데이터라 Accuracy와 F1을 함께 본다 ──
    pipe, y_pred, accuracy, f1 = _fit_and_eval(
        RandomForestClassifier(
            class_weight="balanced",   # 불균형 대응 (docs/decisions.md)
            random_state=RANDOM_STATE, n_jobs=-1),
        X_train, y_train, X_test, y_test)
    clf_report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])
    recall_high = recall_score(y_test, y_pred)   # 소수 클래스(>50K) recall — 과제 목적상 핵심 지표
    logging.info(
        "[모델 2/5] RandomForest 평가 완료\n"
        f"- Accuracy {accuracy:.4f} / F1 {f1:.4f}\n"
        f"- 소수 클래스(>50K) recall {recall_high:.4f} — class_weight='balanced' 효과"
    )

    # ── 4. 변수 중요도 Top5 — "연봉을 결정하는 진짜 범인" (주제 질문의 답) ──
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    imps = pipe.named_steps["classifier"].feature_importances_
    ser = pd.Series(imps, index=names)
    # 원핫 조각("cat__education_Bachelors")을 원변수 단위로 합산
    # ※ 이 데이터는 컬럼·카테고리가 하이픈만 쓰므로 첫 '_' 기준 분리가 안전함
    top = (ser.groupby(lambda n: n.split("__", 1)[1].split("_", 1)[0])
              .sum().nlargest(5).round(4))
    top.index.name = "변수"
    top.name = "중요도(합산)"
    top_features_md = top.to_markdown()
    logging.info(f"[모델 3/5] 변수 중요도 Top5:\n{top.to_string()}")

    # ── 5. joblib 저장 + 재로딩 검증 (③연쇄: 검증 실패해도 결과 리턴은 유지) ──
    joblib.dump(pipe, MODEL_PATH)
    try:
        loaded = joblib.load(MODEL_PATH)
        loaded.predict(X_test.iloc[:5])   # 저장본이 실제로 동작하는지 확인
        logging.info(f"[모델 4/5] joblib 저장·재로딩 검증 완료: {MODEL_PATH}")
    except Exception as e:
        logging.error(f"joblib 재로딩 검증 실패(파일은 저장됨): {e}")

    # ── 6. 비교 모델: LogisticRegression (개별 try — 실패해도 메인 결과는 살림) ──
    comparison_md = None
    try:
        _, _, acc_lr, f1_lr = _fit_and_eval(
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
            X_train, y_train, X_test, y_test)
        comparison_md = (
            "| 모델 | Accuracy | F1-score |\n|---|---|---|\n"
            f"| RandomForest (메인) | {accuracy:.4f} | {f1:.4f} |\n"
            f"| LogisticRegression (비교) | {acc_lr:.4f} | {f1_lr:.4f} |"
        )
        logging.info(f"[모델 5/5] LogReg 비교 — Accuracy {acc_lr:.4f} / F1 {f1_lr:.4f}")
    except Exception as e:
        logging.error(f"비교 모델 실패 — 메인 결과만 리턴: {e}")

    # ── 7. 리턴 (키 이름 고정 — report가 그대로 소비) ──
    return {
        "model_name": "RandomForestClassifier (class_weight='balanced') + 전처리 Pipeline",
        "accuracy": accuracy,
        "f1": f1,
        "clf_report": clf_report,
        "model_path": MODEL_PATH,
        "top_features_md": top_features_md,
        "comparison_md": comparison_md,
    }
