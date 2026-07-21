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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── 피처 정의 (0721 팀 결정 반영: 연주 제안 채택) ──
# 제외: fnlwgt(표본 가중치) / education-num(education과 중복, 통계 파트용으로 df엔 유지)
#       / native-country(원핫 차원 폭증·정보량 낮음)
NUM_FEATURES = ["age", "capital-gain", "capital-loss", "hours-per-week"]
CAT_FEATURES = ["workclass", "education", "marital-status",
                "occupation", "relationship", "race", "sex"]
MODEL_PATH = "outputs/model_pipeline.joblib"


def _build_preprocessor() -> ColumnTransformer:
    """수치형=표준화, 범주형=원핫(빈도 1% 미만 희귀 카테고리 자동 통합 — 승현 조사 반영)."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), CAT_FEATURES),
    ])


def run(df) -> dict:
    """ML 파이프라인 전체 실행 — 분할 → 학습 → 평가 → 중요도 → 저장 → 비교."""

    # ── 1. X, y 준비 (②가정 방어: 필요한 컬럼이 실제로 있는지 먼저 확인) ──
    required = set(NUM_FEATURES + CAT_FEATURES + ["income"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필요 컬럼 누락: {sorted(missing)}")

    y = (df["income"] == ">50K").astype(int)   # 이진 타겟: >50K=1, <=50K=0
    if y.nunique() < 2:                        # 타겟 형식이 깨졌으면 즉시 중단
        raise ValueError("타겟이 한 클래스뿐 — income 값 형식을 재확인할 것")
    X = df[NUM_FEATURES + CAT_FEATURES]

    # ── 2. 분할 — 클래스 불균형(약 75:25) 대응으로 stratify 필수 ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info(f"train {X_train.shape} / test {X_test.shape} (stratify 적용)")

    # ── 3. 메인 모델: 전처리 + RandomForest 파이프라인 ──
    pipe = Pipeline(steps=[
        ("preprocessor", _build_preprocessor()),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",   # 불균형 대응 (docs/decisions.md)
            random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)

    # ── 4. 평가 — 불균형 데이터라 Accuracy와 F1을 함께 본다 ──
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])
    logging.info(f"[RandomForest] Accuracy {accuracy:.4f} / F1 {f1:.4f}")

    # ── 5. 변수 중요도 Top5 — "연봉을 결정하는 진짜 범인" (주제 질문의 답) ──
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
    logging.info(f"변수 중요도 Top5:\n{top.to_string()}")

    # ── 6. joblib 저장 + 재로딩 검증 (③연쇄: 검증 실패해도 결과 리턴은 유지) ──
    joblib.dump(pipe, MODEL_PATH)
    try:
        loaded = joblib.load(MODEL_PATH)
        loaded.predict(X_test.iloc[:5])   # 저장본이 실제로 동작하는지 확인
        logging.info(f"joblib 저장·재로딩 검증 완료: {MODEL_PATH}")
    except Exception as e:
        logging.error(f"joblib 재로딩 검증 실패(파일은 저장됨): {e}")

    # ── 7. 비교 모델: LogisticRegression (개별 try — 실패해도 메인 결과는 살림) ──
    comparison_md = None
    try:
        pipe_lr = Pipeline(steps=[
            ("preprocessor", _build_preprocessor()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42)),
        ])
        pipe_lr.fit(X_train, y_train)
        y_pred_lr = pipe_lr.predict(X_test)
        acc_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        comparison_md = (
            "| 모델 | Accuracy | F1-score |\n|---|---|---|\n"
            f"| RandomForest (메인) | {accuracy:.4f} | {f1:.4f} |\n"
            f"| LogisticRegression (비교) | {acc_lr:.4f} | {f1_lr:.4f} |"
        )
        logging.info(f"[LogReg 비교] Accuracy {acc_lr:.4f} / F1 {f1_lr:.4f}")
    except Exception as e:
        logging.error(f"비교 모델 실패 — 메인 결과만 리턴: {e}")

    # ── 8. 리턴 (키 이름 고정 — report가 그대로 소비) ──
    return {
        "model_name": "RandomForestClassifier (class_weight='balanced') + 전처리 Pipeline",
        "accuracy": accuracy,
        "f1": f1,
        "clf_report": clf_report,
        "model_path": MODEL_PATH,
        "top_features_md": top_features_md,
        "comparison_md": comparison_md,
      }