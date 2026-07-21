"""4단계 · ML Pipeline — RandomForest 메인 + LogisticRegression 비교, joblib 저장
담당: 김승현 / 브랜치: feat/model  (팀 결정: docs/decisions.md 참조)

리턴 dict 키(변경 금지):
  model_name / accuracy / f1 / clf_report / model_path
  top_features_md (선택) : 변수 중요도 Top5 markdown 표 — 주제 질문의 답!
  comparison_md   (선택) : RF vs LogReg 비교 markdown 표
"""

import logging

# ── 피처 정의 (0721 팀 결정 반영: 연주 제안 채택) ──
# 제외: fnlwgt(표본 가중치) / education-num(education과 중복, 통계 파트용으로 df엔 유지)
#       / native-country(원핫 차원 폭증·정보량 낮음)
NUM_FEATURES = ["age", "capital-gain", "capital-loss", "hours-per-week"]
CAT_FEATURES = ["workclass", "education", "marital-status",
                "occupation", "relationship", "race", "sex"]


def run(df) -> dict:
    """ML 파이프라인 전체 실행. 아래 TODO 순서대로 구현하세요."""

    # ── TODO 1. X, y 준비 ──
    #   y = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
    #   X = df[NUM_FEATURES + CAT_FEATURES]

    # ── TODO 2. 분할 — 클래스 불균형(75:25) 대응 stratify 필수 ──
    #   train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ── TODO 3. 전처리 + 메인 모델 Pipeline ──
    #   preprocessor = ColumnTransformer([
    #       ('num', StandardScaler(), NUM_FEATURES),
    #       ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=0.01), CAT_FEATURES)])
    #   ※ min_frequency=0.01 = 빈도 1% 미만 희귀 카테고리 자동 통합 (승현 조사 반영)
    #   pipe = Pipeline([('preprocessor', preprocessor),
    #                    ('classifier', RandomForestClassifier(
    #                         class_weight='balanced', random_state=42, n_jobs=-1))])

    # ── TODO 4. 학습 → 예측 → 평가 (accuracy_score, f1_score, classification_report) ──

    # ── TODO 5. 변수 중요도 Top5 — "연봉을 결정하는 진짜 범인" ──
    #   import pandas as pd
    #   names = pipe.named_steps['preprocessor'].get_feature_names_out()
    #   imps  = pipe.named_steps['classifier'].feature_importances_
    #   ser = pd.Series(imps, index=names)   # 원핫 조각을 원변수 단위로 합산:
    #   top = ser.groupby(lambda n: n.split('__', 1)[1].split('_', 1)[0]).sum().nlargest(5)
    #   top_features_md = top.round(4).to_markdown()

    # ── TODO 6. joblib 저장 + 재로딩 검증 ──
    #   joblib.dump(pipe, 'outputs/model_pipeline.joblib')
    #   loaded = joblib.load('outputs/model_pipeline.joblib'); loaded.predict(X_test[:5])
    #   logging.info("joblib 재로딩 검증 완료")

    # ── TODO 7. (시간 되면) LogisticRegression(class_weight='balanced', max_iter=1000) 비교 ──
    #   같은 preprocessor로 Pipeline 하나 더 → acc/F1 비교표 → comparison_md

    # ── TODO 8. 리턴 (키 이름 고정, 선택 키는 없으면 생략 가능) ──

    raise NotImplementedError("feat/model 브랜치에서 구현 후 이 줄을 삭제하세요")
