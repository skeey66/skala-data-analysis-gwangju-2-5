# 팀 의사결정 기록 (0721 확정)

**주제**: 연봉 예측에 가장 큰 영향을 준 변수 찾기 — "연봉이 높은 사람들의 특징은 무엇일까?"

## 담당
- 데이터 준비 `src/load.py` — 장병헌 (구현 완료, main 포함)
- 시각화 `src/viz.py` / `feat/viz` — 김단빈
- 통계 `src/stats_test.py` / `feat/stats` — 박연주
- ML Pipeline `src/model.py` / `feat/model` — 김승현
- report.md 자동생성 + 통합·머지 — 장병헌

## 피처 확정 (연주 안 채택 — 모델 X 구성 기준)
- 수치형: age, capital-gain, capital-loss, hours-per-week
- 범주형: workclass, education, marital-status, occupation, relationship, race, sex
- 제외: fnlwgt(표본 가중치, 개인 특성 아님) / education-num(education과 중복 — 단, df_clean에는 남겨서 통계 파트 상관계수에 사용) / native-country(원핫 시 차원 폭증, 대부분 미국이라 정보량 낮음)
- "핵심 4변수"(education, marital-status, occupation, hours-per-week)는 리포트에서 집중 해석하는 주인공 변수

## 결측·이상치 방침
- 결측(" ?"): dropna로 행 제거 — 약 7% 수준, 최빈값 대체는 직업 분포 왜곡 리스크 (대안은 PDF 의견에 서술)
- 이상치: **탐지·건수 보고하되 제거하지 않음** — age는 우측 편포라 IQR 기계 적용 시 정상 고령 근로자 제거 위험 / capital-gain 99999는 top-coding(구조적 특성)이며 income과 강한 신호 / 극단 근무시간은 실제 패턴

## 모델
- 메인: RandomForestClassifier(class_weight='balanced') → feature_importances_로 주제 질문에 직접 답변
- 비교: LogisticRegression(class_weight='balanced') — coef_로 영향 방향(+/-) 제시, RF의 중요도(크기)와 상보적
- 희귀 카테고리: OneHotEncoder(min_frequency=0.01)로 자동 통합
- XGBoost는 미도입 — 승현 PDF 개선사항 의견으로 서술

## 통계 역할
- 상관계수 매트릭스: 수치형 5개(age, education-num, capital-gain, capital-loss, hours-per-week)
- t-test: 고소득(>50K) vs 저소득(<=50K)의 hours-per-week, H0/H1 명시 + 그룹 평균 포함
- 범주형 변수의 소득 영향은 corr 대신 RF importance로 답함

## PDF 의견 소재 배분
병헌=Polars 공백 처리 차이·fnlwgt 제외 근거 / 단빈=native-country 차원 폭증 / 연주=age 편포와 IQR 한계 / 승현=99999 top-coding·클래스 불균형·XGBoost 도입 제안
