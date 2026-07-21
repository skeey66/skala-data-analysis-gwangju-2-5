# End2End 데이터 분석 보고서 — Adult Census Income

> **프로젝트 질문: "연봉 예측에 가장 큰 영향을 준 변수는 무엇인가?"**
> SKALA 광주캠퍼스 2반 5조 · 자동 생성: 2026-07-21 23:51 (Jinja2 템플릿 렌더링)

---

## 1. 데이터 준비 & EDA (Pandas vs Polars)

- **원본 shape**: (32561, 15) → **정제 후**: (30139, 15)
- **정제 내역 (단계 분리)**: 중복 제거 **24건** → 결측(" ?") 제거 **2398건**
- **로딩 시간 (로컬 파일 기준)**: Pandas 0.0259초 / Polars 0.0425초
- **비교 소견**: 동일 로컬 파일 기준 Polars가 Pandas 대비 약 0.6배 빠르게 로딩됨(멀티스레드 파서). 결측 마커 처리 방향이 정반대인 점이 핵심 차이: Pandas는 skipinitialspace가 공백을 먼저 제거해 na_values='?'(공백 없음)여야 하고, Polars는 공백이 보존되어 null_values=' ?'(공백 포함)여야 인식됨. 또한 파일 말미 빈 줄을 Pandas는 자동 스킵하지만 Polars는 전체-null 행으로 읽어 후처리로 제거함. 문자열 선행 공백도 Polars는 str.strip_chars() 후처리 필요. 정합 후 두 라이브러리 shape 동일 확인.


## 2. 시각화

- **Seaborn 정적 차트**: `outputs/seaborn_chart.png`
- **Plotly 인터랙티브 차트**: `outputs/plotly_chart.html`
- **차트 해설**: 박스플롯은 소득 그룹별 주당 근무시간 분포를 비교한 것으로, 고소득(>50K) 그룹의 중앙값과 사분위 범위가 저소득 그룹보다 뚜렷하게 위에 있어, 통계 파트의 t-test 결과(고소득 그룹이 주당 약 6.4시간 더 근무, p<0.05로 유의)를 시각적으로 그대로 보여준다. 산점도(무작위 1,000건 표본)는 나이-근무시간 평면에서 소득 그룹(>50K vs <=50K)을 색으로 구분한 것으로, 고소득 그룹이 30~50대·주 40시간 이상 구간에 상대적으로 밀집하는 경향을 보여 변수 중요도 결과(age 1위, hours-per-week 3위)와도 일관된다.


## 3. 통계 분석

### 기술통계
|       |      age |   education-num |   capital-gain |   capital-loss |   hours-per-week |
|:------|---------:|----------------:|---------------:|---------------:|-----------------:|
| count | 30139    |        30139    |       30139    |       30139    |         30139    |
| mean  |    38.44 |           10.12 |        1092.84 |          88.44 |            40.93 |
| std   |    13.13 |            2.55 |        7409.11 |         404.45 |            11.98 |
| min   |    17    |            1    |           0    |           0    |             1    |
| 25%   |    28    |            9    |           0    |           0    |            40    |
| 50%   |    37    |           10    |           0    |           0    |            40    |
| 75%   |    47    |           13    |           0    |           0    |            45    |
| max   |    90    |           16    |       99999    |        4356    |            99    |

### 상관계수 (수치형 변수)
|                |   age |   education-num |   capital-gain |   capital-loss |   hours-per-week |
|:---------------|------:|----------------:|---------------:|---------------:|-----------------:|
| age            | 1     |           0.043 |          0.08  |          0.06  |            0.101 |
| education-num  | 0.043 |           1     |          0.124 |          0.08  |            0.153 |
| capital-gain   | 0.08  |           0.124 |          1     |         -0.032 |            0.08  |
| capital-loss   | 0.06  |           0.08  |         -0.032 |          1     |            0.052 |
| hours-per-week | 0.101 |           0.153 |          0.08  |          0.052 |            1     |

### 독립표본 t-test (고소득(>50K) vs 저소득(<=50K)의 주당 근무시간)
- 그룹 평균: 45.71 vs 39.35
- t-statistic: 43.1697 / p-value: 0.000e+00
- **해석**: p-value(< 1e-300)가 유의수준 0.05보다 작으므로 귀무가설을 기각한다. 고소득 그룹의 주당 근무시간 평균(45.71h)이 저소득 그룹(39.35h)보다 약 6.36시간 길다. 따라서 두 소득 그룹의 평균 차이는 통계적으로 유의하다 (Welch t-test).


## 4. ML Pipeline 평가

- **모델**: RandomForestClassifier (class_weight='balanced') + 전처리 Pipeline
- **Accuracy**: 0.8358 / **F1-score**: 0.6916
- **저장 모델**: `outputs/model_pipeline.joblib` (joblib, 재로딩 검증 완료)


### 변수 중요도 Top 5 — 연봉을 결정하는 진짜 범인
| 변수             |   중요도(합산) |
|:---------------|----------:|
| age            |    0.2324 |
| marital-status |    0.1579 |
| hours-per-week |    0.1114 |
| relationship   |    0.1033 |
| occupation     |    0.1025 |



### 모델 비교 (RandomForest vs LogisticRegression)
| 모델 | Accuracy | F1-score |
|---|---|---|
| RandomForest (메인) | 0.8358 | 0.6916 |
| LogisticRegression (비교) | 0.8031 | 0.6775 |


```text
              precision    recall  f1-score   support

       <=50K       0.91      0.87      0.89      4527
        >50K       0.65      0.74      0.69      1501

    accuracy                           0.84      6028
   macro avg       0.78      0.80      0.79      6028
weighted avg       0.84      0.84      0.84      6028

```


---
*본 리포트는 `python main.py` 실행 시 자동 생성됩니다. 방법론 근거: `docs/decisions.md`*