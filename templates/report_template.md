# End2End 데이터 분석 보고서 — Adult Census Income

> **프로젝트 질문: "연봉 예측에 가장 큰 영향을 준 변수는 무엇인가?"**
> SKALA 광주캠퍼스 2반 5조 · 자동 생성: {{ generated }} (Jinja2 템플릿 렌더링)

---

## 1. 데이터 준비 & EDA (Pandas vs Polars)
{% if data %}
- **원본 shape**: {{ data.shape_raw }} → **정제 후**: {{ data.shape_clean }}
- **정제 내역 (단계 분리)**: 중복 제거 **{{ data.n_dup_removed }}건** → 결측(" ?") 제거 **{{ data.n_na_removed }}건**
- **로딩 시간 (로컬 파일 기준)**: Pandas {{ "%.4f"|format(data.pd_time) }}초 / Polars {{ "%.4f"|format(data.pl_time) }}초
- **비교 소견**: {{ data.compare_note }}
{% else %}
> ⚠️ 이 단계는 실행되지 않았습니다 — `outputs/pipeline.log` 확인
{% endif %}

## 2. 시각화
{% if viz %}
- **Seaborn 정적 차트**: `{{ viz.seaborn_path }}`
- **Plotly 인터랙티브 차트**: `{{ viz.plotly_path }}`
- **차트 해설**: {{ viz.desc }}
{% else %}
> ⚠️ 이 단계는 실행되지 않았습니다 — `outputs/pipeline.log` 확인
{% endif %}

## 3. 통계 분석
{% if stats %}
### 기술통계
{{ stats.describe_md }}

### 상관계수 (수치형 변수)
{{ stats.corr_md }}

### 독립표본 t-test ({{ stats.group_label }})
- 그룹 평균: {{ "%.2f"|format(stats.mean_a) }} vs {{ "%.2f"|format(stats.mean_b) }}
- t-statistic: {{ "%.4f"|format(stats.t_stat) }} / p-value: {{ "%.3e"|format(stats.p_value) }}
- **해석**: {{ stats.interp }}
{% else %}
> ⚠️ 이 단계는 실행되지 않았습니다 — `outputs/pipeline.log` 확인
{% endif %}

## 4. ML Pipeline 평가
{% if model %}
- **모델**: {{ model.model_name }}
- **Accuracy**: {{ "%.4f"|format(model.accuracy) }} / **F1-score**: {{ "%.4f"|format(model.f1) }}
- **저장 모델**: `{{ model.model_path }}` (joblib, 재로딩 검증 완료)
{% if model.top_features_md %}

### 변수 중요도 Top 5 — 연봉을 결정하는 진짜 범인
{{ model.top_features_md }}
{% endif %}
{% if model.comparison_md %}

### 모델 비교 (RandomForest vs LogisticRegression)
{{ model.comparison_md }}
{% endif %}

```text
{{ model.clf_report }}
```
{% else %}
> ⚠️ 이 단계는 실행되지 않았습니다 — `outputs/pipeline.log` 확인
{% endif %}

---
*본 리포트는 `python main.py` 실행 시 자동 생성됩니다. 방법론 근거: `docs/decisions.md`*
