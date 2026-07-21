# skala-data-analysis-gwangju-2-5

SKALA 광주캠퍼스 2반 **5조** — [Day 2] 종합실습: **End2End 데이터 분석 프로젝트** (Adult Census Income)

Pandas·Polars 로딩 비교 → 전처리·EDA → Seaborn/Plotly 시각화 → t-test 통계검정 → sklearn Pipeline + joblib → **report.md 자동 생성(Jinja2)** 까지 한 번의 실행으로 이어지는 파이프라인입니다.

---

## 역할 분담

| 단계 | 모듈 | 브랜치 | 담당 |
|---|---|---|---|
| 1. 데이터 준비 (Pandas/Polars 비교, 결측·중복, EDA) | `src/load.py` | `feat/load` | 장병헌 (구현 완료) |
| 2. 시각화 (Seaborn + Plotly) | `src/viz.py` | `feat/viz` | 김단빈 |
| 3. 통계 분석 (기술통계·상관·t-test) | `src/stats_test.py` | `feat/stats` | 박연주 |
| 4. ML Pipeline (전처리+모델+평가+joblib) | `src/model.py` | `feat/model` | 김승현 |
| 5. report.md 자동 생성 + 통합·머지 관리 | `src/report.py`, `main.py` | `feat/report` | 장병헌 (메인테이너) |

각 모듈 파일 안에 **TODO 가이드 주석**이 순서대로 적혀 있습니다. **`run()`의 시그니처와 리턴 dict 키는 변경 금지** — report가 그대로 받아 씁니다.

## 폴더 구조

```
skala-data-analysis-gwangju-2-5/
├── main.py                     # 실행 진입점 — 5단계를 개별 try로 연결
├── src/                        # 재사용 모듈 (단계별 1파일)
├── templates/report_template.md# Jinja2 리포트 템플릿
├── notebooks/                  # 개인 실험용 (파일명: 01_이름_주제.ipynb)
├── data/raw/                   # 원본 데이터 — git 제외, 실행 시 자동 다운로드
└── outputs/                    # 차트·report.md·모델·로그 — git 제외(최종 통합 시만 커밋)
```

## 환경 설정

```bash
git clone <레포 URL> && cd skala-data-analysis-gwangju-2-5
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 실행 방법

```bash
python main.py    # 반드시 레포 루트에서 실행
```

- 최초 실행 시 원본 데이터를 `data/raw/adult.data`로 자동 다운로드합니다.
- 어떤 단계가 미구현/실패여도 끝까지 돌고, `outputs/report.md`에 단계별 상태가 표기됩니다.
- 로그: `outputs/pipeline.log`

## 데이터 출처 (원본은 git에 올리지 않습니다)

- Adult Census Income — UCI ML Repository:
  `https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`
- 참고: 결측치는 NaN이 아니라 문자열 `" ?"`로 들어 있음 (로딩 시 처리 필수)

---

## 협업 컨벤션 (필독)

### 작업 순서
1. 시작 전 항상: `git checkout main && git pull origin main`
2. 브랜치 이동: `git checkout feat/<파트>` (원격에 이미 생성돼 있어 자동으로 잡힘)
3. 담당 모듈의 TODO 순서대로 구현 → 커밋 → `git push origin feat/<파트>`
4. GitHub에서 **main 대상 PR 생성** → 메인테이너가 리뷰·머지 (Squash and merge)

### 커밋 메시지
`type: 한글 설명` 형식 — `feat:`(기능) `fix:`(수정) `docs:`(문서) `chore:`(설정)
예) `feat: Polars 로딩 및 결측치 처리 구현`

### PR 규칙
- **main 직접 push 금지** — 모든 변경은 PR로만, 머지는 메인테이너가
- PR 본문에 **실행 출력 캡처**(또는 붙여넣기) 포함
- 머지 전 셀프체크 3종:
  - [ ] 주석 달았나? (**주석 누락 시 감점** — 채점 기준 명시)
  - [ ] try/except 예외처리 넣었나? (강사 명시 필수)
  - [ ] 레포 루트에서 `python main.py` 돌려봤나?

### 산출물·데이터 규칙
- `data/`, `outputs/`는 **커밋 금지**(.gitignore 처리됨) — 최종 통합 시 메인테이너가 `git add -f outputs/`로 1회 커밋
- API 키·개인정보는 절대 커밋 금지 (`.env` 사용, .gitignore로 차단됨)

---

## 제출 (개인별!)

- 마감: **7/21(화) 21:00 엄수** (초과 시 감점 / 강사 구두 최종선: 7/22 09:00)
- 코드 zip: 레포 다운로드 → `광주_2반_본인이름_day2종합실습.zip` 으로 이름 변경 후 제출
- 실행결과 PDF: 실행 화면 캡처 + 본인 의견(개선점·코드 품질) + **팀원별 의견 + 종합의견**
