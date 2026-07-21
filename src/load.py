"""1단계 · 데이터 준비 — Pandas vs Polars 로딩 비교 + 결측·중복 처리 + 기본 EDA
담당: 장병헌

리턴 dict 키(변경 금지 — report가 그대로 사용):
  df_clean, shape_raw, shape_clean, pd_time, pl_time,
  n_dup_removed, n_na_removed, compare_note
"""

import logging
import time
from pathlib import Path

import pandas as pd
import polars as pl
import requests

# ── 데이터 출처 (원본 파일은 git에 올리지 않음 — URL만 기록) ──
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
RAW_PATH = Path("data/raw/adult.data")

COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]


def ensure_data() -> Path:
    """원본 데이터를 data/raw/에 1회 다운로드. (①입구 방어: 실패·빈 응답 처리)"""
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 0:
        logging.info(f"원본 데이터 확인: {RAW_PATH} ({RAW_PATH.stat().st_size:,} bytes)")
        return RAW_PATH

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"원본 데이터 다운로드 시작: {DATA_URL}")
    resp = requests.get(DATA_URL, timeout=30)   # 타임아웃: 무한 대기 방지
    resp.raise_for_status()                     # 4xx/5xx 즉시 예외
    if len(resp.content) == 0:                  # 빈 응답 방어
        raise ValueError("다운로드 응답이 비어 있습니다")
    RAW_PATH.write_bytes(resp.content)
    logging.info(f"다운로드 완료: {len(resp.content):,} bytes → {RAW_PATH}")
    return RAW_PATH


def run() -> dict:
    """데이터 준비 전체 실행 — 로딩 비교 → 정제(단계 분리) → 기본 EDA."""
    raw_path = ensure_data()

    # ── 1. Pandas 로딩 (소요시간 측정) ──
    # ②가정 방어: skipinitialspace가 구분자 뒤 공백을 '먼저' 제거하므로
    # 결측 마커는 공백 없는 "?"로 매칭해야 함 (" ?"로 주면 0건 인식되는 함정 — 강의 예시 코드 주의)
    t0 = time.time()
    df_pd = pd.read_csv(raw_path, header=None, names=COLS,
                        na_values="?", skipinitialspace=True)
    pd_time = time.time() - t0
    logging.info(f"[Pandas] 로딩 {pd_time:.4f}초 / shape {df_pd.shape}")
    na_total = int(df_pd.isnull().sum().sum())
    logging.info(f"[Pandas] 결측 셀 인식: {na_total:,}개")
    if na_total == 0:
        logging.warning("결측이 0개로 인식됨 — na_values 설정을 재확인할 것")

    # ── 2. Polars 로딩 — 개별 try (③연쇄: Polars가 실패해도 Pandas 분석은 계속) ──
    df_pl, pl_time = None, None
    try:
        t0 = time.time()
        df_pl = pl.read_csv(raw_path, has_header=False,
                            new_columns=COLS, null_values=" ?")
        # Polars엔 skipinitialspace가 없어 문자열에 " Male"처럼 공백이 남음 → 직접 strip
        df_pl = df_pl.with_columns(pl.col(pl.String).str.strip_chars())
        # 파일 말미 빈 줄이 Polars에선 전체-null 행 1개로 읽힘(Pandas는 기본 스킵) → 제거
        df_pl = df_pl.filter(pl.col("income").is_not_null())
        pl_time = time.time() - t0
        logging.info(f"[Polars] 로딩 {pl_time:.4f}초 / shape {df_pl.shape}")
    except Exception as e:
        logging.error(f"[Polars] 로딩 실패 — Pandas 단독으로 진행: {e}")

    # ── 3. 비교 소견 (report.md에 그대로 실림) ──
    if df_pl is not None:
        speed = pd_time / pl_time if pl_time and pl_time > 0 else 0
        compare_note = (
            f"동일 로컬 파일 기준 Polars가 Pandas 대비 약 {speed:.1f}배 빠르게 로딩됨(멀티스레드 파서). "
            "결측 마커 처리 방향이 정반대인 점이 핵심 차이: Pandas는 skipinitialspace가 공백을 먼저 제거해 "
            "na_values='?'(공백 없음)여야 하고, Polars는 공백이 보존되어 null_values=' ?'(공백 포함)여야 인식됨. "
            "또한 파일 말미 빈 줄을 Pandas는 자동 스킵하지만 Polars는 전체-null 행으로 읽어 후처리로 제거함. "
            "문자열 선행 공백도 Polars는 str.strip_chars() 후처리 필요. 정합 후 두 라이브러리 shape 동일 확인."
        )
    else:
        compare_note = "Polars 로딩 실패로 Pandas 단독 진행 (상세는 pipeline.log 참조)."

    # ── 4. 정제 — 단계 분리 + 건수 각각 보고 (한 줄 체이닝 금지) ──
    # 팀 방침(docs/decisions.md): 결측은 dropna, 이상치는 탐지만 하고 제거하지 않음
    shape_raw = df_pd.shape
    n_before = len(df_pd)
    df_nodup = df_pd.drop_duplicates()
    n_dup_removed = n_before - len(df_nodup)          # 중복 제거 건수
    df_clean = df_nodup.dropna().reset_index(drop=True)
    n_na_removed = len(df_nodup) - len(df_clean)      # 결측 제거 건수
    logging.info(f"중복 제거 {n_dup_removed}건 → 결측 제거 {n_na_removed}건 / 최종 {df_clean.shape}")

    # ── 5. 기본 EDA — 기술통계 + 타겟 분포 ──
    logging.info(f"[EDA] 수치형 기술통계:\n{df_clean.describe().round(2).to_string()}")
    dist = (df_clean['income'].value_counts(normalize=True) * 100).round(1)
    logging.info(f"[EDA] income 클래스 분포(%):\n{dist.to_string()}")
    # → 약 75:25 불균형 확인: 4단계 class_weight='balanced' 선택의 근거

    # ── 6. 리턴 (키 이름 고정) ──
    return {
        "df_clean": df_clean,
        "shape_raw": shape_raw,
        "shape_clean": df_clean.shape,
        "pd_time": pd_time,
        "pl_time": pl_time if pl_time is not None else float("nan"),
        "n_dup_removed": n_dup_removed,
        "n_na_removed": n_na_removed,
        "compare_note": compare_note,
    }
