"""
SKALA 광주 2반 5조 — [Day 2] 종합실습: End2End 데이터 분석 (Adult Census Income)

실행 방법 (반드시 레포 루트에서):
    python main.py

설계 원칙 (12장 파이프라인 원칙 적용):
  - 각 단계 분리: 단계별로 src/ 아래 모듈 하나씩 (load / viz / stats_test / model / report)
  - 오류 기록: 모든 단계는 개별 try — import 오류(문법 오류 포함)까지 단계 안에 격리되어,
    한 사람의 코드가 깨져도 나머지 단계와 리포트 생성은 계속 진행됨
  - 재현 가능: requirements.txt + README 실행 가이드
  - 테스트 가능: 각 모듈은 run() 함수 하나로 독립 호출 가능

팀 규칙: 각 모듈의 run() 시그니처와 리턴 dict 키는 절대 변경 금지 (report가 그대로 소비)
"""

import logging
from pathlib import Path

# ── 산출물 폴더 보장 (로그 파일이 여기에 쌓이므로 로깅 설정보다 먼저) ──
Path("outputs").mkdir(exist_ok=True)

# ── 로깅 설정: 콘솔 + outputs/pipeline.log 동시 기록 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("outputs/pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def main() -> None:
    """5단계 파이프라인 실행 — 단계별 결과를 ctx에 모아 report가 최종 소비."""
    ctx = {"data": None, "viz": None, "stats": None, "model": None}
    logging.info("========== 파이프라인 시작 ==========")

    # ── 1단계: 데이터 준비 (담당: 장병헌) ──
    # import까지 try 안에: 모듈 자체가 깨져 있어도 이 단계만 실패 처리 (③연쇄 차단)
    try:
        from src import load
        ctx["data"] = load.run()
        logging.info("1단계 데이터 준비 완료")
    except Exception as e:
        logging.error(f"1단계 데이터 준비 실패: {e}")

    df = ctx["data"]["df_clean"] if ctx["data"] else None

    # ── 2~4단계: 데이터가 있어야 진행 가능. 서로는 독립이므로 개별 try ──
    if df is None:
        logging.error("정제 데이터가 없어 2~4단계를 건너뜁니다 (report만 생성)")
    else:
        # ── 2단계: 시각화 (담당: 김단빈 / feat/viz) ──
        try:
            from src import viz
            ctx["viz"] = viz.run(df)
            logging.info("2단계 시각화 완료")
        except Exception as e:
            logging.error(f"2단계 시각화 실패: {e}")

        # ── 3단계: 통계 분석 (담당: 박연주 / feat/stats) ──
        try:
            from src import stats_test
            ctx["stats"] = stats_test.run(df)
            logging.info("3단계 통계 분석 완료")
        except Exception as e:
            logging.error(f"3단계 통계 분석 실패: {e}")

        # ── 4단계: ML Pipeline (담당: 김승현 / feat/model) ──
        try:
            from src import model
            ctx["model"] = model.run(df)
            logging.info("4단계 ML 파이프라인 완료")
        except Exception as e:
            logging.error(f"4단계 ML 파이프라인 실패: {e}")

    # ── 5단계: report.md 자동 생성 (담당: 장병헌) ──
    # 어떤 단계가 실패했어도 항상 실행 → 상태판 역할까지 겸함
    try:
        from src import report
        result = report.run(ctx)
        logging.info(f"5단계 리포트 생성 완료: {result['report_path']}")
    except Exception as e:
        logging.error(f"5단계 리포트 생성 실패: {e}")

    logging.info("========== 파이프라인 종료 ==========")


if __name__ == "__main__":
    main()
