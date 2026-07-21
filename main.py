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


class ConsoleFormatter(logging.Formatter):
    """터미널에서는 시간과 상태 기호만 간결하게 표시한다."""

    SYMBOLS = {
        logging.DEBUG: "·",
        logging.INFO: "✓",
        logging.WARNING: "!",
        logging.ERROR: "✗",
        logging.CRITICAL: "✗",
    }

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if not message:
            return ""

        time_text = self.formatTime(record, "%H:%M:%S")
        symbol = self.SYMBOLS.get(record.levelno, "·")
        continuation = "\n           "
        message = message.replace("\n", continuation)
        output = f"{time_text} {symbol} {message}"

        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            output += continuation + exception_text.replace("\n", continuation)

        return output


# ── 산출물 폴더 보장 (로그 파일이 여기에 쌓이므로 로깅 설정보다 먼저) ──
Path("outputs").mkdir(exist_ok=True)

# ── 로깅 설정: 콘솔은 간결하게, 로그 파일은 상세하게 기록 ──
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ConsoleFormatter())

file_handler = logging.FileHandler("outputs/pipeline.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)

root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# 분석 결과와 무관한 Matplotlib 폰트 탐색 메시지는 터미널과 로그에서 제외한다.
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def log_stage(number: int, title: str) -> None:
    """파이프라인 단계 시작을 눈에 띄게 구분한다."""
    logging.info("")
    logging.info("━" * 56)
    logging.info(f"[{number}/5] {title}")
    logging.info("━" * 56)


def main() -> None:
    """5단계 파이프라인 실행 — 단계별 결과를 ctx에 모아 report가 최종 소비."""
    ctx = {"data": None, "viz": None, "stats": None, "model": None}
    logging.info("Adult Census Income 분석 파이프라인 시작")

    # ── 1단계: 데이터 준비 (담당: 장병헌) ──
    # import까지 try 안에: 모듈 자체가 깨져 있어도 이 단계만 실패 처리 (③연쇄 차단)
    log_stage(1, "데이터 준비")
    try:
        from src import load
        ctx["data"] = load.run()
        logging.info("데이터 준비 완료")
    except Exception as e:
        logging.error(f"1단계 데이터 준비 실패: {e}")

    df = ctx["data"]["df_clean"] if ctx["data"] else None

    # ── 2~4단계: 데이터가 있어야 진행 가능. 서로는 독립이므로 개별 try ──
    if df is None:
        logging.error("정제 데이터가 없어 2~4단계를 건너뜁니다 (report만 생성)")
    else:
        # ── 2단계: 시각화 (담당: 김단빈 / feat/viz) ──
        log_stage(2, "데이터 시각화")
        try:
            from src import viz
            ctx["viz"] = viz.run(df)
            logging.info("시각화 완료")
        except Exception as e:
            logging.error(f"2단계 시각화 실패: {e}")

        # ── 3단계: 통계 분석 (담당: 박연주 / feat/stats) ──
        log_stage(3, "통계 분석")
        try:
            from src import stats_test
            ctx["stats"] = stats_test.run(df)
            logging.info("통계 분석 완료")
        except Exception as e:
            logging.error(f"3단계 통계 분석 실패: {e}")

        # ── 4단계: ML Pipeline (담당: 김승현 / feat/model) ──
        log_stage(4, "머신러닝 모델 학습")
        try:
            from src import model
            ctx["model"] = model.run(df)
            logging.info("머신러닝 모델 학습 완료")
        except Exception as e:
            logging.error(f"4단계 ML 파이프라인 실패: {e}")

    # ── 5단계: report.md 자동 생성 (담당: 장병헌) ──
    # 어떤 단계가 실패했어도 항상 실행 → 상태판 역할까지 겸함
    log_stage(5, "분석 리포트 생성")
    try:
        from src import report
        result = report.run(ctx)
        logging.info(f"리포트 생성 완료 → {result['report_path']}")
    except Exception as e:
        logging.error(f"5단계 리포트 생성 실패: {e}")

    logging.info("")
    logging.info("모든 파이프라인 작업이 종료되었습니다.")


if __name__ == "__main__":
    main()
