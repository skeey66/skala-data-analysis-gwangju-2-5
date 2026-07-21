"""5단계 · report.md 자동 생성 — Jinja2 템플릿 렌더링 (12장 응용)
담당: 장병헌 / 브랜치: feat/report

동작:
  - templates/report_template.md 를 로드해 ctx(1~4단계 결과)를 주입
  - 실패한 단계는 None으로 들어오며, 템플릿이 "미실행/실패" 상태로 표기
    → 리포트가 파이프라인 상태판 역할까지 겸함 (오류 기록 원칙)
"""

import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

REPORT_PATH = Path("outputs/report.md")


def run(ctx: dict) -> dict:
    """ctx = {"data":..., "viz":..., "stats":..., "model":...} → outputs/report.md 생성."""
    # ── 템플릿 로드 (레포 루트 기준 상대경로 — 실행은 반드시 루트에서) ──
    env = Environment(loader=FileSystemLoader("templates"))
    tmpl = env.get_template("report_template.md")

    # ── 렌더링: 각 단계 결과 + 생성 시각 주입 ──
    md = tmpl.render(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        data=ctx.get("data"),
        viz=ctx.get("viz"),
        stats=ctx.get("stats"),
        model=ctx.get("model"),
    )

    # ── 저장 (utf-8 명시 — 한글 깨짐 방지) ──
    REPORT_PATH.write_text(md, encoding="utf-8")
    logging.info(f"report.md 생성: {REPORT_PATH.resolve()}")
    return {"report_path": str(REPORT_PATH)}
