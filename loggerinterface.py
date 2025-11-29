# loggerinterface.py
from __future__ import annotations
import logging
import logging.handlers as handlers
import os, sys
from pathlib import Path

_CONFIGURED = False  # 중복 구성 방지

def _ensure_configured():
    global _CONFIGURED
    if _CONFIGURED:
        return

    # ---- 환경 변수로 기본값 제어 ----
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "logs/app.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] (%(filename)s:%(lineno)d) %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # ---- 루트 로거 구성 (한 번만) ----
    root = logging.getLogger()
    root.setLevel(level)

    # 중복 방지: 기존 핸들러 없을 때만 추가
    if not root.handlers:
        # 콘솔
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(sh)

        # 로테이팅 파일
        fh = handlers.RotatingFileHandler(
            log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(fh)

    # 시끄러운 서드파티 억제(선택)
    logging.getLogger("urllib3").setLevel(os.getenv("LOG_LEVEL_URLLIB3", "WARNING"))
    logging.getLogger("transformers").setLevel(os.getenv("LOG_LEVEL_TRANSFORMERS", "INFO"))

    _CONFIGURED = True

def get_logger(name: str | None = None) -> logging.Logger:
    """모듈/패키지에서 호출해 쓰는 함수."""
    _ensure_configured()
    return logging.getLogger(name if name else "app")
