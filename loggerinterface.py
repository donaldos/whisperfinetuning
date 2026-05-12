"""
loggerinterface.py - 프로젝트 공통 로깅 설정 모듈

프로젝트 전체에서 일관된 로깅 형식과 출력 대상을 보장하기 위한 모듈.
루트 로거를 한 번만 구성(싱글톤 패턴)하고, 각 모듈은 get_logger()로 자식 로거를 획득.

출력 대상:
  1. 콘솔 (stdout): 실시간 모니터링용
  2. 로테이팅 파일 (logs/app.log): 10MB 단위 롤오버, 최대 5개 백업

환경 변수로 제어 가능한 설정:
  - LOG_LEVEL: 전체 로그 레벨 (기본: INFO)
  - LOG_FILE: 로그 파일 경로 (기본: logs/app.log)
  - LOG_LEVEL_URLLIB3: urllib3 라이브러리 로그 레벨 (기본: WARNING)
  - LOG_LEVEL_TRANSFORMERS: transformers 라이브러리 로그 레벨 (기본: INFO)

사용 예:
  from loggerinterface import get_logger
  logger = get_logger(__name__)
  logger.info("학습 시작")
"""

from __future__ import annotations
import logging
import logging.handlers as handlers
import os, sys
from pathlib import Path

# 싱글톤 플래그: 루트 로거 구성이 한 번만 실행되도록 보장
_CONFIGURED = False


def _ensure_configured():
    """
    루트 로거를 구성한다 (최초 1회만 실행).

    콘솔 핸들러와 로테이팅 파일 핸들러를 루트 로거에 등록하고,
    시끄러운 서드파티 라이브러리의 로그 레벨을 별도로 제어한다.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # 환경 변수에서 로그 설정 읽기 (미설정 시 기본값 사용)
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "logs/app.log")
    # 로그 파일 디렉토리가 없으면 자동 생성
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # 로그 출력 포맷: [시각] [레벨] [모듈명] (파일명:줄번호) 메시지
    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] (%(filename)s:%(lineno)d) %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # 루트 로거 설정
    root = logging.getLogger()
    root.setLevel(level)

    # 핸들러 중복 등록 방지 (모듈이 여러 번 import 되어도 안전)
    if not root.handlers:
        # 콘솔 핸들러: stdout으로 실시간 출력
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(sh)

        # 로테이팅 파일 핸들러: 10MB 초과 시 롤오버, 최대 5개 백업 유지
        fh = handlers.RotatingFileHandler(
            log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(fh)

    # 서드파티 라이브러리의 로그 레벨을 별도로 제어 (과다 로그 억제)
    logging.getLogger("urllib3").setLevel(os.getenv("LOG_LEVEL_URLLIB3", "WARNING"))
    logging.getLogger("transformers").setLevel(os.getenv("LOG_LEVEL_TRANSFORMERS", "INFO"))

    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """
    프로젝트 공통 로거를 반환한다.

    최초 호출 시 루트 로거를 자동 구성하며, 이후에는 구성된 상태를 재사용.
    각 모듈에서 get_logger(__name__)으로 호출하면 모듈명이 로그에 포함된다.

    Args:
        name: 로거 이름 (보통 __name__ 전달, None이면 "app" 사용)

    Returns:
        구성된 logging.Logger 인스턴스
    """
    _ensure_configured()
    return logging.getLogger(name if name else "app")
