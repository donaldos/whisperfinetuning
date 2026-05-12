"""
config_loader.py - YAML 설정 파일 로더

config.yaml을 읽어 딕셔너리로 반환하며,
필수 키 존재 여부와 데이터 소스/증강 모드 유효성을 검증한다.
"""

import yaml
from pathlib import Path
from loggerinterface import get_logger

logger = get_logger(__name__)

VALID_SOURCES = {"local", "huggingface"}
VALID_AUG_MODES = {"offline", "on_the_fly"}


def load_config(config_path: str = "config.yaml") -> dict:
    """
    YAML 설정 파일을 로드하고 기본 유효성을 검증한다.

    Args:
        config_path: 설정 파일 경로 (기본: config.yaml)

    Returns:
        설정 딕셔너리

    Raises:
        FileNotFoundError: 설정 파일이 없을 때
        ValueError: 필수 키 누락 또는 유효하지 않은 값
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    logger.info(f"설정 로드 완료: source={cfg['data']['source']}, augmentation={cfg['augmentation']['mode']}")
    return cfg


def _validate(cfg: dict):
    """필수 섹션과 키의 존재 여부, 값의 유효성을 검증한다."""
    for section in ("data", "augmentation", "model", "preprocessing", "training"):
        if section not in cfg:
            raise ValueError(f"설정 파일에 '{section}' 섹션이 없습니다.")

    source = cfg["data"].get("source")
    if source not in VALID_SOURCES:
        raise ValueError(f"data.source는 {VALID_SOURCES} 중 하나여야 합니다. 현재: '{source}'")

    if source == "local" and "local" not in cfg["data"]:
        raise ValueError("data.source가 'local'이면 data.local 섹션이 필요합니다.")
    if source == "huggingface" and "huggingface" not in cfg["data"]:
        raise ValueError("data.source가 'huggingface'이면 data.huggingface 섹션이 필요합니다.")

    aug_mode = cfg["augmentation"].get("mode")
    if aug_mode not in VALID_AUG_MODES:
        raise ValueError(f"augmentation.mode는 {VALID_AUG_MODES} 중 하나여야 합니다. 현재: '{aug_mode}'")
