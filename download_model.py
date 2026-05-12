"""
download_model.py - Whisper 모델 로컬 다운로드

config.yaml에 지정된 Whisper 모델 파일을 model/<모델명>/ 디렉토리에 저장합니다.
huggingface_hub.snapshot_download()를 사용하므로 PyTorch 없이도 동작합니다.

다운로드 후 config.yaml의 model.name을 로컬 경로로 변경하면
HuggingFace Hub 접속 없이 오프라인으로 사용할 수 있습니다.

사용법:
  python download_model.py
  python download_model.py --config custom.yaml
  python download_model.py --model openai/whisper-small
  python download_model.py --model openai/whisper-base --force
"""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from config_loader import load_config
from loggerinterface import get_logger

logger = get_logger(__name__)

MODEL_ROOT = Path(__file__).parent / "model"

# PyTorch 외 포맷(Flax, TensorFlow) 제외하여 불필요한 파일 다운로드 방지
IGNORE_PATTERNS = [
    "*.msgpack",
    "flax_model*",
    "tf_model*",
    "*.h5",
    "rust_model*",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper 모델 로컬 다운로드")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="설정 파일 경로 (기본값: config.yaml)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help="HuggingFace 모델 ID (기본값: config.yaml의 model.name)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 존재하는 경우 덮어쓰기",
    )
    return parser.parse_args()


def download_model(model_id: str, force: bool = False) -> Path:
    local_path = MODEL_ROOT / model_id.split("/")[-1]

    # 파일이 실제로 있는지 확인 (빈 폴더는 미완성으로 간주)
    is_complete = local_path.exists() and any(local_path.iterdir())

    if is_complete:
        if not force:
            logger.info(f"이미 존재합니다 (스킵): {local_path}")
            logger.info("덮어쓰려면 --force 옵션을 사용하세요.")
            return local_path
        shutil.rmtree(local_path)
        logger.info(f"기존 디렉토리 삭제 후 재다운로드: {local_path}")
    elif local_path.exists():
        # 빈 폴더가 남아 있으면 제거
        local_path.rmdir()

    logger.info(f"다운로드 시작: {model_id}")
    logger.info(f"저장 경로: {local_path}")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_path),
            ignore_patterns=IGNORE_PATTERNS,
        )
    except Exception as e:
        logger.error(f"다운로드 실패: {e}")
        shutil.rmtree(local_path, ignore_errors=True)
        sys.exit(1)

    logger.info(f"다운로드 완료: {local_path}")
    return local_path


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    model_id = args.model or cfg["model"]["name"]

    local_path = download_model(model_id, force=args.force)

    local_model_name = f"model/{local_path.name}"
    print(
        f"\n완료: {local_path}\n"
        f"\nconfig.yaml의 model.name을 로컬 경로로 변경하면 오프라인 사용이 가능합니다:\n"
        f"  model:\n"
        f'    name: "{local_model_name}"'
    )
