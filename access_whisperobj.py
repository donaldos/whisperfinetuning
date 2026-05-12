"""
access_whisperobj.py - Whisper 모델 관련 객체 로더

Hugging Face Hub에서 Whisper의 주요 구성 요소를 로드하는 유틸리티 모듈.
  - FeatureExtractor: 원시 오디오 → log-mel spectrogram 변환
  - Tokenizer: 텍스트 ↔ 토큰 ID 변환
  - Processor: FeatureExtractor + Tokenizer를 통합한 객체
  - Model: Whisper 조건부 생성 모델 (WhisperForConditionalGeneration)

모든 로더 함수는 예외를 내부에서 처리하여 (성공여부, 결과) 튜플을 반환한다.
"""

from typing import Optional, Tuple, Union
from transformers import (
    PreTrainedTokenizerBase,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    IntervalStrategy,
    Seq2SeqTrainer,
)
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from loggerinterface import get_logger
logger = get_logger(__name__)


def get_feature_extractor(modelsname: str) -> Optional[WhisperFeatureExtractor]:
    """
    Whisper FeatureExtractor를 로드한다.

    FeatureExtractor는 원시 오디오 파형(waveform)을 모델이 이해할 수 있는
    80채널 log-mel spectrogram으로 변환하는 역할을 한다.

    Args:
        modelsname: Hugging Face Hub 모델 ID (예: "openai/whisper-tiny")

    Returns:
        성공 시 WhisperFeatureExtractor, 실패 시 None
    """
    try:
        return WhisperFeatureExtractor.from_pretrained(modelsname)
    except Exception as e:
        logger.error(f"FeatureExtractor load error: {e}")
        return None


def get_tokenizer(modelsname: str, langcode: str, task: str) -> Tuple[bool, Union[WhisperTokenizer, str]]:
    """
    지정한 Whisper 모델로부터 토크나이저(WhisperTokenizer)를 로드하는 함수.

    WhisperTokenizer는 Whisper 모델이 텍스트를 처리할 수 있도록
    문자열을 토큰 ID로 변환하는 역할을 한다.
    언어 코드와 태스크에 따라 특수 토큰(<|en|>, <|transcribe|> 등)이 자동 설정된다.

    Args:
        modelsname: Hugging Face 모델 이름 (예: "openai/whisper-small")
        langcode: 사용할 언어 코드 (예: "en", "ko", "ja", "fr" 등)
        task: Whisper의 작업 모드 ("transcribe" 또는 "translate")

    Returns:
        Tuple[bool, Union[WhisperTokenizer, str]]:
            - (True, tokenizer): 토크나이저 로드 성공
            - (False, error_message): 오류 발생 시 메시지 반환

    Example:
        >>> ok, tok = get_tokenizer("openai/whisper-small", "en", "transcribe")
        >>> if ok:
        ...     ids = tok("Hello world!").input_ids
        ...     print(ids)
    """
    try:
        tokenizer = WhisperTokenizer.from_pretrained(
            modelsname,
            language=langcode,
            task=task
        )
        return True, tokenizer

    except Exception as e:
        return False, f"{e}"


def get_processor(modelsname: str, langcode: str, task: str) -> Tuple[bool, Union[WhisperProcessor, str]]:
    """
    지정한 Whisper 모델로부터 WhisperProcessor를 로드하는 함수.

    WhisperProcessor는 내부적으로 WhisperFeatureExtractor와 WhisperTokenizer를
    모두 포함하는 통합 객체이다.
      - 입력 처리: 오디오 파형 → log-mel spectrogram (FeatureExtractor)
      - 출력 처리: 텍스트 → 토큰 ID / 토큰 ID → 텍스트 (Tokenizer)

    Args:
        modelsname: Hugging Face Hub 모델 이름 (예: "openai/whisper-small")
        langcode: Whisper 모델이 사용할 언어 코드 (예: "en", "ko", "ja")
        task: 작업 유형 ("transcribe" 또는 "translate")

    Returns:
        Tuple[bool, Union[WhisperProcessor, str]]:
            - (True, processor): Processor 로드 성공
            - (False, error_message): 로드 실패 시 오류 메시지 반환

    Example:
        >>> ok, processor = get_processor("openai/whisper-small", "en", "transcribe")
        >>> if ok:
        ...     inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        ...     print(inputs.input_features.shape)
    """
    try:
        proc = WhisperProcessor.from_pretrained(modelsname, language=langcode, task=task)
        return True, proc

    except Exception as e:
        return False, f"{e}"


def get_model(model_name: str):
    """
    Whisper 사전학습 모델을 로드한다.

    HuggingFace Hub 모델 ID 또는 로컬 경로(model/<name>/)를 모두 지원한다.
    download_model.py로 미리 다운로드한 경우 로컬 경로를 config.yaml에 지정하면
    오프라인 환경에서도 사용할 수 있다.

    Args:
        model_name: HuggingFace 모델 ID (예: "openai/whisper-tiny") 또는
                    로컬 경로 (예: "model/whisper-tiny")

    Returns:
        WhisperForConditionalGeneration: 사전학습 가중치가 로드된 Whisper 모델
    """
    return WhisperForConditionalGeneration.from_pretrained(model_name)


def get_trainer_args(argfilepath: str):
    """
    JSON 파일로부터 Seq2SeqTrainingArguments를 파싱한다.

    HfArgumentParser를 사용하여 JSON 형태의 학습 설정 파일을 읽고,
    Seq2SeqTrainingArguments 객체로 변환한다.

    Args:
        argfilepath: 학습 인자가 저장된 JSON 파일 경로
    """
    parser = HfArgumentParser(Seq2SeqTrainingArguments)
    (training_args,) = parser.parse_json_file(json_file=argfilepath)
