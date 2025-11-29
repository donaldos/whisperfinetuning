from typing import Optional, Tuple, Union
from transformers import (
    WhisperConfig,
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
    try:
        return WhisperFeatureExtractor.from_pretrained(modelsname)
    except Exception as e:
        logger.error(f"FeatureExtractor load error: {e}")
        return None


def get_tokenizer(modelsname: str, langcode: str, task: str) -> Tuple[bool, Union[WhisperTokenizer, str]]:
    """
    지정한 Whisper 모델로부터 토크나이저(WhisperTokenizer)를 로드하는 함수.

    WhisperTokenizer는 Whisper 모델이 텍스트를 처리할 수 있도록 
    문자열을 토큰 ID로 변환하는 역할을 합니다.

    Args:
        modelsname (str): Hugging Face 모델 이름 (예: "openai/whisper-small")
        langcode (str): 사용할 언어 코드 (예: "en", "ko", "ja", "fr" 등)
        task (str): Whisper의 작업 모드 ("transcribe" 또는 "translate")

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
        # 지정한 모델에서 WhisperTokenizer 로드
        tokenizer = WhisperTokenizer.from_pretrained(
            modelsname,
            language=langcode,
            task=task
        )
        return True, tokenizer

    except Exception as e:
        # 로드 중 예외 발생 시 False와 오류 메시지 반환
        return False, f"{e}"


def get_processor(modelsname: str, langcode: str, task: str) -> Tuple[bool, Union[WhisperProcessor, str]]:
    """
    지정한 Whisper 모델로부터 WhisperProcessor를 로드하는 함수.

    WhisperProcessor는 Whisper 모델에서 오디오 전처리(feature extraction)와 
    텍스트 토크나이징(tokenization)을 모두 담당하는 통합 객체입니다.
    즉, Whisper 모델의 입력과 출력을 함께 관리할 수 있습니다.

    Args:
        modelsname (str): Hugging Face Hub 모델 이름 (예: "openai/whisper-small")
        language (str): Whisper 모델이 사용할 언어 코드 (예: "en", "ko", "ja")
        task (str): 작업 유형 ("transcribe" 또는 "translate")

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
        # Hugging Face Hub 또는 로컬 캐시에서 WhisperProcessor 로드
        proc = WhisperProcessor.from_pretrained(modelsname, language=langcode, task=task)
        return True, proc

    except Exception as e:
        # 로드 실패 시 오류 메시지 반환
        return False, f"{e}"
    
def get_model(model_name:str):
    cfg = WhisperConfig.from_pretrained(model_name)
    return WhisperForConditionalGeneration(cfg)

def get_trainer_args(argfilepath: str):
    parser = HfArgumentParser(Seq2SeqTrainingArguments)
    (training_args,) = parser.parse_json_file(json_file=argfilepath)