"""
preprocess_datasetdict_batch.py - 데이터 전처리 및 DataCollator 모듈

Whisper 파인튜닝을 위한 데이터 전처리 파이프라인:
  1. 컬럼 정리: 불필요한 컬럼 제거 (audio, sentence만 보존)
  2. 필터링: 오디오 길이, 텍스트 품질 기반 필터링
  3. 특징 추출: 오디오 → 80채널 log-mel spectrogram (input_features)
  4. 토크나이징: 텍스트 → 토큰 ID 시퀀스 (labels)
  5. DataCollator: 배치 패딩, 라벨 마스킹(-100), decoder_input_ids 생성
"""

import re
import unicodedata
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional, Tuple
from transformers import WhisperProcessor
from datasets import DatasetDict, Audio, concatenate_datasets, load_dataset, disable_caching
from loggerinterface import get_logger
logger = get_logger(__name__)

# ============================================================
# 0) 전역 Processor 초기화
# ============================================================
# 모듈 로드 시점에 Processor를 한 번만 생성하여 재사용
# 전처리 함수들이 공통으로 참조하는 TARGET_SR(타깃 샘플레이트) 결정
MODEL_ID = "openai/whisper-tiny"   # Whisper 모델 크기 (tiny/base/small/medium/large)
LANG     = "ko"                    # 기본 언어 설정
TASK     = "transcribe"            # 태스크: "transcribe"(전사) 또는 "translate"(번역)
processor: WhisperProcessor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANG, task=TASK)
TARGET_SR = processor.feature_extractor.sampling_rate  # Whisper 표준: 16000Hz


# ============================================================
# 1) 불필요 컬럼 제거
# ============================================================

def remove_columns_from_datasetdict(common_voice: DatasetDict, protected=("audio", "sentence")) -> Tuple[Optional[DatasetDict], str]:
    """
    DatasetDict에서 보호 대상(protected) 외의 모든 컬럼을 제거한다.

    Whisper 학습에 필요한 컬럼은 audio(오디오 데이터)와 sentence(전사 텍스트)뿐이므로,
    나머지 메타데이터 컬럼(speaker_id, chapter_id 등)을 제거하여 메모리를 절약한다.
    Arrow 기반의 zero-copy 연산으로 효율적으로 수행된다.

    Args:
        common_voice: 원본 DatasetDict
        protected: 제거하지 않을 컬럼 이름 튜플

    Returns:
        (정제된 DatasetDict, 오류 메시지) 튜플
        성공 시 오류 메시지는 빈 문자열
    """
    try:
        # train split의 컬럼 목록에서 보호 대상을 제외한 나머지를 제거 목록으로 생성
        present = set(common_voice["train"].column_names)
        cols_to_drop = [c for c in present if c not in protected]
        logger.info(f"Removed columns: {cols_to_drop}")
        # DatasetDict.remove_columns()는 모든 split에 일괄 적용
        return common_voice.remove_columns(cols_to_drop), ""

    except AttributeError:
        return None, "Error: 입력이 DatasetDict 객체가 아닙니다."
    except ValueError as e:
        return None, f"컬럼 제거 중 오류 발생: {e}"
    except Exception as e:
        return None, f"Unexpected Error: {e}"


# ============================================================
# 2) 오디오/텍스트 필터링
# ============================================================

def ensure_audio_cast(dsd: DatasetDict, audio_col="audio", sr=16000) -> DatasetDict:
    """audio 컬럼을 Audio 타입으로 캐스팅하여 자동 디코딩/리샘플링을 보장한다."""
    return dsd.cast_column(audio_col, Audio(sampling_rate=sr))


def add_duration_ms(dsd: DatasetDict, audio_col="audio") -> DatasetDict:
    """
    각 샘플에 오디오 길이(밀리초) 컬럼을 추가한다.

    계산식: duration_ms = len(array) * 1000 / sampling_rate
    이후 filter_by_duration()에서 길이 기반 필터링에 사용.
    """
    def _dur(b):
        a = b[audio_col]
        return {"duration_ms": int(len(a["array"]) * 1000 / a["sampling_rate"])}
    return DatasetDict({k: v.map(_dur) for k, v in dsd.items()})


def filter_by_duration(dsd: DatasetDict, min_ms=500, max_ms=30000) -> DatasetDict:
    """
    오디오 길이 기반 필터링.

    너무 짧은 오디오(< 500ms)는 의미 있는 음성이 아닐 수 있고,
    너무 긴 오디오(> 30s)는 Whisper의 30초 윈도우를 초과한다.

    Args:
        dsd: duration_ms 컬럼이 추가된 DatasetDict
        min_ms: 최소 오디오 길이 (밀리초)
        max_ms: 최대 오디오 길이 (밀리초)
    """
    def _ok(ex):
        d = ex.get("duration_ms")
        return (d is not None) and (min_ms <= d <= max_ms)
    return DatasetDict({k: v.filter(_ok) for k, v in dsd.items()})


# --- 텍스트 필터링 관련 상수 및 함수 ---

# 전사 텍스트에 포함되면 안 되는 특수 토큰 (노이즈, 음악, 웃음 등의 태그)
_DISALLOWED_TOKENS = {
    "[noise]", "<noise>", "(noise)", "[music]", "<music>", "(music)",
    "[laughter]", "<laughter>", "(laughter)", "<unk>", "[unk]"
}

# 한글/영문/숫자 중 하나라도 포함되는지 판단하는 정규식
# (특수문자만으로 이루어진 "문장"은 학습에 무의미)
_CONTENT_RE = re.compile(r"[A-Za-z0-9가-힣]")


def normalize_sentence(s: str) -> str:
    """
    전사 텍스트를 정규화한다.

    처리 내용:
      - Unicode NFC 정규화 (한글 자모 결합 등)
      - 말줄임표(…) → 마침표 세 개(...) 통일
      - 연속 공백 → 단일 공백
      - 앞뒤 공백 제거
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("…", "...")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def has_disallowed_tokens(s: str) -> bool:
    """전사 텍스트에 비음성 태그([noise], [music] 등)가 포함되어 있는지 확인한다."""
    s_low = s.lower()
    return any(tok in s_low for tok in _DISALLOWED_TOKENS)


def looks_like_text(s: str) -> bool:
    """문자열에 의미 있는 내용(문자/숫자/한글)이 하나라도 있는지 확인한다."""
    return bool(_CONTENT_RE.search(s))


def add_sentence_norm_and_filter(
    dsd: DatasetDict,
    sent_col="sentence",
    min_chars=1,
    max_chars=200
) -> DatasetDict:
    """
    전사 텍스트를 정규화하고, 품질 기준으로 필터링한다.

    필터 조건 (하나라도 해당하면 제거):
      - 문자 수가 min_chars 미만 또는 max_chars 초과
      - [noise], [music] 등 비음성 태그 포함
      - 의미 있는 문자(한글/영문/숫자)가 하나도 없음

    Args:
        dsd: 원본 DatasetDict
        sent_col: 전사 텍스트 컬럼명
        min_chars: 최소 문자 수
        max_chars: 최대 문자 수
    """
    # 정규화된 문장을 별도 컬럼으로 추가
    def _norm(b):
        s = normalize_sentence(b.get(sent_col, ""))
        return {"sentence_norm": s}
    dsd2 = DatasetDict({k: v.map(_norm) for k, v in dsd.items()})

    # 품질 기준 필터링
    def _ok(ex):
        s = ex["sentence_norm"]
        L = len(s)
        if not (min_chars <= L <= max_chars):
            return False
        if has_disallowed_tokens(s):
            return False
        if not looks_like_text(s):
            return False
        return True

    return DatasetDict({k: v.filter(_ok) for k, v in dsd2.items()})


def apply_audio_text_filters(
    dsd: DatasetDict,
    audio_col="audio",
    sent_col="sentence",
    sr=16000,
    min_ms=500,
    max_ms=30000,
    min_chars=1,
    max_chars=200
) -> DatasetDict:
    """
    오디오 길이 필터 + 텍스트 품질 필터를 한 번에 적용하는 편의 함수.

    내부 처리 순서:
      1. 오디오 길이(ms) 컬럼 추가
      2. 길이 기반 필터링 (500ms ~ 30s)
      3. 문장 정규화 및 품질 필터링
    """
    dsd = add_duration_ms(dsd, audio_col=audio_col)
    dsd = filter_by_duration(dsd, min_ms=min_ms, max_ms=max_ms)
    dsd = add_sentence_norm_and_filter(dsd, sent_col=sent_col,
                                       min_chars=min_chars, max_chars=max_chars)
    return dsd


# ============================================================
# 3) DataCollator - 배치 패딩 및 라벨 마스킹
# ============================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Whisper Seq2Seq 학습용 커스텀 DataCollator.

    배치 내 샘플들의 길이가 제각각이므로, 최대 길이에 맞춰 패딩하고
    loss 계산에서 패딩 위치를 무시하도록 -100으로 마스킹한다.

    처리 과정:
      1. input_features(log-mel): feature_extractor.pad()로 패딩
      2. labels(토큰 ID): tokenizer.pad()로 패딩 → 패딩 위치를 -100으로 마스킹
      3. decoder_input_ids: labels를 right-shift하여 생성
         [<|startoftranscript|>] + labels[:-1]

    Attributes:
        processor: WhisperProcessor (feature_extractor + tokenizer 포함)
        decoder_start_token_id: 디코더 시작 토큰 ID (<|startoftranscript|>)
    """
    processor: Any                      # WhisperProcessor
    decoder_start_token_id: int         # 디코더 시작 토큰 (반드시 외부에서 주입)

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        배치 데이터를 모델 입력 형태로 변환한다.

        Args:
            features: DataLoader가 전달하는 샘플 리스트
                      각 샘플: {"input_features": ndarray, "labels": list[int]}

        Returns:
            {"input_features": Tensor, "labels": Tensor, "decoder_input_ids": Tensor}
        """
        # --- 1) 오디오 → input_features 패딩 ---
        # 각 샘플의 input_features를 추출하여 동일 길이로 패딩
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # --- 2) 텍스트 라벨 → 패딩 후 -100 마스킹 ---
        # torch Tensor면 list로 변환 (tokenizer.pad()가 list를 기대)
        def to_list_ids(x):
            return x.tolist() if isinstance(x, torch.Tensor) else x

        label_features = [{"input_ids": to_list_ids(f["labels"])} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"]
        attention_mask = labels_batch["attention_mask"]
        # 패딩 위치(attention_mask == 0)의 라벨을 -100으로 설정
        # → CrossEntropyLoss가 해당 위치를 loss 계산에서 자동 제외
        labels = labels.masked_fill(attention_mask.eq(0), -100).to(torch.long)

        # --- 3) decoder_input_ids 생성 (teacher forcing용 shift-right) ---
        tok = self.processor.tokenizer
        # pad_token이 없으면 eos_token으로 대체 (방어 코드)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        start_id = self.decoder_start_token_id if self.decoder_start_token_id is not None else pad_id

        # -100 마스크를 pad_id로 복원하여 shift 연산에 사용
        labels_for_shift = labels.clone()
        labels_for_shift[labels_for_shift == -100] = pad_id

        # shift-right: [start_token] + labels[:-1]
        # 디코더는 이전 토큰을 입력으로 받아 다음 토큰을 예측
        decoder_input_ids = torch.full((labels_for_shift.size(0), 1), start_id, dtype=torch.long)
        decoder_input_ids = torch.cat([decoder_input_ids, labels_for_shift[:, :-1]], dim=1)

        # 텐서 형상 검증 (디버깅용 방어적 assertion)
        assert decoder_input_ids is not None and decoder_input_ids.ndim == 2, "decoder_input_ids malformed"
        assert "input_features" in batch and batch["input_features"].ndim == 3, "input_features missing or malformed"
        assert labels is not None and labels.ndim == 2, "labels malformed"

        batch["decoder_input_ids"] = decoder_input_ids
        batch["labels"] = labels
        return batch


# ============================================================
# 4) 배치 전처리 함수 - 오디오→log-mel, 텍스트→토큰 ID
# ============================================================

def prepare_dataset_with_processor(
    batch: Dict[str, List[Any]],
    processor: WhisperProcessor,
    audio_col: str = "audio",
    text_col: str = "sentence",
    feature_dtype: str = "float32",
    max_target_len: int = None,
    truncation: bool = True
) -> Dict[str, List[Any]]:
    """
    Dataset.map(batched=True)에 전달되는 배치 전처리 함수.

    각 배치에 대해:
      1. 오디오 배열 추출 → WhisperProcessor로 log-mel spectrogram 생성
      2. 전사 텍스트 → WhisperTokenizer로 토큰 ID 시퀀스 생성

    log-mel spectrogram: 80개 mel 필터 뱅크 × T 프레임
    Whisper는 30초 오디오를 480,000 샘플(16kHz) → 3000 프레임으로 변환

    Args:
        batch: 배치 딕셔너리 (audio_col, text_col 키 포함)
        processor: WhisperProcessor
        audio_col: 오디오 컬럼명
        text_col: 텍스트 컬럼명
        feature_dtype: 특징 벡터 데이터 타입 ("float32" 권장)
        max_target_len: 최대 토큰 길이 (None이면 제한 없음, 필요 시 448 등 설정)
        truncation: 토큰 길이 초과 시 잘라내기 여부

    Returns:
        {"input_features": List[ndarray], "labels": List[List[int]]}
    """
    # 빈 배치에 대한 방어 처리
    if not batch.get(audio_col):
        return {"input_features": [], "labels": []}

    # 1) 오디오 numpy 배열 추출
    audio_arrays = [a["array"] for a in batch[audio_col]]
    sampling_rate = TARGET_SR  # Whisper 표준 16kHz

    # 2) 오디오 → log-mel spectrogram 변환
    # return_tensors=None: 텐서 대신 numpy 배열로 반환 (Arrow 캐시 호환)
    inputs = processor(
        audio_arrays,
        sampling_rate=sampling_rate,
        return_tensors=None,
    )
    feats = inputs["input_features"]
    # 지정된 dtype으로 변환 (메모리 절약이 필요하면 float16 사용 가능)
    if feature_dtype == "float16":
        feats = [f.astype("float16") for f in feats]
    else:
        feats = [f.astype("float32") for f in feats]

    # 3) 텍스트 → 토큰 ID 시퀀스
    tok_kwargs = dict(padding="longest", truncation=truncation, return_tensors=None)
    if max_target_len is not None:
        tok_kwargs["max_length"] = max_target_len

    labels = processor.tokenizer(batch[text_col], **tok_kwargs)["input_ids"]

    return {
        "input_features": feats,    # List[np.ndarray], 각 원소 shape: (80, T)
        "labels": labels,           # List[List[int]], 각 원소: 토큰 ID 시퀀스
    }


# ============================================================
# 5) DatasetDict 전체에 전처리 일괄 적용
# ============================================================

def preprocess_datasetdict_batched(
    dsd: DatasetDict,
    processor: WhisperProcessor,
    audio_col: str = "audio",
    text_col: str = "sentence",
    num_proc: int = None,
    batch_size: int = 32,
    remove_others: bool = False,
    feature_dtype: str = "float32",
    max_target_len: int = None,
    truncation: bool = True
) -> DatasetDict:
    """
    DatasetDict의 모든 split에 Whisper 전처리를 일괄 적용한다.

    처리 과정:
      1. audio 컬럼을 Audio 타입으로 캐스팅 (자동 디코딩 + 리샘플링 보장)
      2. Dataset.map()을 batched 모드로 호출하여 log-mel 추출 + 토크나이징
      3. (선택) input_features/labels 외 컬럼 제거

    Args:
        dsd: 원본 DatasetDict
        processor: WhisperProcessor
        audio_col: 오디오 컬럼명
        text_col: 텍스트 컬럼명
        num_proc: 병렬 프로세스 수 (None이면 단일 프로세스, I/O 병목 시 2~4 권장)
        batch_size: map 함수에 한 번에 전달되는 샘플 수
        remove_others: True면 input_features/labels만 남기고 나머지 컬럼 제거
        feature_dtype: 특징 벡터 dtype ("float32" 권장)
        max_target_len: 최대 토큰 길이 제한
        truncation: 토큰 길이 초과 시 잘라내기 여부

    Returns:
        전처리가 완료된 DatasetDict (input_features, labels 컬럼 추가)
    """
    # Audio 타입 캐스팅: 파일 경로 → 자동 디코딩 + TARGET_SR로 리샘플링
    dsd = dsd.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))

    # batched map으로 전처리 적용
    # batched=True: 개별 샘플이 아닌 batch_size개 샘플을 한 번에 처리 (효율적)
    dsd_proc = dsd.map(
        lambda b: prepare_dataset_with_processor(
            b, processor,
            audio_col=audio_col, text_col=text_col,
            feature_dtype=feature_dtype,
            max_target_len=max_target_len,
            truncation=truncation
        ),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Preparing Whisper features"
    )

    # 학습에 필요한 컬럼만 남기기 (선택 사항)
    if remove_others:
        keep = {"input_features", "labels"}
        dsd_proc = DatasetDict({
            k: v.remove_columns([c for c in v.column_names if c not in keep])
            for k, v in dsd_proc.items()
        })

    return dsd_proc
