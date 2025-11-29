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
# -------------------------------------------
# 0) Processor 준비
# -------------------------------------------
MODEL_ID = "openai/whisper-tiny"   # 필요에 맞게 변경 (tiny/base/small...)
LANG     = "ko"                    # 언어
TASK     = "transcribe"            # 혹은 "translate"
processor: WhisperProcessor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANG, task=TASK)
TARGET_SR = processor.feature_extractor.sampling_rate  # 보통 16000
# -------------------------------------------
# 1) Remove column
# -------------------------------------------
def remove_columns_from_datasetdict(common_voice: DatasetDict, protected=("audio", "sentence")) -> Tuple[Optional[DatasetDict], str]:
    """
    필요없는 컬럼을 제거해 데이터 크기를 줄이고, 메모리 사용량 절감, 모델 입력 파이프라인을 단순화
    Args:
    - 다운로드 받은 DatasetDict를 입력
    Returns:
    - DatasetDict 리턴 (필요없는 컬럼이 제거된)
    """
    try:        
        # train 데이터셋에 존재하는 컬럼 내임을 set의 형태로 확보
        present = set(common_voice["train"].column_names)
        # present 셋에서 하나씩 꺼냐, drop_candidates에 존재하는 컬럼 리스트
        cols_to_drop = [c for c in present if c not in protected]
        logger.info(f"🧹 Removed columns: {cols_to_drop}")
        # DatasetDict.remove_columns() 메서드로 일괄 제거 및 새로운 DatasetDict 객체를 반환
        # 메모레이 효율적이며 Arrow 기반으로 즉시 반영
        return common_voice.remove_columns(cols_to_drop), ""
    
    except AttributeError:
        return None, "❌ Error: 입력이 DatasetDict 객체가 아닙니다."
    except ValueError as e:
        return None, f"❌ 컬럼 제거 중 오류 발생: {e}"
    except Exception as e:
        return None, f"⚠️ Unexpected Error: {e}"
    

# -------------------------------------------
# 2) Filtering
# -------------------------------------------
# ===== 0) (필요시) 오디오 컬럼 캐스팅 =====
# 이미 Audio 타입이면 생략 가능
def ensure_audio_cast(dsd: DatasetDict, audio_col="audio", sr=16000) -> DatasetDict:
    return dsd.cast_column(audio_col, Audio(sampling_rate=sr))

# ===== 1) 길이(ms) 계산 (한 번만) =====
def add_duration_ms(dsd: DatasetDict, audio_col="audio") -> DatasetDict:
    def _dur(b):
        a = b[audio_col]
        return {"duration_ms": int(len(a["array"]) * 1000 / a["sampling_rate"])}
    return DatasetDict({k: v.map(_dur) for k, v in dsd.items()})

# ===== 2) 오디오 길이 필터 =====
def filter_by_duration(dsd: DatasetDict, min_ms=500, max_ms=30000) -> DatasetDict:
    def _ok(ex):
        d = ex.get("duration_ms")
        return (d is not None) and (min_ms <= d <= max_ms)
    return DatasetDict({k: v.filter(_ok) for k, v in dsd.items()})

# ===== 3) 문장 정규화 & 문제 텍스트 필터 =====
_DISALLOWED_TOKENS = {
    "[noise]", "<noise>", "(noise)", "[music]", "<music>", "(music)",
    "[laughter]", "<laughter>", "(laughter)", "<unk>", "[unk]"
}

# 한글/영문/숫자 중 하나라도 포함되는지 (내용 존재 판단)
_CONTENT_RE = re.compile(r"[A-Za-z0-9\uAC00-\uD7A3]")

def normalize_sentence(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("…", "...")         # 예: 통일
    s = re.sub(r"\s+", " ", s)        # 연속 공백 정리
    return s.strip()

def has_disallowed_tokens(s: str) -> bool:
    s_low = s.lower()
    return any(tok in s_low for tok in _DISALLOWED_TOKENS)

def looks_like_text(s: str) -> bool:
    # 내용(문자/숫자/한글)이 하나라도 있어야 함
    return bool(_CONTENT_RE.search(s))

def add_sentence_norm_and_filter(
    dsd: DatasetDict,
    sent_col="sentence",
    min_chars=1,
    max_chars=200
) -> DatasetDict:
    # 3-1) 정규화된 문장 추가
    def _norm(b):
        s = normalize_sentence(b.get(sent_col, ""))
        return {"sentence_norm": s}
    dsd2 = DatasetDict({k: v.map(_norm) for k, v in dsd.items()})

    # 3-2) 문제 문장 제거
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

# ===== 4) 한 번에 적용하는 헬퍼 =====
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
    dsd = add_duration_ms(dsd, audio_col=audio_col)
    dsd = filter_by_duration(dsd, min_ms=min_ms, max_ms=max_ms)
    dsd = add_sentence_norm_and_filter(dsd, sent_col=sent_col,
                                       min_chars=min_chars, max_chars=max_chars)
    return dsd


# -------------------------------------------
# 1) DataCollator (패딩 + label -100 마스킹)
#    * transformers의 DataCollatorSpeechSeq2SeqWithPadding을 써도 무방
# -------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any                      # WhisperProcessor
    decoder_start_token_id: int         # 반드시 주입!

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 1) 오디오 -> input_features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 2) 텍스트 라벨 -> padding & -100 마스킹
        def to_list_ids(x):
            return x.tolist() if isinstance(x, torch.Tensor) else x

        label_features = [{"input_ids": to_list_ids(f["labels"])} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"]
        attention_mask = labels_batch["attention_mask"]
        labels = labels.masked_fill(attention_mask.eq(0), -100).to(torch.long)

        # 3) decoder_input_ids 생성 (shift-right)
        tok = self.processor.tokenizer
        # pad가 없다면 eos로 맞춰두기(마지막 방어)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        start_id = self.decoder_start_token_id if self.decoder_start_token_id is not None else pad_id

        labels_for_shift = labels.clone()
        labels_for_shift[labels_for_shift == -100] = pad_id

        # [start_id] + labels[:-1]
        decoder_input_ids = torch.full((labels_for_shift.size(0), 1), start_id, dtype=torch.long)
        decoder_input_ids = torch.cat([decoder_input_ids, labels_for_shift[:, :-1]], dim=1)

        # 방어적 체크(문제 있으면 바로 실패해서 원인 파악 쉬움)
        assert decoder_input_ids is not None and decoder_input_ids.ndim == 2, "decoder_input_ids malformed"
        assert "input_features" in batch and batch["input_features"].ndim == 3, "input_features missing or malformed"
        assert labels is not None and labels.ndim == 2, "labels malformed"

        batch["decoder_input_ids"] = decoder_input_ids
        batch["labels"] = labels
        return batch
# -------------------------------------------
# 2) batched map용 전처리 함수
#    - 오디오 리스트 → log-mel input_features
#    - 문장 리스트 → token id 시퀀스 labels
#    - 반환은 리스트 형태(Arrow 캐시 안정성 ↑)
# -------------------------------------------
def prepare_dataset_with_processor(
    batch: Dict[str, List[Any]],
    processor: WhisperProcessor,
    audio_col: str = "audio",
    text_col: str = "sentence",
    feature_dtype: str = "float32",     # "float32" 권장 (원하면 "float16")
    max_target_len: int = None,         # 필요 시 토큰 길이 제한
    truncation: bool = True
) -> Dict[str, List[Any]]:

    # 빈 배치 방어
    if not batch.get(audio_col):
        return {"input_features": [], "labels": []}

    # 1) 오디오 배열/샘플레이트 추출
    audio_arrays = [a["array"] for a in batch[audio_col]]
    # WhisperProcessor가 자동 리샘플링하므로 sampling_rate는 TARGET_SR 사용
    sampling_rate = TARGET_SR

    # 2) 오디오 → log-mel 특징 (리스트로 반환)
    inputs = processor(
        audio_arrays,
        sampling_rate=sampling_rate,
        return_tensors=None,              # 텐서는 collator에서 처리
        # return_attention_mask=True,     # 필요 시 활성화
    )
    feats = inputs["input_features"]
    if feature_dtype == "float16":
        feats = [f.astype("float16") for f in feats]
    else:
        feats = [f.astype("float32") for f in feats]

    # 3) 텍스트 → token ids (리스트)
    # as_target_processor()를 써도 되지만, tokenizer 직접 호출이 명확합니다.
    tok_kwargs = dict(padding="longest", truncation=truncation, return_tensors=None)
    if max_target_len is not None:
        tok_kwargs["max_length"] = max_target_len

    labels = processor.tokenizer(batch[text_col], **tok_kwargs)["input_ids"]

    return {
        "input_features": feats,    # List[np.ndarray]  (80, T)
        "labels": labels,           # List[List[int]]
        # 디버깅용으로 길이 보고 싶으면 아래 주석 해제
        # "input_length": [f.shape[-1] for f in feats],
    }

# -------------------------------------------
# 3) DatasetDict 전체에 전처리 적용
# -------------------------------------------
def preprocess_datasetdict_batched(
    dsd: DatasetDict,
    processor: WhisperProcessor,
    audio_col: str = "audio",
    text_col: str = "sentence",
    num_proc: int = None,            # CPU 상황에 맞게 (None이면 단일 프로세스)
    batch_size: int = 32,            # batched=True일 때의 map batch 크기
    remove_others: bool = False,     # True면 input_features/labels만 남김
    feature_dtype: str = "float32",
    max_target_len: int = None,
    truncation: bool = True
) -> DatasetDict:

    # Audio 캐스팅 보장 (경로/원시 배열에서 자동 로딩 + 리샘플링)
    dsd = dsd.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))

    """
    Dataset.map(
        function,             # 각 샘플(또는 배치)에 적용할 함수
        batched=False,        # True면 batch 단위로, False면 개별 샘플 단위
        batch_size=1000,      # batched=True일 때 배치 크기
        num_proc=None,        # 병렬 프로세스 수
        remove_columns=None,  # 지정한 컬럼을 제거
        load_from_cache_file=True,  # 이전 결과 캐시 사용
    )
    """
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

    if remove_others:
        keep = {"input_features", "labels"}
        dsd_proc = DatasetDict({
            k: v.remove_columns([c for c in v.column_names if c not in keep])
            for k, v in dsd_proc.items()
        })

    return dsd_proc