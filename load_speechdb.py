"""
load_speechdb.py - 음성 데이터베이스 로더

EnumaSpeech 디렉토리에 저장된 train/valid/test 음성 데이터를
HuggingFace DatasetDict으로 변환하는 모듈.

지원하는 변환 방식:
  1. CSV 기반: wav/txt → CSV manifest → load_dataset("csv")
  2. Parquet (pandas): wav/txt → Parquet(zstd 압축) → load_dataset("parquet")
  3. Parquet (PyArrow 스트리밍): 대용량 데이터를 청크 단위로 처리하여 메모리 효율적

데이터 디렉토리 구조 (EnumaSpeech_01):
  enuma_clean_train/   ← 학습 데이터
  enuma_clean_valid/   ← 검증 데이터
  enuma_clean_test/    ← 테스트 데이터
  각 디렉토리에 .wav(오디오)와 .txt(전사 텍스트) 파일 포함

전사 파일(.txt) 포맷:
  speaker-chapter-utt <transcription text>
  (한 줄에 하나의 발화, 첫 토큰이 key, 나머지가 전사 텍스트)
"""

import csv
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Optional
from datasets import load_dataset, Audio, DatasetDict
from loggerinterface import get_logger

logger = get_logger(__name__)


# ============================================================
# HuggingFace Hub 데이터셋 로더
# ============================================================

def load_from_huggingface(
    dataset_id: str,
    subset: Optional[str] = None,
    audio_column: str = "audio",
    text_column: str = "sentence",
    sampling_rate: int = 16000,
) -> DatasetDict:
    """
    HuggingFace Hub에서 음성 데이터셋을 로드한다.

    Hub의 데이터셋은 이미 DatasetDict 형태로 제공되므로 manifest 생성이 불필요하다.
    오디오/텍스트 컬럼명이 다를 수 있으므로, 내부적으로 "audio"/"sentence"로 통일한다.

    Args:
        dataset_id: Hub 데이터셋 ID (예: "mozilla-foundation/common_voice_17_0")
        subset: 데이터셋 서브셋/언어 코드 (예: "en", "ko"). None이면 기본 서브셋
        audio_column: 원본 데이터셋의 오디오 컬럼명
        text_column: 원본 데이터셋의 전사 텍스트 컬럼명
        sampling_rate: 타깃 샘플레이트 (기본 16000Hz)

    Returns:
        DatasetDict: "audio", "sentence" 컬럼을 가진 데이터셋
    """
    logger.info(f"HuggingFace Hub에서 데이터셋 로드: {dataset_id} (subset={subset})")

    if subset:
        ds = load_dataset(dataset_id, subset)
    else:
        ds = load_dataset(dataset_id)

    # 컬럼명이 표준("audio", "sentence")과 다르면 리네임하여 파이프라인 호환성 보장
    if audio_column != "audio":
        ds = ds.rename_column(audio_column, "audio")
        logger.info(f"오디오 컬럼 리네임: '{audio_column}' → 'audio'")
    if text_column != "sentence":
        ds = ds.rename_column(text_column, "sentence")
        logger.info(f"텍스트 컬럼 리네임: '{text_column}' → 'sentence'")

    # 오디오 컬럼을 Audio 타입으로 캐스팅 (16kHz 리샘플링)
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    logger.info(f"HuggingFace 데이터셋 로드 완료: {ds}")
    return ds


# ============================================================
# 공통 유틸리티 함수
# ============================================================

def index_files(rootpath: Path, extension: str) -> Dict[str, Path]:
    """
    지정 디렉토리 하위를 재귀 탐색하여 특정 확장자 파일을 인덱싱한다.

    파일명(확장자 제외)을 key, 절대 경로를 value로 하는 딕셔너리를 반환.
    동일 stem의 파일이 여러 개 있으면 마지막 것이 사용된다.

    Args:
        rootpath: 탐색 시작 디렉토리
        extension: glob 패턴 (예: "*.wav", "*.txt")

    Returns:
        {파일명(stem): 절대경로(Path)} 딕셔너리
        예: {'spk1-ch1-001': Path('/data/spk1-ch1-001.wav')}
    """
    idxs: Dict[str, Path] = {}

    # rglob: rootpath 하위 모든 디렉토리를 재귀적으로 탐색
    for p in rootpath.rglob(extension):
        # p.stem: 확장자 제외 파일명, p.resolve(): 절대 경로로 변환
        idxs[p.stem] = p.resolve()

    return idxs


def parse_tran_line(line: str) -> Tuple[str, str]:
    """
    전사 파일의 한 줄을 (오디오 key, 전사 텍스트)로 파싱한다.

    포맷: "speaker-chapter-utt This is the transcription"
    첫 공백 전까지가 key, 나머지가 전사 텍스트.
    key에 확장자(.wav 등)가 붙어 있으면 자동 제거.

    Args:
        line: 전사 파일의 한 줄 문자열

    Returns:
        (key, text) 튜플

    Raises:
        ValueError: 빈 줄이거나 파싱 불가능한 포맷
    """
    s = line.strip()
    if not s:
        raise ValueError("empty line")
    m = re.match(r"^(\S+)\s+(.*)$", s)
    if not m:
        raise ValueError(f"cannot parse: {s}")
    key, text = m.group(1), m.group(2).strip()
    # key에 확장자가 붙어 있다면 제거 (예: "spk1-ch1-001.wav" → "spk1-ch1-001")
    key = os.path.splitext(key)[0]
    return key, text


def split_key_parts(key: str) -> Tuple[str, str, str]:
    """
    오디오 key를 speaker_id, chapter_id, utt_id로 분리한다.

    예상 포맷: "speaker-chapter-utt" (하이픈으로 구분)
    파트가 부족하면 빈 문자열로 채운다.

    Args:
        key: 오디오 파일 key (예: "1234-5678-0001")

    Returns:
        (speaker_id, chapter_id, utt_id) 튜플
    """
    parts = key.split("-")
    spk = parts[0] if len(parts) > 0 else ""
    chap = parts[1] if len(parts) > 1 else ""
    utt = parts[2] if len(parts) > 2 else ""
    return spk, chap, utt


# ============================================================
# 방식 1: CSV 기반 manifest 생성
# ============================================================

def build_parquet_for_split(split_root: Path, output_path: Path, stype: Literal["csv","pandas","stream"]) -> int:
    """
    하나의 split(train/valid/test) 디렉토리를 스캔하여 manifest 파일을 생성한다.

    처리 과정:
      1. split_root에서 모든 .wav와 .txt 파일을 인덱싱
      2. 전사 파일(.txt)을 한 줄씩 읽어 (key, text) 추출
      3. key에 대응하는 wav 파일 경로를 매칭
      4. 지정된 포맷(csv/pandas)으로 저장

    manifest 컬럼: audio(경로), sentence(전사), speaker_id, chapter_id, utt_id, key

    Args:
        split_root: split 디렉토리 경로 (예: enuma_clean_train/)
        output_path: 출력 파일 경로 (예: _manifests/train.csv)
        stype: 출력 포맷 ("csv" 또는 "pandas"로 parquet)

    Returns:
        생성된 행(row) 수
    """
    # split_root 하위의 모든 wav/txt 파일을 {stem: Path} 형태로 인덱싱
    wav_idxs = index_files(split_root, "*.wav")
    tran_idxs = index_files(split_root, "*.txt")

    rows: List[Dict[str, str]] = []

    # 각 전사 파일을 순회하며 (key, text) → wav 경로 매칭
    for key in tran_idxs.keys():
        tranpath = tran_idxs[key]
        with tranpath.open('r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue

                try:
                    key, text = parse_tran_line(ln)
                except Exception:
                    continue

                # key로 wav 파일 경로 검색
                path = wav_idxs.get(key)
                if path is None:
                    # 인덱스에 없으면 전사 파일과 같은 디렉토리에서 직접 탐색
                    cands = list(tran.parent.glob(key + ".wav"))
                    path = cands[0].resolve() if cands else None
                if not path:
                    continue

                # key를 speaker-chapter-utt 구조로 분리하여 메타데이터 저장
                parts = key.split("-")
                rows.append({
                    "audio": str(path),
                    "sentence": text,
                    "speaker_id": parts[0] if len(parts) > 0 else "",
                    "chapter_id": parts[1] if len(parts) > 1 else "",
                    "utt_id":    parts[2] if len(parts) > 2 else "",
                    "key": key,
                })

    # 지정 포맷으로 저장
    if stype == "csv":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["audio", "sentence", "speaker_id", "chapter_id", "utt_id", "key"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return len(rows)

    elif stype == "pandas":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 모든 컬럼을 문자열로 고정하여 혼합 타입 방지
        df = pd.DataFrame(rows, dtype="string")
        # zstd 압축: snappy 대비 압축률 높고 속도도 준수
        df.to_parquet(output_path, index=False, compression="zstd")
        return len(rows)


def build_and_load_enuma_dataset_basedon_csv(root_dir: str,
                                             cache_dir: str = None,
                                             sampling_rate: int = 16000) -> DatasetDict:
    """
    EnumaSpeech 음성 DB를 CSV manifest 기반으로 DatasetDict으로 변환한다.

    처리 과정:
      1. train/valid/test 각 디렉토리에서 CSV manifest 생성
      2. HuggingFace load_dataset("csv")로 DatasetDict 로드
      3. audio 컬럼을 Audio 타입으로 캐스팅 (지연 디코딩 + 16kHz 리샘플링)

    Args:
        root_dir: EnumaSpeech DB 루트 디렉토리 경로
        cache_dir: HuggingFace 캐시 디렉토리 (미사용, None)
        sampling_rate: 타깃 샘플레이트 (기본 16000Hz)

    Returns:
        DatasetDict: {"train", "validation", "test"} 키를 가진 데이터셋
    """
    # 원본 음성 데이터 디렉토리 경로 설정
    root = Path(root_dir).resolve()
    train_root = root / "enuma_clean_train"
    valid_root = root / "enuma_clean_valid"
    test_root  = root / "enuma_clean_test"

    # CSV manifest 저장 경로 (_manifests 디렉토리 하위)
    mani_dir = root / "_manifests"
    train_csv = mani_dir / "train.csv"
    valid_csv = mani_dir / "validation.csv"
    test_csv  = mani_dir / "test.csv"

    # 각 split별 CSV manifest 생성
    n_tr = build_parquet_for_split(train_root, train_csv, "csv")
    n_va = build_parquet_for_split(valid_root, valid_csv, "csv")
    n_te = build_parquet_for_split(test_root,  test_csv, "csv")
    logger.info(f"manifest → train:{n_tr}, valid:{n_va}, test:{n_te}")

    # CSV 파일로부터 HuggingFace DatasetDict 로드
    data_files = {"train": str(train_csv), "validation": str(valid_csv), "test": str(test_csv)}
    ds = load_dataset("csv", data_files=data_files, cache_dir=None)

    # audio 컬럼: 문자열(파일 경로) → Audio 타입으로 캐스팅
    # Audio 타입은 접근 시점에 wav를 디코딩하고 16kHz로 자동 리샘플링
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


# ============================================================
# 방식 2: Parquet 기반 (pandas 사용)
# ============================================================

def build_and_load_enuma_dataset_basedon_parquet_with_pandas(root_dir: str, cache_dir: str = None, sr: int = 16000) -> DatasetDict:
    """
    CSV 대신 Parquet 포맷으로 manifest를 생성하는 대안 함수.

    Parquet은 CSV 대비 다음과 같은 장점이 있다:
      - 컬럼형 저장으로 읽기 성능 우수
      - zstd 압축으로 파일 크기 절약
      - 타입 정보가 파일에 내장됨

    Args:
        root_dir: EnumaSpeech DB 루트 디렉토리
        cache_dir: HuggingFace 캐시 디렉토리
        sr: 타깃 샘플레이트

    Returns:
        DatasetDict
    """
    root = Path(root_dir).resolve()
    mani = root / "_manifests"
    mani.mkdir(parents=True, exist_ok=True)

    n_tr = build_parquet_for_split(root/"enuma_clean_train", mani/"train.parquet")
    n_va = build_parquet_for_split(root/"enuma_clean_valid", mani/"validation.parquet")
    n_te = build_parquet_for_split(root/"enuma_clean_test",  mani/"test.parquet")
    print(f"manifest → train:{n_tr}, valid:{n_va}, test:{n_te}")

    data_files = {"train": str(mani/"train.parquet"), "validation":str(mani/"validation.parquet"), "test":str(mani/"test.parquet")}
    ds = load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=cache_dir,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=sr))
    return ds


# ============================================================
# 방식 3: PyArrow 스트리밍 (메모리 효율 버전)
# ============================================================

import os, re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import pyarrow as pa
import pyarrow.parquet as pq


def rows_generator(split_root: Path) -> Iterable[dict]:
    """
    split 디렉토리에서 (wav, txt) 쌍을 읽어 행 딕셔너리를 yield하는 제너레이터.

    모든 데이터를 메모리에 한 번에 올리지 않고, 한 줄씩 yield하여
    대용량 데이터에서도 메모리를 절약한다.
    build_parquet_for_split_arrow()에서 청크 단위로 소비된다.

    Args:
        split_root: split 디렉토리 경로

    Yields:
        {"audio", "sentence", "speaker_id", "chapter_id", "utt_id", "key"} 딕셔너리
    """
    wav_idxs = index_files(split_root, "*.wav")
    trn_idxs = index_files(split_root, "*.txt")

    for key in trn_idxs.keys():
        tran = trn_idxs[key]
        with tran.open("r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                m = re.match(r"^(\S+)\s+(.*)$", s)
                if not m:
                    continue
                key, text = os.path.splitext(m.group(1))[0], m.group(2).strip()
                path = wav_idxs.get(key)
                if path is None:
                    cands = list(tran.parent.glob(key + ".wav"))
                    path = cands[0].resolve() if cands else None
                if not path:
                    continue
                # key를 하이픈으로 분리, 부족한 파트는 빈 문자열로 채움
                spk, chap, utt = (key.split("-")+["","",""])[:3]
                yield {
                    "audio": str(path),
                    "sentence": text,
                    "speaker_id": spk,
                    "chapter_id": chap,
                    "utt_id": utt,
                    "key": key,
                }


def build_parquet_for_split_arrow(split_root: Path, out_parquet: Path, chunk_size: int = 20000) -> int:
    """
    PyArrow를 직접 사용하여 Parquet 파일을 청크 단위로 생성한다.

    pandas를 거치지 않고 PyArrow의 ParquetWriter를 사용하여
    chunk_size개 행씩 버퍼링 후 기록한다.
    대용량 데이터(수백만 행)에서 메모리 사용량을 일정 수준으로 유지할 수 있다.

    스키마를 명시적으로 고정(전부 string)하여, 빈 값이나 숫자처럼 보이는
    문자열에 의한 타입 추론 오류를 방지한다.

    Args:
        split_root: split 디렉토리 경로
        out_parquet: 출력 Parquet 파일 경로
        chunk_size: 한 번에 버퍼링할 행 수 (기본 20,000)

    Returns:
        총 기록된 행 수
    """
    # 모든 컬럼을 문자열로 고정하여 타입 불일치 방지
    schema = pa.schema([
        ("audio", pa.string()),
        ("sentence", pa.string()),
        ("speaker_id", pa.string()),
        ("chapter_id", pa.string()),
        ("utt_id", pa.string()),
        ("key", pa.string()),
    ])
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    buf = []       # 행 버퍼
    n_total = 0
    try:
        for row in rows_generator(split_root):
            buf.append(row)
            # 버퍼가 chunk_size에 도달하면 Parquet에 기록
            if len(buf) >= chunk_size:
                batch = pa.Table.from_pylist(buf, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(out_parquet, schema, compression="zstd")
                writer.write_table(batch)
                n_total += len(buf)
                buf.clear()
        # 남은 버퍼 기록
        if buf:
            batch = pa.Table.from_pylist(buf, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet, schema, compression="zstd")
            writer.write_table(batch)
            n_total += len(buf)
    finally:
        # ParquetWriter는 반드시 close()해야 파일이 올바르게 닫힘
        if writer is not None:
            writer.close()
    return n_total
