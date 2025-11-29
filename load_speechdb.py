"""
EnumaSpeech_01 디렉토리에 저장되어있는 train, valid, test 데이터셋을 
load_dataset이 로딩 과정까지.
1. csv 형태
2. parquet(pandas 사용)
3. parquet(메모리 효율버전)
"""

import csv
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Literal
from datasets import load_dataset, Audio, DatasetDict
from loggerinterface import get_logger

logger = get_logger(__name__)

"""
공통함수: 특정 디렉토리 하위 디렉토리까지 검색하여 특정 파일확장자 리스트 확보
"""
def index_files(rootpath: Path, extension: str) -> Dict[str, Path]:
    """split 루트 하위 모든 .wav를 stem → Path로 인덱싱"""
    
    # 1. 결과를 담을 빈 딕셔너리 생성
    # 예: {'file1': Path('/path/to/file1.wav'), 'file2': Path('/path/to/file2.wav')}
    idxs: Dict[str, Path] = {}
    
    # 2. rootpath와 그 하위 모든 폴더를 재귀적으로 탐색
    # extension 패턴과 일치하는 모든 파일(p)을 순회
    for p in rootpath.rglob(extension):
        
        # 3. 딕셔너리에 파일 정보 추가
        #    - p.stem: 확장자를 제외한 파일 이름 (예: 'sound.wav' -> 'sound')
        #    - p.resolve(): 파일의 전체 절대 경로 (예: './sound.wav' -> '/home/user/project/sound.wav')
        idxs[p.stem] = p.resolve()
        
    # 4. 완성된 딕셔너리 반환
    return idxs

def parse_tran_line(line: str) -> Tuple[str, str]:
    """한 줄을 (오디오 key, 전사)로 파싱: 첫 공백 전까지를 key, 나머지는 전사"""
    s = line.strip()
    if not s: 
        raise ValueError("empty line")
    m = re.match(r"^(\S+)\s+(.*)$", s)
    if not m:
        raise ValueError(f"cannot parse: {s}")
    key, text = m.group(1), m.group(2).strip()
    # 확장자 붙어있다면 제거
    key = os.path.splitext(key)[0]
    return key, text

def split_key_parts(key: str) -> Tuple[str, str, str]:
    """예상 패턴: speaker-chapter-utt → (speaker_id, chapter_id, utt_id)"""
    parts = key.split("-")
    spk = parts[0] if len(parts) > 0 else ""
    chap = parts[1] if len(parts) > 1 else ""
    utt = parts[2] if len(parts) > 2 else ""
    return spk, chap, utt

def build_parquet_for_split(split_root: Path, output_path: Path, stype: Literal["csv","pandas","stream"])-> int:
    """
    split_root로 train, valid, test
    output_path로 각 split_root 정보가 저장된...
    Literal["csv","pandas","stream"] 형태의 포맷 지정
    """
    
    # 경로를 사전 형태의리스트로 획득 {'file1': Path('/path/to/file1.wav'), 'file2': Path('/path/to/file2.wav')}
    # wav파일은 여러개, tran_idxs는 하나?
    wav_idxs = index_files(split_root,"*.wav")
    tran_idxs = index_files(split_root,"*.txt")
    
    rows: List[Dict[str,str]] = []
    
    # 특정 디렉토리(train, valid, test)로 들어온 데이터에 대하여 trans 파일(각 화자별-챕터별 등)을 읽고 idx와 script 분리
    for key in tran_idxs.keys():
        tranpath = tran_idxs[key]
        with tranpath.open('r',encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                
                try:
                    key, text = parse_tran_line(ln)
                except Exception:
                    continue
                
                path = wav_idxs.get(key)
                if path is None:
                    cands = list(tran.parent.glob(key + ".wav"))
                    path = cands[0].resolve() if cands else None
                if not path:
                    continue

                # key가 "speaker-chapter-utt" 형태라는 가정
                parts = key.split("-")
                rows.append({
                    "audio": str(path),
                    "sentence": text,
                    "speaker_id": parts[0] if len(parts) > 0 else "",
                    "chapter_id": parts[1] if len(parts) > 1 else "",
                    "utt_id":    parts[2] if len(parts) > 2 else "",
                    "key": key,
                })

    if stype == "csv":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["audio","sentence","speaker_id","chapter_id","utt_id","key"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return len(rows)
        
    elif stype == "pandas":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 문자열 컬럼으로 고정 (혼합 타입 방지)
        df = pd.DataFrame(rows, dtype="string")
        # 압축은 zstd 권장(작고 빠름); snappy도 OK
        df.to_parquet(output_path, index=False, compression="zstd")
        return len(rows)        

def build_and_load_enuma_dataset_basedon_csv(root_dir: str, 
                                             cache_dir: str = None, 
                                             sampling_rate: int = 16000) -> DatasetDict:
    """
    지정한 루트 디렉토리의 wav 파일과 txt 파일을 검색하여
    csv파일 생성

    Args:
        root_dir (str): _description_
        cache_dir (str, optional): _description_. Defaults to None.
        sampling_rate (int, optional): _description_. Defaults to 16000.

    Returns:
        DatasetDict: _description_
    """
    # 디렉토리 지정(기존 데이터베이스에 train, valid, test)음성데이터베이스 
    root = Path(root_dir).resolve()             # root_dir(상대경로) --> 절대경로의 Path 객체로 전달.
    train_root = root / "enuma_clean_train"
    valid_root = root / "enuma_clean_valid"
    test_root  = root / "enuma_clean_test"

    mani_dir = root / "_manifests"
    train_csv = mani_dir / "train.csv"
    valid_csv = mani_dir / "validation.csv"
    test_csv  = mani_dir / "test.csv"

    # 
    n_tr = build_parquet_for_split(train_root, train_csv, "csv")
    n_va = build_parquet_for_split(valid_root, valid_csv, "csv")
    n_te = build_parquet_for_split(test_root,  test_csv, "csv")
    logger.info(f"manifest → train:{n_tr}, valid:{n_va}, test:{n_te}")

    data_files = {"train": str(train_csv), "validation": str(valid_csv), "test": str(test_csv)}
    ds = load_dataset("csv", data_files=data_files, cache_dir=None)

    # 오디오 경로 문자열 → Audio 타입(지연 디코딩)으로 변환 + 16kHz로 캐스팅
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds    

def build_and_load_enuma_dataset_basedon_parquet_with_pandas(root_dir: str, cache_dir: str = None, sr: int = 16000) -> DatasetDict:
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


import os, re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import pyarrow as pa
import pyarrow.parquet as pq

def rows_generator(split_root: Path) -> Iterable[dict]:
    wav_idxs = index_files(split_root,"*.wav")
    trn_idxs = index_files(split_root,"*.txt")
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
    # 스키마 고정(전부 문자열) → 타입 흔들림 방지
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
    buf = []
    n_total = 0
    try:
        for row in rows_generator(split_root):
            buf.append(row)
            if len(buf) >= chunk_size:
                batch = pa.Table.from_pylist(buf, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(out_parquet, schema, compression="zstd")
                writer.write_table(batch)
                n_total += len(buf)
                buf.clear()
        if buf:
            batch = pa.Table.from_pylist(buf, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet, schema, compression="zstd")
            writer.write_table(batch)
            n_total += len(buf)
    finally:
        if writer is not None:
            writer.close()
    return n_total
    