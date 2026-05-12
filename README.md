# Whisper Fine-Tuning

OpenAI Whisper 모델(`whisper-tiny`)을 커스텀 음성 데이터셋으로 파인튜닝하는 프로젝트입니다.  
Hugging Face `transformers`와 `datasets` 라이브러리 기반으로 구성되어 있습니다.

## 주요 기능

- **데이터 소스 선택**: 로컬 음성 파일(wav/txt) 또는 HuggingFace Hub 데이터셋
- **증강 방식 선택**: 오프라인(사전 생성) 또는 On-the-fly(실시간 랜덤 증강)
- **config.yaml 기반**: 모든 설정을 하나의 파일에서 관리

## 파이프라인 개요

```
데이터 로드 (로컬 or HF Hub)
  → 불필요 컬럼 제거
  → 데이터 증강 (offline or on-the-fly)
  → 전처리 (log-mel + 토크나이징)
  → Seq2Seq 학습 (WER 평가)
```

## 프로젝트 구조

```
whisperfinetuning/
├── config.yaml                      # 전체 설정 파일 (데이터 소스, 증강 방식, 학습 파라미터)
├── config_loader.py                 # YAML 설정 로더 및 검증
├── main.py                          # 학습 진입점 (설정 기반 파이프라인)
├── load_speechdb.py                 # 데이터 로더 (로컬 CSV manifest + HF Hub)
├── augumented_audio.py              # 오디오 증강 (offline + on-the-fly)
├── preprocess_datasetdict_batch.py  # 필터링, 전처리, DataCollator
├── access_whisperobj.py             # Whisper tokenizer, processor, model 로드
├── loggerinterface.py               # 로깅 설정 (콘솔 + 로테이팅 파일)
├── requirements.txt                 # Python 의존성
├── Q&A.md                           # 설계 의사결정 Q&A
└── storage/                         # 데이터셋 캐시 디렉토리
```

## 설정 (config.yaml)

### 데이터 소스 선택

```yaml
data:
  source: "local"          # "local" 또는 "huggingface"

  local:
    db_path: "storage/EnumaSpeech_01"
    raw_cache_dir: "storage/raw_enuma_speech"
    sampling_rate: 16000

  huggingface:
    dataset_id: "mozilla-foundation/common_voice_17_0"
    subset: "en"
    audio_column: "audio"
    text_column: "sentence"
    sampling_rate: 16000
```

### 증강 방식 선택

```yaml
augmentation:
  mode: "offline"          # "offline" (사전 저장) 또는 "on_the_fly" (실시간)

  speed:
    enabled: true
    prob: 0.5
  volume:
    enabled: true
    prob: 0.5
  noise:
    enabled: true
    prob: 0.5
    noise_dir: "/path/to/noise_wavs"
```

| 모드 | 설명 | 장점 | 단점 |
|------|------|------|------|
| `offline` | 증강을 1회 적용 후 디스크 저장 | 빠른 학습 속도 | 고정된 증강, 디스크 2배 |
| `on_the_fly` | 매 에폭 실시간 증강 | 다양한 변형, 디스크 절약 | 약간 느린 데이터 로딩 |

## 데이터셋 구조 (로컬 모드)

`storage/EnumaSpeech_01/` 하위에 다음 디렉토리가 필요합니다:

```
EnumaSpeech_01/
├── enuma_clean_train/    # 학습 데이터 (.wav + .txt)
├── enuma_clean_valid/    # 검증 데이터
└── enuma_clean_test/     # 테스트 데이터
```

전사 파일(.txt) 포맷: `<speaker-chapter-utt> <transcription>` (한 줄에 하나씩)

## 설치

```bash
pip install -r requirements.txt
```

주요 의존성: `torch`, `transformers`, `datasets`, `torchaudio`, `evaluate`, `PyYAML`

## 실행

```bash
python main.py
```

### 주요 학습 설정 (config.yaml)

| 항목 | 기본값 |
|------|--------|
| 베이스 모델 | `openai/whisper-tiny` |
| 에폭 | 4 |
| 배치 크기 | 8 (유효 배치: 16) |
| 학습률 | 1e-4 |
| 평가 주기 | 500 steps |
| 평가 지표 | WER (Word Error Rate) |

## 로깅

- 콘솔 + 로테이팅 파일 로그 (`logs/app.log`)
- 환경 변수 제어: `LOG_LEVEL=DEBUG python main.py`
- TensorBoard: `tensorboard --logdir ./checkpoints/whisper-tiny-en/`
