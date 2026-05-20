# Q&A

---

## Q1. 데이터 로드 부분에는 로컬에 있는 파일에 대하여 CSV manifest를 생성하는데, 허깅페이스에 있는 데이터셋도 로딩 가능한 구조인가?

### A. config.yaml에서 `data.source`를 선택하여 로컬/HuggingFace 모두 지원한다.

`config.yaml`의 `data.source` 값에 따라 데이터 소스가 결정된다:

```yaml
data:
  source: "local"        # "local" 또는 "huggingface"

  local:
    db_path: "storage/EnumaSpeech_01"
    ...

  huggingface:
    dataset_id: "mozilla-foundation/common_voice_17_0"
    subset: "en"
    audio_column: "audio"
    text_column: "sentence"
    ...
```

#### 두 방식의 차이점

| 항목 | 로컬 (`local`) | HF Hub (`huggingface`) |
|------|---------------|------------------------|
| 로드 방식 | wav/txt 스캔 → CSV manifest → `load_dataset("csv")` | `load_dataset("dataset_id")` 직접 호출 |
| 오디오 컬럼 | `"audio"` (경로 → Audio 캐스팅 필요) | `"audio"` (이미 Audio 타입인 경우 많음) |
| 텍스트 컬럼 | `"sentence"` (고정) | 데이터셋마다 다름 → config에서 지정 |
| 불필요 컬럼 | 적음 (직접 만들었으므로) | 많음 (`accent`, `age` 등) → 자동 제거 |

HF Hub 데이터셋은 컬럼명이 다를 수 있으므로, `audio_column`과 `text_column`을 config에서 지정하면 내부적으로 `"audio"` / `"sentence"`로 자동 리네임하여 이후 파이프라인과 호환된다.

#### 구현 위치

- `load_speechdb.py`: `load_from_huggingface()` 함수 추가
- `main.py`: `load_dataset_from_config()` 함수에서 `data.source` 값에 따라 분기

---

## Q2. 오디오 데이터 증강(speed, volume, noise)은 on the fly로 적용되는가?

### A. config.yaml에서 `augmentation.mode`를 선택하여 offline/on-the-fly 모두 지원한다.

```yaml
augmentation:
  mode: "offline"        # "offline" 또는 "on_the_fly"
```

#### 두 방식의 차이점

| 항목 | `offline` (오프라인) | `on_the_fly` (실시간) |
|------|---------------------|----------------------|
| 적용 시점 | 학습 전 1회 `map()` → 디스크 저장 | 매 샘플 접근 시 `set_transform()`으로 실시간 |
| 랜덤성 | 한 번 생성 후 **고정** | 에폭마다 **다른 변형** 적용 |
| 디스크 사용 | 증강 데이터 별도 저장 (2배 용량) | 원본만 저장 (추가 디스크 불필요) |
| 데이터 다양성 | 샘플당 1가지 변형만 학습 | 에폭마다 다양한 변형 → 일반화 향상 |
| 처리 속도 | 캐시된 데이터 읽기 (빠름) | 매번 증강+log-mel 추출 (약간 느림) |

#### offline 모드 흐름

```
원본 데이터 → map(augment_example) 1회 → 디스크 저장 → map(preprocess) → 학습
               (증강 고정)
```

#### on_the_fly 모드 흐름

```
원본 데이터 → set_transform(augment + preprocess) → DataCollator → 학습
               (매 접근 시 새로운 랜덤 증강)
```

#### 증강 기법별 개별 제어

각 증강 기법은 config에서 독립적으로 활성화/비활성화 및 파라미터 조정이 가능하다:

```yaml
augmentation:
  speed:
    enabled: true        # false로 설정하면 speed 증강 비활성화
    prob: 0.5
    min_speed: 0.9
    max_speed: 1.1
  volume:
    enabled: true
    prob: 0.5
    min_gain_db: -6.0
    max_gain_db: 6.0
  noise:
    enabled: true
    prob: 0.5
    noise_dir: "/path/to/noise_wavs"
    min_snr_db: 5.0
    max_snr_db: 20.0
```

#### 구현 위치

- `augumented_audio.py`: `make_on_the_fly_transform()`, `make_eval_transform()` 함수 추가
- `main.py`: `prepare_datasets_offline()` / `prepare_datasets_on_the_fly()` 분기
- `config.yaml`: `augmentation.mode` 및 각 기법별 설정

---

## Q3. 두 설정을 config.yaml에서 선택적으로 조합할 수 있는가?

### A. 데이터 소스와 증강 방식은 독립적이므로 4가지 조합이 모두 가능하다.

| 데이터 소스 | 증강 방식 | 설정 예시 |
|------------|----------|----------|
| 로컬 파일 | offline | `source: "local"` + `mode: "offline"` |
| 로컬 파일 | on-the-fly | `source: "local"` + `mode: "on_the_fly"` |
| HF Hub | offline | `source: "huggingface"` + `mode: "offline"` |
| HF Hub | on-the-fly | `source: "huggingface"` + `mode: "on_the_fly"` |

두 설정은 파이프라인의 서로 다른 단계를 제어하므로 완전히 독립적으로 동작한다:
- `data.source`: 1단계 (데이터 로드)에만 영향
- `augmentation.mode`: 2단계 (증강 + 전처리)에만 영향

---

## Q4. 홀처럼 울리는 환경의 데이터 증강에는 FIR 필터를 사용하는가? On-the-fly도 가능한가?

### A. FIR 필터가 아니라 RIR(Room Impulse Response) 컨볼루션을 사용한다. On-the-fly도 가능하다.

#### FIR vs RIR 비교

| | FIR 필터 | RIR 컨볼루션 |
|--|---------|-------------|
| 원리 | 주파수 특성을 변형 (EQ, 밴드패스 등) | 실제 공간의 음향 반사 패턴을 적용 |
| 용도 | 주파수 대역 필터링, 톤 변형 | **잔향/울림 시뮬레이션** |
| 핵심 연산 | 둘 다 convolution이지만 목적이 다름 | `output = fftconvolve(clean, rir)` |

FIR 필터도 수학적으로는 convolution이지만, 홀 울림 효과를 만들려면 **실제 공간에서 측정한 RIR 데이터**(또는 합성 RIR)를 사용해야 한다.

#### RIR 컨볼루션 원리

```
깨끗한 음성 ──┐
              ├── fftconvolve ──→ 홀에서 녹음한 것 같은 울림 음성
RIR 임펄스 ───┘
```

- **작은 방**: RIR이 짧고 (수십 ms), 울림이 적음
- **큰 홀/성당**: RIR이 길고 (수 초), 긴 잔향

#### On-the-fly 지원

RIR 컨볼루션은 FFT 기반(`torchaudio.functional.fftconvolve`)으로 수 ms 내에 처리되므로 on-the-fly 성능 병목이 되지 않는다. 노이즈 뱅크와 동일한 방식으로 동작한다:

```
시작 시 RIR wav 파일들을 메모리에 로드 → 매 샘플 접근 시 랜덤 선택 → convolution
```

#### 증강 적용 순서

음향적으로 자연스러운 순서를 따른다:

```
Speed → Volume → Reverb → Noise
```

잔향은 공간 효과이므로 노이즈 혼합 전에 적용해야 현실적이다. (실제로 잔향이 있는 공간에서 배경 노이즈가 추가되는 것이므로)

#### config.yaml 설정

```yaml
augmentation:
  reverb:
    enabled: true              # true로 변경하면 RIR 잔향 증강 활성화
    prob: 0.3                  # 적용 확률 (0.2~0.4 권장, 과도하면 인식률 저하)
    rir_dir: "/path/to/rir_wavs"
```

#### 공개 RIR 데이터셋

| 데이터셋 | 규모 | 특징 |
|---------|------|------|
| MIT IR Survey | 271개 | 다양한 실내/홀 환경 |
| OpenSLR-26 (RWCP) | 수백 개 | 소/중/대규모 방 |
| OpenSLR-28 (simulated) | 60,000개 | 합성 RIR, 대규모 |
| BUT ReverbDB | 다양 | 실측 RIR |

#### 구현 위치

- `augumented_audio.py`: `add_reverb()`, `load_rir_from_dir()` 함수 추가, `augment_waveform()` 및 `make_on_the_fly_transform()`에 reverb 반영
- `main.py`: `prepare_datasets_on_the_fly()`에서 RIR 뱅크 로딩 추가
- `config.yaml`: `augmentation.reverb` 섹션 추가

---

## Q5. 전처리 설정의 `num_proc`과 `batch_size`의 의미는?

### A. 둘 다 `Dataset.map()` 호출 시 전달되는 파라미터로, 전처리 속도를 제어한다.

```yaml
preprocessing:
  num_proc: 4
  batch_size: 32
```

#### `num_proc` — 병렬 프로세스 수

전체 데이터셋을 `num_proc`개로 **분할(shard)**하여 각 프로세스가 독립적으로 처리한다.

```
Dataset (10,000 샘플, num_proc=4)
  ├── Process 0: 샘플 0~2499     → log-mel + tokenize
  ├── Process 1: 샘플 2500~4999  → log-mel + tokenize
  ├── Process 2: 샘플 5000~7499  → log-mel + tokenize
  └── Process 3: 샘플 7500~9999  → log-mel + tokenize
→ 결과 병합
```

| 값 | 동작 |
|----|------|
| `null` (None) | 단일 프로세스 (순차 처리) |
| `2~4` | I/O 병목 시 권장 (오디오 디코딩이 주 병목) |
| `8~12` | CPU 코어가 충분할 때 |

#### `batch_size` — map 함수에 한 번에 전달되는 샘플 수

`Dataset.map(batched=True, batch_size=32)` 에서 사용되는 **전처리 전용** 파라미터이다. 학습 배치 크기(`training.train_batch_size`)와는 별개이다.

| 값 | 영향 |
|----|------|
| 작은 값 (8~16) | 메모리 적게 사용, 함수 호출 오버헤드 증가 |
| 큰 값 (32~64) | WhisperProcessor가 여러 샘플을 한 번에 처리 → 효율적 |
| 너무 큰 값 (256+) | 메모리 부족 위험 (오디오 배열이 크므로) |

#### 두 파라미터의 관계

```
Dataset (10,000 샘플, num_proc=4, batch_size=32)

Process 0 (2,500 샘플 담당):
  ├── batch 1: 샘플 32개 → processor(audio[0:32]) → log-mel + tokenize
  ├── batch 2: 샘플 32개 → processor(audio[32:64]) → ...
  └── ... (총 약 79 배치)

Process 1~3도 동시에 동일 작업 수행
```

- `num_proc`: **프로세스 간 병렬화** (데이터셋을 나눠서)
- `batch_size`: **프로세스 내 벡터화** (한 번에 묶어서)

둘 다 높이면 빠르지만 메모리를 더 사용한다. 일반적으로 `num_proc=4`, `batch_size=32`가 무난한 출발점이다.

---

## Q6. 학습 설정의 각 파라미터(`num_epochs`, `train_batch_size` 등)의 의미는?

### A. `training` 섹션의 전체 파라미터 설명

```yaml
training:
  output_dir: "./checkpoints/whisper-tiny-en"
  num_epochs: 4
  train_batch_size: 8
  eval_batch_size: 8
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  learning_rate: 1.0e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  label_smoothing_factor: 0.1
  eval_steps: 500
  save_steps: 500
  save_total_limit: 3
  logging_steps: 50
  fp16: true
  seed: 42
  dataloader_num_workers: 0
```

#### 배치 및 에폭

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `num_epochs` | 4 | 전체 데이터셋을 몇 번 반복 학습할지. 1 에폭 = 전체 train 데이터를 1회 순회 |
| `train_batch_size` | 8 | GPU 1개당 한 번에 처리하는 샘플 수 |
| `eval_batch_size` | 8 | 평가 시 GPU 1개당 배치 크기 (학습보다 크게 잡아도 됨, gradient 불필요) |
| `gradient_accumulation_steps` | 2 | N스텝의 gradient를 누적 후 1회 업데이트 |

**유효 배치 크기** 계산:

```
유효 배치 = train_batch_size × gradient_accumulation_steps × GPU 수
         = 8 × 2 × 1 = 16
```

GPU 메모리가 부족하면 `train_batch_size`를 줄이고 `gradient_accumulation_steps`를 늘려서 유효 배치를 유지할 수 있다:

```
batch_size=4, accumulation=4 → 유효 배치 16 (같은 효과, 메모리 절반)
```

#### 옵티마이저 및 학습률

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `learning_rate` | 1e-4 | 가중치 업데이트 크기. 너무 크면 발산, 너무 작으면 학습 느림 |
| `warmup_ratio` | 0.1 | 전체 스텝의 10%를 워밍업에 사용 |
| `weight_decay` | 0.01 | L2 정규화 계수 (과적합 방지) |
| `max_grad_norm` | 1.0 | gradient clipping 임계값 (gradient 폭주 방지) |
| `label_smoothing_factor` | 0.1 | 정답 라벨을 부드럽게 하여 과신 방지 |

**warmup 스케줄** 시각화:

```
학습률
  |
1e-4 ┤          ╭────────────────────╮
     │        ╱                       ╲
     │      ╱                          ╲  (cosine/linear decay)
     │    ╱
0    ┤──╱
     └──┬──────┬──────────────────────┬──→ 스텝
        0    10%                     100%
        ← warmup →
```

워밍업 구간에서는 학습률을 0에서 `learning_rate`까지 서서히 올린다. 초반에 큰 학습률로 불안정하게 업데이트되는 것을 방지한다.

**label_smoothing** 동작:

```
smoothing=0.0 → 정답: [0, 0, 1, 0, 0]  (확신 100%)
smoothing=0.1 → 정답: [0.025, 0.025, 0.9, 0.025, 0.025]  (약간의 불확실성 부여)
```

모델이 정답에 100% 확신하는 것을 방지하여 일반화 성능을 향상시킨다.

#### 평가 / 저장 / 로그 주기

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `eval_steps` | 500 | 매 500스텝마다 validation 데이터로 평가 수행 |
| `save_steps` | 500 | 매 500스텝마다 체크포인트 저장 |
| `save_total_limit` | 3 | 최근 3개 체크포인트만 유지 (디스크 절약) |
| `logging_steps` | 50 | 매 50스텝마다 loss 등 학습 로그 출력 |

```
스텝:  50   100  150  ...  500        1000       1500
       │    │    │         │           │           │
       LOG  LOG  LOG  ...  LOG+EVAL   LOG+EVAL   LOG+EVAL
                           +SAVE      +SAVE      +SAVE
```

#### 정밀도 및 기타

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `fp16` | true | FP16 혼합 정밀도 학습. 메모리 절약 + 속도 향상 (CUDA 전용) |
| `seed` | 42 | 랜덤 시드 고정 (재현성 보장) |
| `dataloader_num_workers` | 0 | DataLoader 워커 수 (0=메인 프로세스에서 로드) |
| `output_dir` | `./checkpoints/...` | 체크포인트 및 TensorBoard 로그 저장 경로 |

**fp16 혼합 정밀도**:

```
일반 (FP32):  가중치 32bit, 연산 32bit → 메모리 多, 속도 普通
혼합 (FP16):  가중치 32bit, 연산 16bit → 메모리 약 50% 절약, 속도 향상
```

주의: MPS(Apple Silicon)에서는 `fp16: false`로 설정해야 한다. CUDA GPU 전용 기능이다.

**dataloader_num_workers**:

| 값 | 동작 |
|----|------|
| 0 | 메인 프로세스에서 데이터 로드 (디버깅 용이) |
| 2~4 | 별도 워커 프로세스가 미리 데이터를 준비 (GPU 유휴 시간 감소) |

---

## Q7. 학습 파라미터 튜닝은 훈련 시 오류 방지를 위한 것인가?

### A. 오류 방지가 아니라, 동일한 데이터에서 최대 성능을 뽑아내기 위한 것이다.

성능 향상의 3대 요소:

| | 목적 | 비유 |
|--|------|------|
| 1. 데이터 품질 | 정확한 학습 재료 확보 | 좋은 식재료 |
| 2. 데이터 증강 | 다양한 환경에 대한 일반화 | 다양한 조리법으로 경험 축적 |
| 3. 학습 파라미터 | **최적의 학습 과정 설계** | 불 세기, 시간 조절 |

#### 학습 파라미터가 성능에 미치는 영향

학습 파라미터가 잘못되면 오류(crash)가 나는 것이 아니라, **학습은 정상 완료되지만 성능이 나쁘다**:

```
같은 데이터, 같은 모델이라도:

learning_rate=1e-2  (너무 큼)   → 학습 발산, WER 90%
learning_rate=1e-4  (적절)      → 정상 수렴, WER 15%
learning_rate=1e-7  (너무 작음)  → 학습 안 됨, WER 85%
                                   ↑ 세 경우 모두 오류 없이 완료됨
```

#### 파라미터별 성능 영향

```
                    ┌─ 너무 작으면: GPU 메모리 낭비, 학습 느림
  batch_size ───────┤
                    └─ 너무 크면: 일반화 저하 (sharp minima)

                    ┌─ 너무 적으면: 과소적합 (underfitting)
  num_epochs ───────┤
                    └─ 너무 많으면: 과적합 (overfitting)

                    ┌─ 너무 크면: 발산 (loss 증가)
  learning_rate ────┤
                    └─ 너무 작으면: 수렴 못함 (loss 정체)

                    ┌─ 0이면: 과적합 위험
  weight_decay ─────┤
                    └─ 너무 크면: 과소적합

                    ┌─ 0이면: 초반 학습 불안정
  warmup_ratio ─────┤
                    └─ 너무 크면: 실질 학습 시간 부족
```

#### 정리: 성능 향상의 3단계

```
1단계 (데이터 품질)     → 쓰레기를 넣으면 쓰레기가 나온다 (GIGO)
2단계 (데이터 증강)     → 모델이 다양한 상황에 견디게 한다
3단계 (학습 파라미터)   → 같은 데이터로 최대 성능을 뽑아낸다
```

3단계가 없으면 1, 2단계에서 아무리 좋은 데이터를 준비해도 **모델이 그 데이터에서 효과적으로 배우지 못한다**. 오류 방지가 아니라, 동일한 데이터에서 WER 20% vs 15%의 차이를 만드는 핵심이다.

---

## Q8. 데이터 품질, 데이터 증강, 학습 파라미터 외에 추가적인 성능 개선 방법론은?

### A. 모델 스케일, 학습 기법, 후처리 등 다양한 방법론이 존재한다.

기존 3대 요소(데이터 품질, 데이터 증강, 학습 파라미터) 외에 Whisper 파인튜닝 성능을 추가로 개선할 수 있는 주요 방법론은 다음과 같다.

#### 4. SpecAugment (스펙트로그램 증강)

오디오 원본이 아닌 **log-mel 스펙트로그램 단계에서** 마스킹하는 기법이다.

| 기법 | 설명 |
|------|------|
| **Time Masking** | 시간 축의 일부 구간을 0으로 마스킹 |
| **Frequency Masking** | 주파수 축의 일부 채널을 0으로 마스킹 |

- Whisper 내부에도 이미 일부 적용되어 있지만, 추가 적용 시 과적합 방지에 효과적
- 오디오 증강(speed/noise/reverb)과 **별도 레이어**로 동작하므로 병행 가능

#### 5. 모델 스케일 업

현재 `whisper-tiny` (39M 파라미터)를 사용 중인데, 더 큰 모델로 교체하면 성능이 크게 향상된다.

```
tiny (39M) → base (74M) → small (244M) → medium (769M) → large-v3 (1.5B)
```

- 모델이 클수록 WER이 낮아지지만 학습 시간/GPU 메모리 증가
- `config.yaml`에서 `model.name`만 변경하면 되는 구조

#### 6. LoRA / PEFT (파라미터 효율적 파인튜닝)

전체 모델을 다 학습하지 않고 **소수의 어댑터 파라미터만** 학습하는 방법이다.

| 방식 | 설명 |
|------|------|
| **Full Fine-tuning** | 전체 파라미터 학습 (현재 방식) |
| **LoRA** | Attention 레이어에 저랭크 행렬만 추가 학습 |
| **QLoRA** | 4bit 양자화 + LoRA (GPU 메모리 대폭 절감) |

- 데이터가 적을 때 과적합 방지에 효과적
- `whisper-large-v3` 같은 큰 모델도 단일 GPU에서 학습 가능

#### 7. Pseudo-labeling (자기 학습)

```
[학습된 모델] → 라벨 없는 오디오에 전사 생성 → 높은 신뢰도만 필터링 → 학습 데이터에 추가 → 재학습
```

- 라벨이 없는 대량의 음성 데이터를 활용 가능
- 반복할수록 성능 향상 (Semi-supervised learning)

#### 8. Knowledge Distillation (지식 증류)

```
Teacher (whisper-large-v3) → soft labels 생성 → Student (whisper-tiny) 학습
```

- 큰 모델의 지식을 작은 모델에 전달
- 추론 속도는 유지하면서 성능 향상
- Distil-Whisper가 이 방식으로 만들어짐

#### 9. 후처리 (Post-processing)

모델 출력 이후 텍스트 레벨에서 보정하는 방법이다.

| 기법 | 설명 |
|------|------|
| **Beam Search 튜닝** | `num_beams` 증가 (5~10) → 더 나은 디코딩 |
| **Language Model Rescoring** | 외부 LM으로 후보 문장 재점수화 |
| **텍스트 정규화** | 숫자/약어/특수문자 표준화 |
| **Punctuation 복원** | 별도 모델로 구두점 추가 |

#### 10. Curriculum Learning (교육과정 학습)

```
1단계: 짧고 깨끗한 음성으로 학습
2단계: 점차 긴 음성, 노이즈 음성 추가
3단계: 전체 데이터로 학습
```

- 모델이 쉬운 패턴부터 점진적으로 학습하여 수렴 안정성 향상

#### 영향도 요약

```
[높은 영향]
  ├── 모델 스케일 업 (tiny → small → medium)
  ├── 데이터 양 증가 (Pseudo-labeling)
  └── LoRA + 큰 모델 조합

[중간 영향]
  ├── SpecAugment
  ├── Knowledge Distillation
  └── Beam Search 튜닝

[보완적 영향]
  ├── Curriculum Learning
  ├── 후처리 (LM Rescoring)
  └── 텍스트 정규화
```

#### 현재 프로젝트에의 적용 용이성

| 방법론 | 적용 난이도 | 비고 |
|--------|-----------|------|
| SpecAugment | 쉬움 | log-mel 추출 후 마스킹 추가만 하면 됨 |
| Beam Search 튜닝 | 쉬움 | `generation_config.num_beams` 변경만으로 적용 |
| 모델 스케일 업 | 쉬움 | `config.yaml`에서 `model.name` 변경 (GPU 메모리 필요) |
| LoRA/PEFT | 중간 | `peft` 라이브러리 추가 및 모델 래핑 필요 |
| Pseudo-labeling | 중간 | 추론 파이프라인 + 신뢰도 필터링 로직 구현 필요 |
| Knowledge Distillation | 높음 | Teacher/Student 이중 학습 파이프라인 구성 필요 |
| Curriculum Learning | 중간 | 데이터 난이도 분류 + 단계별 DataLoader 전환 필요 |

---

## Q9. `pyarrow.lib.ArrowNotImplementedError: Unsupported cast from large_string to struct` 오류가 발생하는 이유는?

### A. `load_dataset("csv")` 가 문자열을 `large_string`으로 저장하지만, `Audio.cast_storage()`는 `string` 타입만 허용하기 때문이다.

#### 오류 발생 위치

```
load_speechdb.py
  ds = ds.cast_column("audio", Audio(sampling_rate=16000))
                                ↑
  PyArrow가 large_string → struct 직접 변환을 지원하지 않아 오류 발생
```

#### 원인

`load_dataset("csv", ...)` 는 내부적으로 PyArrow를 사용하며, 문자열 컬럼을 `large_string` 타입으로 저장한다. HuggingFace `Audio` feature의 `cast_storage()`는 `string` 타입 입력만 지원하여 `large_string`을 바로 받으면 오류가 발생한다.

```
CSV 로드 → audio 컬럼 타입: large_string
Audio.cast_storage() 요구 타입: string
→ 직접 변환 불가 → ArrowNotImplementedError
```

#### 해결 방법

`Audio`로 캐스팅하기 전에 `Value("string")`으로 먼저 다운캐스트한다.

```python
# load_speechdb.py
from datasets import Audio, Value

ds = ds.cast_column("audio", Value("string"))        # large_string → string
ds = ds.cast_column("audio", Audio(sampling_rate=sr))  # string → Audio
```

#### 구현 위치

- `load_speechdb.py`: `build_and_load_enuma_dataset_basedon_csv()` 함수 내 캐스팅 2단계로 수정

---

## Q10. `RuntimeError: Could not load libtorchcodec` 오류가 발생하는 이유와 해결책은?

### A. `torchcodec`이 필요로 하는 FFmpeg 공유 라이브러리가 없거나 PyTorch 버전과 호환되지 않기 때문이다.

#### 오류 발생 배경

`datasets` 최신 버전(4.x)은 오디오 디코딩 시 기본적으로 `torchcodec`을 사용한다. `torchcodec`은 FFmpeg의 공유 라이브러리(`libavutil.so.*`)가 시스템에 설치되어 있어야 동작한다.

```
datasets Audio.decode_example()
  └→ torchcodec.AudioDecoder
       └→ FFmpeg 공유 라이브러리 (libavutil.so.*)
            └→ 없으면 RuntimeError
```

#### 증상

- FFmpeg 미설치 환경에서 `datasets.map()` 실행 시 발생
- 멀티프로세싱(`num_proc > 1`) 환경에서는 워커 프로세스 내부에서 발생하여 전체 map이 실패

#### 해결 방법

**방법 1 — FFmpeg 설치 (근본 해결):**
```bash
sudo apt-get install -y ffmpeg
```

**방법 2 — torchcodec 비활성화 (FFmpeg 없이 soundfile로 폴백):**
```bash
pip uninstall torchcodec -y
pip install soundfile
```

**방법 3 — 코드에서 환경변수로 비활성화:**
```python
# main.py 최상단
import os
os.environ.setdefault("DISABLE_TORCHCODEC", "1")
```

#### 구현 위치

- `main.py`: `os.environ.setdefault("DISABLE_TORCHCODEC", "1")` 추가
- `requirements.txt`: `soundfile` 추가

---

## Q11. `python main.py --steps prepare` 실행 중 터미널이 갑자기 죽는 이유는?

### A. `num_proc` 설정이 물리 CPU 코어 수를 초과하여 Linux OOM Killer가 프로세스를 강제 종료하기 때문이다.

#### 원인 분석

`datasets.map(num_proc=N)`은 현재 Python 프로세스를 N개 **fork**한다. 각 fork는 부모 프로세스의 메모리를 복사하므로, 사용 메모리가 N배로 증가한다.

```
부모 프로세스 메모리 (Python + torch + 데이터셋): ~2~3 GB
num_proc=12 → 12개 fork → 최대 ~24~36 GB 필요
실제 가용 RAM: 12 GB → OOM → Linux가 프로세스 강제 종료
```

#### OOM Killer 동작

리눅스 OOM Killer는 메모리 부족 시 가장 많은 메모리를 사용하는 프로세스를 조용히 종료한다. 별도 오류 메시지 없이 터미널이 죽는 것처럼 보인다.

#### 잘못된 설정 예시

```yaml
augmentation:
  num_proc: 12   # CPU 6코어 환경에서 12개 fork → OOM
```

#### 올바른 설정 기준

| 상황 | 권장 num_proc |
|------|--------------|
| 메모리 여유 없음 / 소규모 데이터셋 | `1` (fork 없음, 가장 안전) |
| 일반적인 경우 | `물리 CPU 코어 수 / 2` |
| 메모리 여유 충분 + 대규모 데이터셋 | `물리 CPU 코어 수` 이하 |

```
note: 소규모 데이터셋(수천 개)에서는 num_proc=1이 fork 오버헤드가 없어
      num_proc=4보다 오히려 빠른 경우가 많다.
```

#### 해결 방법

```yaml
# config.yaml
augmentation:
  num_proc: 1       # fork 없이 순차 처리 (메모리 안전)

preprocessing:
  num_proc: 1
  batch_size: 16    # 한 번에 로드하는 샘플 수 축소
```

#### 구현 위치

- `config.yaml`: `augmentation.num_proc: 1`, `preprocessing.num_proc: 1`, `preprocessing.batch_size: 16` 으로 수정

---

## Q12. GPU를 사용하려는데 `CUDA available: False` 이고 PyTorch가 GPU를 인식하지 못하는 이유는?

### A. 설치된 PyTorch의 CUDA 버전이 시스템 NVIDIA 드라이버가 지원하는 CUDA 버전보다 높기 때문이다.

#### 오류 메시지

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12020).
RuntimeError: The NVIDIA driver on your system is too old.
```

#### 원인

PyTorch는 특정 CUDA 버전용으로 컴파일된다. 설치된 PyTorch의 CUDA 버전이 드라이버가 지원하는 최대 CUDA 버전보다 높으면 GPU를 사용할 수 없다.

```
설치된 PyTorch: 2.11.0+cu130  → CUDA 13.0 필요 → 드라이버 570+ 필요
실제 드라이버:  535.288.01    → CUDA 12.2 지원까지
→ 버전 불일치 → CUDA available: False
```

#### NVIDIA 드라이버 버전과 지원 CUDA 버전 관계

| NVIDIA 드라이버 | 지원 최대 CUDA |
|----------------|--------------|
| 535.x | 12.2 |
| 550.x | 12.4 |
| 570.x | 13.0 |

> 드라이버 업데이트가 아닌 **PyTorch 재설치**가 올바른 해결책이다. 드라이버 535.x는 현재 환경(GTX 1660)에 적합한 버전이다.

#### 해결 방법

```bash
# 기존 PyTorch 제거
pip uninstall torch torchaudio torchcodec -y

# 드라이버(CUDA 12.2)에 맞는 cu121 빌드로 재설치
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 확인
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

#### 구현 위치

- `requirements.txt`: `torch`, `torchaudio` cu121 빌드로 교체

---

## Q13. 학습 중 `ImportError: To support decoding audio data, please install 'torchcodec'` 오류가 반복되는 이유는?

### A. 전처리 완료된 캐시에 `audio` 컬럼(Audio 타입)이 그대로 남아있어 DataLoader가 학습 중 오디오를 재디코딩하려 하기 때문이다.

#### 오류 발생 흐름

```
trainer.train()
  └→ DataLoader가 배치 로드
       └→ datasets가 모든 컬럼 디코딩 시도
            └→ audio 컬럼(Audio 타입) → torchcodec으로 디코딩
                 └→ torchcodec 미설치 → ImportError
```

#### 원인

`preprocess_datasetdict_batched(remove_others=False)` 설정으로 전처리 후에도 원본 컬럼(`audio`, `sentence`)이 캐시에 그대로 보존된다. 학습에 필요한 컬럼은 `input_features`와 `labels`뿐이지만, DataLoader는 전체 행을 읽으며 `audio` 컬럼 디코딩을 시도한다.

```python
# 전처리 후 캐시 컬럼 상태 (문제 있음)
columns: ['audio', 'sentence', 'input_features', 'labels']
#          ↑ Audio(decode=True) → DataLoader가 torchcodec 호출
```

#### 해결 방법

**방법 1 — 학습 직전 컬럼 제거 (기존 캐시 즉시 적용):**

```python
# main.py - trainer.train() 호출 전
drop_cols = [c for c in ["audio", "sentence"] if c in train_dataset.column_names]
if drop_cols:
    train_dataset = train_dataset.remove_columns(drop_cols)
    eval_dataset  = eval_dataset.remove_columns(
        [c for c in drop_cols if c in eval_dataset.column_names]
    )
```

**방법 2 — 전처리 시 원본 컬럼 제거 (근본 해결, 캐시 재생성 필요):**

```python
# main.py - prepare_datasets_offline() 내부
dsd_proc = preprocess_datasetdict_batched(
    ...
    remove_others=True,   # False → True 변경
    ...
)
```

```bash
# 기존 전처리 캐시 삭제 후 재생성
rm -rf storage/preprocessed_enuma_speech
python main.py --steps prepare
```

#### 전처리 후 올바른 컬럼 구조

```python
# 학습에 필요한 컬럼만 남긴 상태
columns: ['input_features', 'labels']
#          ↑ log-mel 특징    ↑ 토큰 ID
```

#### 구현 위치

- `main.py`: 학습 직전 `remove_columns(["audio", "sentence"])` 추가
- `main.py`: `prepare_datasets_offline()` 에서 `remove_others=True` 로 수정
- `storage/preprocessed_enuma_speech/`: 삭제 후 재생성

---

## Q14. `datasets.map(num_proc=1)`이어도 torchcodec 오류가 subprocess에서 반복되는 이유는?

### A. `num_proc=1`이어도 datasets가 내부적으로 subprocess를 spawn하며, 해당 subprocess는 Python 코드에서 설정한 환경변수를 상속받지 못하기 때문이다.

#### 원인

```python
os.environ.setdefault("DISABLE_TORCHCODEC", "1")  # main.py에서 설정
```

이 환경변수는 메인 프로세스에는 적용되지만, `num_proc=1`로 spawn된 worker subprocess는 `spawn` 방식으로 생성되어 런타임 환경변수를 상속받지 못한다.

```
메인 프로세스: DISABLE_TORCHCODEC=1 (설정됨)
  └→ datasets.map(num_proc=1) → subprocess spawn
       └→ 새 프로세스: DISABLE_TORCHCODEC 없음 → torchcodec 호출 → ImportError
```

#### num_proc 값에 따른 동작 차이

| 값 | 동작 | torchcodec 문제 |
|----|------|----------------|
| `1` | subprocess spawn (multiprocess 사용) | 발생 |
| `None` | 메인 프로세스에서 직접 실행 (subprocess 없음) | 발생 안 함 |
| `2+` | N개 subprocess spawn | 발생 |

#### 해결 방법

`config.yaml`에서 `num_proc`을 `null`(Python의 `None`)로 설정한다.

```yaml
augmentation:
  num_proc: null    # subprocess 없이 메인 프로세스에서 실행

preprocessing:
  num_proc: null
```

`None`으로 설정하면 `Dataset.map()`이 subprocess를 전혀 생성하지 않고 메인 프로세스에서 직접 실행하여 환경변수 상속 문제가 사라진다.

#### 구현 위치

- `config.yaml`: `augmentation.num_proc: null`, `preprocessing.num_proc: null` 로 수정

---

## Q15. `Seq2SeqTrainingArguments`에서 `unexpected keyword argument 'group_by_length'` 오류가 발생하는 이유는?

### A. `transformers` 5.x에서 `group_by_length`와 `length_column_name` 파라미터가 제거됐기 때문이다.

#### 오류 메시지

```
TypeError: Seq2SeqTrainingArguments.__init__() got an unexpected keyword argument 'group_by_length'
```

#### 원인

`transformers 4.x`에서 존재하던 `group_by_length`, `length_column_name` 파라미터가 `5.x`에서 삭제됐다.

#### 해결 방법

`Seq2SeqTrainingArguments`에서 두 인자를 제거한다.

```python
# 제거 전
training_args = Seq2SeqTrainingArguments(
    ...
    group_by_length=False,       # 삭제
    length_column_name="input_length",  # 삭제
    ...
)

# 제거 후
training_args = Seq2SeqTrainingArguments(
    ...
    # 위 두 줄 없음
    ...
)
```

#### 구현 위치

- `main.py`: `Seq2SeqTrainingArguments` 생성 시 `group_by_length`, `length_column_name` 인자 제거

---

## Q16. 학습 로그(`loss`, `grad_norm`, `learning_rate`, `epoch`)는 각각 무엇을 의미하는가?

### A. 각 지표는 학습이 올바르게 수렴하고 있는지 판단하는 핵심 신호다.

#### 실제 로그 예시

```
{'loss': '5.43',  'grad_norm': '10.02', 'learning_rate': '9.854e-05', 'epoch': '0.53'}
{'loss': '3.626', 'grad_norm': '5.084', 'learning_rate': '8.392e-05', 'epoch': '1.05'}
{'loss': '3.278', 'grad_norm': '5.851', 'learning_rate': '6.93e-05',  'epoch': '1.58'}
{'loss': '3.189', 'grad_norm': '4.181', 'learning_rate': '5.468e-05', 'epoch': '2.11'}
{'loss': '3.037', 'grad_norm': '3.243', 'learning_rate': '4.006e-05', 'epoch': '2.63'}
{'loss': '2.995', 'grad_norm': '1.736', 'learning_rate': '2.544e-05', 'epoch': '3.16'}
{'loss': '2.94',  'grad_norm': '1.832', 'learning_rate': '1.082e-05', 'epoch': '3.68'}
```

#### loss (손실값)

모델 예측과 정답의 차이. **낮을수록 좋으며 감소 추세여야 정상이다.**

```
5.43 → 3.63 → 3.28 → 3.19 → 3.04 → 3.00 → 2.94   ← 정상 수렴
```

| 상태 | 의미 |
|------|------|
| 꾸준히 감소 | 정상 학습 |
| 감소 후 증가 | 과적합 시작 |
| 거의 변화 없음 | 학습률 너무 작음 또는 수렴 완료 |
| 폭발적 증가 | 학습률 너무 크거나 gradient 발산 |

#### grad_norm (그레이디언트 노름)

파라미터 업데이트 크기. **초반에 크다가 점차 안정화되는 것이 정상이다.**

```
10.02 → 5.08 → 5.85 → 4.18 → 3.24 → 1.74 → 1.83   ← 안정화
```

`max_grad_norm: 1.0` 설정으로 gradient clipping이 적용되어 있어 `grad_norm > 1.0`이면 클리핑된 것이다. 지속적으로 매우 큰 값(20+)이 나오면 학습률을 낮춰야 한다.

#### learning_rate (학습률 스케줄)

Warmup 후 cosine 감쇠로 자동 조절된다.

```
학습률
  │
1e-4 ─┤     ╭───────╮
      │    ╱         ╲
      │   ╱            ╲
      │  ╱               ╲
  0 ──┼─╱─────────────────╲──→ step
      ←warmup→←── cosine decay ──→
     (10%)
```

`warmup_ratio: 0.1` → 전체 380 steps 중 처음 38 steps 동안 0 → 1e-4 로 선형 증가 후 감쇠

#### epoch

전체 학습 데이터를 몇 번 반복했는지.

```
1,519 샘플 ÷ 배치8 ÷ gradient_accumulation2 = 95 steps/에폭
4 에폭 × 95 = 380 steps 총합
```

#### 학습 완료 후 생성 결과물

```
checkpoints/whisper-tiny-en/
├── checkpoint-500/           # 500 steps 체크포인트
├── ...
└── (best model 자동 저장)    # eval_loss 기준 최적 모델
```

`load_best_model_at_end: true` 설정으로 학습 완료 시 가장 낮은 `eval_loss`를 기록한 체크포인트가 최종 모델로 선택된다.

---

## Q17. 8에폭 학습 결과에서 에폭 4 이후 eval_loss는 증가하는데 eval_wer은 계속 감소하는 이유는?

### A. eval_loss와 eval_wer은 서로 다른 것을 측정하기 때문이며, 음성인식에서는 WER이 실용적 지표다.

#### 에폭별 Validation 성능 (실제 학습 결과)

| 에폭 | eval_loss | eval_wer | loss 변화 | wer 변화 |
|------|-----------|----------|----------|---------|
| 1 | 1.693 | 27.74% | - | - |
| 2 | 1.652 | 25.52% | ▼ -0.041 | ▼ -2.22%p |
| 3 | 1.632 | 23.08% | ▼ -0.020 | ▼ -2.44%p |
| 4 | **1.628** | 21.84% | ▼ -0.004 | ▼ -1.24%p |
| 5 | 1.631 | 20.72% | ▲ +0.003 | ▼ -1.12%p |
| 6 | 1.633 | 20.38% | ▲ +0.002 | ▼ -0.34%p |
| 7 | 1.643 | 20.45% | ▲ +0.010 | ▲ +0.07%p |
| 8 | **1.649** | **20.16%** | ▲ +0.006 | ▼ -0.29%p |

에폭 4에서 eval_loss 최솟값(1.628), 에폭 8에서 eval_wer 최솟값(20.16%)으로 두 지표의 최적 시점이 다르다.

#### eval_loss란?

**Cross-Entropy Loss** — 모델이 각 토큰을 얼마나 확신하며 예측하는지를 확률 기준으로 측정한다.

```
정답 토큰 시퀀스:  ["안", "녕", "하", "세", "요"]
모델 예측 확률:   [ 0.8,  0.6,  0.9,  0.7,  0.5 ]

Loss = -( log(0.8) + log(0.6) + log(0.9) + log(0.7) + log(0.5) ) / 5
     = 0.52  ← 낮을수록 정답 토큰에 자신 있음
```

#### eval_loss vs eval_wer 비교

| 항목 | eval_wer | eval_loss |
|------|----------|-----------|
| 단위 | 단어 오류율 (%) | 토큰 예측 불확실성 (nats) |
| 측정 대상 | 최종 출력 단어가 맞는지 | 각 토큰 예측 확률이 높은지 |
| 계산 시점 | 디코딩 완료 후 | 토큰 생성 중 |
| 직관 | "틀린 단어 비율" | "예측에 얼마나 자신 있는지" |

#### 괴리가 발생하는 원리

에폭 4 이후 두 지표가 반대 방향으로 움직이는 이유:

```
정답: "안녕하세요"

에폭 4 모델:  "안녕하세요" 예측 (확률 0.85) → loss 낮음, WER 0%
에폭 8 모델:  "안녕하세요" 예측 (확률 0.72) → loss 높음, WER 0%
```

두 모델 모두 **정답을 맞히지만** 확률 분포가 달라 loss 값이 다르다. 모델이 특정 패턴에 과신(overconfident)해지면 오히려 loss가 증가할 수 있다.

#### Train Loss 수렴 분석

```
에폭 0.5:  5.685  ← warmup 초반
에폭 2.0:  3.242  ← 빠른 감소 구간
에폭 4.0:  2.936
에폭 6.0:  2.865
에폭 7.0:  2.851
에폭 7.5:  2.848  ← 변화량 < 0.01, 수렴 완료
```

에폭 6 이후 train loss 변화량이 0.01 미만으로 **모델이 데이터에 충분히 적응** 완료.

#### Grad Norm 안정화

```
초반(에폭 0.5): 9.84  ← 큰 업데이트 (초기 탐색)
에폭 4.2:       5.16  ← 일시 급등 (새 국소 최적점 탐색)
후반(에폭 7.5): 0.51  ← 안정적 수렴
```

에폭 4.2의 grad_norm 급등 후 빠르게 안정화된 것은 정상적인 학습 흐름이다.

#### 결론 및 권장 설정

에폭 4 이후 성능 향상 폭이 급격히 감소하므로 **이 데이터셋(1,519개)의 적정 에폭은 4~6**이다.

음성인식에서는 최종 단어 정확도(WER)가 실용적 지표이므로 best model 선택 기준을 WER로 변경하는 것이 적합하다.

```yaml
# config.yaml
training:
  metric_for_best_model: "eval_wer"   # eval_loss → eval_wer
  greater_is_better: false             # WER은 낮을수록 좋음 (그대로 유지)
```

#### 구현 위치

- `config.yaml`: `metric_for_best_model: "eval_wer"` 로 수정

---

## Q19. 전체 파라미터 파인튜닝 대신 LoRA / QLoRA를 적용할 수 있는가?

### A. config.yaml의 `lora.mode`를 변경하는 것만으로 Full / LoRA / QLoRA 세 가지 방식을 선택할 수 있다.

#### 세 방식 비교

| 방식 | 학습 파라미터 | GPU 메모리 (whisper-tiny 기준) | 설치 필요 패키지 |
|------|-------------|-------------------------------|----------------|
| **Full Fine-tuning** | 39M (100%) | ~2.5 GB | 없음 (현재 방식) |
| **LoRA** | ~3.7M (9.5%) | ~1.5 GB | `peft` |
| **QLoRA** | ~3.7M (9.5%) | ~0.8 GB | `peft` + `bitsandbytes` |

```
pip install peft              # LoRA 사용 시
pip install bitsandbytes      # QLoRA 추가 시
```

#### config.yaml 설정

```yaml
lora:
  mode: "full"               # "full" | "lora" | "qlora" 중 하나 선택

  # LoRA / QLoRA 공통 파라미터
  r: 32                      # 랭크: 클수록 학습 파라미터 ↑ (8~64 권장)
  lora_alpha: 64             # 스케일링 계수 (보통 r * 2)
  lora_dropout: 0.05         # 드롭아웃 (과적합 방지, 0.0~0.1)
  target_modules:            # LoRA를 적용할 Whisper 레이어
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "out_proj"
    - "fc1"
    - "fc2"

  # QLoRA 전용 파라미터
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"
  use_double_quant: true
```

#### LoRA 원리

LoRA(Low-Rank Adaptation)는 기존 가중치 행렬을 고정(freeze)하고, 저랭크(low-rank) 행렬 두 개(A, B)만 학습하는 방식이다.

```
기존 방식 (Full Fine-tuning):
  W_new = W_pretrained + ΔW     # ΔW는 W와 동일한 크기 (예: 512×512 = 262,144 파라미터)

LoRA:
  W_new = W_pretrained + B × A  # A: (r×512), B: (512×r), r=32 → 512×32×2 = 32,768 파라미터
  ↑ 고정 (학습 안 함)  ↑ 이 부분만 학습 (전체의 12.5%)
```

랭크 `r`이 작을수록 파라미터 절약이 크고, 클수록 표현력이 높아진다.

#### QLoRA 원리

QLoRA는 LoRA에 4비트 양자화를 추가한 방식이다.

```
기존 모델 가중치: FP32 (32비트)
QLoRA 모델 가중치: NF4 (4비트) ← 메모리 8분의 1
연산: 4비트 → FP16으로 dequantize → 연산 → 다시 4비트 저장
LoRA 어댑터: FP16으로 학습 (양자화 없음)
```

```
모델 메모리: 39M × 4bit = ~20MB (기존 FP32: 39M × 32bit = ~150MB)
```

#### Whisper에서 LoRA를 적용하는 레이어

```
Whisper 구조
  ├── Encoder (음성 → 표현)
  │     └── 각 레이어: Self-Attention (q/k/v/out_proj) + FFN (fc1/fc2)
  └── Decoder (표현 → 텍스트)
        └── 각 레이어: Self-Attention + Cross-Attention (q/k/v/out_proj) + FFN (fc1/fc2)

LoRA 적용 대상 (target_modules):
  "q_proj", "v_proj"  ← Query/Value 행렬 (가장 중요, 많은 연구에서 이것만으로도 효과)
  "k_proj", "out_proj" ← Key/Output 행렬
  "fc1", "fc2"         ← Feed-Forward Network (추가하면 파라미터 ↑, 성능 ↑)
```

#### 파이프라인 흐름 변화

**Full Fine-tuning (기존):**
```
get_model() → generation_config 설정 → Seq2SeqTrainer 학습
→ checkpoints/에 전체 모델 저장 (~150MB)
```

**LoRA:**
```
get_model() → generation_config 설정 → get_peft_model() → Seq2SeqTrainer 학습
→ checkpoints/에 어댑터만 저장 (~2MB)
→ final_adapter/에 최종 어댑터 저장
```

**QLoRA:**
```
get_model(quantization_config=BitsAndBytesConfig(4bit)) → prepare_model_for_kbit_training()
→ generation_config 설정 → get_peft_model() → Seq2SeqTrainer 학습 (fp16 비활성화)
→ checkpoints/에 어댑터만 저장 (~2MB)
→ final_adapter/에 최종 어댑터 저장
```

#### 학습 후 저장 구조

```
checkpoints/whisper-tiny-en/
  ├── checkpoint-95/          # 스텝별 체크포인트 (어댑터 가중치만)
  ├── checkpoint-190/
  └── final_adapter/          # 최종 어댑터 (LoRA/QLoRA 시에만 생성)
        ├── adapter_config.json
        └── adapter_model.safetensors   # ~2MB (전체 모델 대비 1% 크기)
```

#### 학습 후 추론 시 모델 병합

```python
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

# 방법 1: 어댑터를 분리된 상태로 사용 (메모리 효율적)
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model = PeftModel.from_pretrained(base_model, "checkpoints/whisper-tiny-en/final_adapter")

# 방법 2: 원본 모델에 병합 (추론 속도 동일, 배포 용이)
model = model.merge_and_unload()
```

#### QLoRA 사용 시 주의사항

| 항목 | 내용 |
|------|------|
| fp16 | QLoRA 사용 시 자동으로 비활성화됨 (4비트 모델과 충돌) |
| GPU | CUDA GPU 필수 (CPU는 bitsandbytes 미지원) |
| 저장 | 체크포인트에 양자화된 기본 모델은 저장되지 않음 (어댑터만 저장) |
| 추론 | 병합 후 추론 시 원본 모델을 다시 로드해야 함 |

#### 어떤 방식을 선택할까?

```
GPU 메모리 충분 (8GB+) + 소규모 데이터  → LoRA   (과적합 방지 + 빠름)
GPU 메모리 부족 (4GB 이하)             → QLoRA  (대형 모델도 단일 GPU에서 가능)
대규모 데이터 + 충분한 GPU 자원         → Full   (최대 성능)
whisper-large-v3 같은 대형 모델        → QLoRA  (단일 GPU에서 유일한 실용적 선택)
```

#### 구현 위치

- `config.yaml`: `lora` 섹션 추가 (`mode`, `r`, `lora_alpha`, `target_modules` 등)
- `config_loader.py`: `lora.mode` 유효성 검증 및 필수 키 확인 추가
- `access_whisperobj.py`: `get_model()`에 `quantization_config` 파라미터 추가 (QLoRA용)
- `main.py`: `_build_qlora_bnb_config()`, `_apply_lora()` 함수 추가, `setup_model()` 수정, 학습 후 어댑터 저장 로직 추가

---

## Q18. `num_epochs`, `train_batch_size`, `gradient_accumulation_steps`의 관계는?

### A. 세 파라미터가 결합되어 실제 가중치 업데이트 횟수(총 optimizer steps)가 결정된다.

#### 기본 개념

```yaml
training:
  num_epochs: 8           # 전체 데이터를 8회 반복
  train_batch_size: 8     # 1회 순전파에 투입되는 샘플 수
  gradient_accumulation_steps: 2  # N회 순전파 후 1회 가중치 업데이트
```

| 개념 | 설명 |
|------|------|
| **1 epoch** | 훈련 데이터 전체(10,000개)를 1회 완전히 학습 |
| **8 epochs** | 훈련 데이터 전체를 8회 반복 학습 |
| **train_batch_size=8** | 매 순전파마다 8개 샘플을 묶어 모델에 투입 |

#### 전체 학습 흐름

```
훈련 데이터 10,000개, batch=8, gradient_accumulation=2

1 에폭:
  10,000 ÷ 8 = 1,250번 순전파(forward pass)
  1,250 ÷ 2  =   625번 가중치 업데이트(optimizer step)

8 에폭 전체:
  1,250 × 8 = 10,000번 순전파
    625 × 8 =  5,000번 가중치 업데이트
```

#### gradient_accumulation의 역할

GPU 메모리가 부족할 때 큰 배치를 여러 번에 나눠 처리하면서도 효과는 큰 배치와 동일하게 만드는 기법이다.

```
[batch 1: 8개] → gradient 계산 → 저장
[batch 2: 8개] → gradient 계산 → 누적
                                  ↓
                          가중치 1회 업데이트
                     (유효 배치 = 8 × 2 = 16개 효과)
```

```
직접 batch 16으로 학습  ≈  batch 8 + gradient_accumulation 2
(메모리 16개 필요)           (메모리 8개만 필요, 효과는 동일)
```

#### 요약표

| 설정 | 값 | 의미 |
|------|-----|------|
| 훈련 데이터 | 10,000개 | 1에폭당 전체 사용 |
| `num_epochs` | 8 | 전체 데이터 8회 반복 |
| `train_batch_size` | 8 | 1회 순전파 투입 샘플 수 |
| `gradient_accumulation_steps` | 2 | 2회 순전파 후 1회 가중치 업데이트 |
| **유효 배치 크기** | **16** | 실질적 학습 단위 (8 × 2) |
| 총 순전파 횟수 | 10,000회 | num_epochs × (데이터 수 ÷ batch) |
| 총 optimizer steps | 5,000회 | 실제 가중치 업데이트 횟수 |

---

## Q19. 학습 파라미터 각각의 역할을 비유로 설명하면?

### A. 각 파라미터를 일상적인 비유와 함께 설명한다.

현재 `config.yaml`의 학습 설정 기준:

```yaml
training:
  output_dir: "./checkpoints/whisper-tiny-en"
  num_epochs: 8
  train_batch_size: 8
  eval_batch_size: 8
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  learning_rate: 1.0e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  label_smoothing_factor: 0.1
  eval_steps: 95
  save_steps: 95
  save_total_limit: 3
  logging_steps: 50
  fp16: true
  seed: 42
  dataloader_num_workers: 0
```

#### `output_dir` — 체크포인트 저장 경로

시험 공부 중 **중간 필기 노트를 보관하는 폴더**. 나중에 가장 잘 정리된 노트를 골라 쓸 수 있다.

#### `num_epochs: 8` — 전체 데이터 반복 횟수

교과서를 **8번 정독**하는 것. 1번 읽으면 대략적 이해, 반복할수록 깊은 이해. 하지만 너무 많이 읽으면 교과서를 통째로 외워버려서(과적합) 응용 문제를 못 푼다.

```
에폭 1~2: "아, 이런 패턴이 있구나" (과소적합 구간)
에폭 3~5: "패턴을 잘 이해했다" (최적 구간)
에폭 6~8: "이제 문제 자체를 외우고 있다" (과적합 위험 구간)
         → 이를 감시하기 위해 eval_steps로 중간 평가
```

#### `train_batch_size: 8` — GPU당 한 번에 처리하는 샘플 수

선생님이 **한 번에 8명의 학생 답안을 채점**한 후 피드백을 주는 것. 1명씩 보면 편향되기 쉽고, 100명을 한꺼번에 보면 책상(GPU 메모리)이 부족하다.

```
batch_size=1:  학생 1명 보고 판단 → 불안정 (답안 하나에 휘둘림)
batch_size=8:  학생 8명 보고 판단 → 적당한 균형
batch_size=64: 학생 64명 보고 판단 → 안정적이나 GPU 메모리 부족
```

#### `eval_batch_size: 8` — 평가 시 배치 크기

**모의고사 채점** 시 한 번에 몇 장씩 채점하느냐. 학습과 달리 피드백(gradient)을 줄 필요가 없으므로 더 많이 한꺼번에 처리할 수 있다. 학습 batch_size보다 크게 잡아도 된다.

#### `gradient_accumulation_steps: 2` — gradient 누적 횟수

선생님이 8명씩 **2번 채점**하고, 총 16명의 결과를 종합하여 수업 방향을 수정하는 것. GPU 메모리는 8명분만 필요하지만 16명분의 의견을 반영한다.

```
실제 GPU 메모리 사용: batch_size=8 만큼
실제 학습 효과:       batch_size=16 만큼

[8개 처리 → gradient 저장] → [8개 처리 → gradient 누적] → [16개분 한꺼번에 업데이트]
```

#### `max_grad_norm: 1.0` — gradient clipping 임계값

자동차의 **속도 제한장치**. 내리막길에서 갑자기 속도가 치솟아 사고(발산)가 나는 것을 방지. 평지에서는 작동하지 않지만, 급경사(loss 급변) 구간에서 안전장치 역할을 한다.

```
gradient가 0.5 → 그대로 사용 (제한 이내)
gradient가 3.0 → 1.0으로 축소 (폭주 방지)
gradient가 100  → 1.0으로 축소 (발산 방지)
```

#### `learning_rate: 1.0e-4` — 가중치 업데이트 크기

**산에서 하산할 때 보폭의 크기**.

```
1e-2 (큰 보폭):   빠르지만 골짜기를 지나칠 수 있다 → 발산
1e-4 (적절 보폭):  안정적으로 골짜기(최적점)에 도달
1e-7 (너무 작은 보폭): 한 발짝에 1mm → 해가 저물어도 도착 못함
```

#### `warmup_ratio: 0.1` — 학습률 워밍업 비율

자동차 **워밍업 운전**. 시동 걸자마자 고속도로에 들어가면 엔진에 무리가 간다. 서서히 속도를 올려야 안정적.

```
학습률
  │
1e-4 ┤          ╭────────────╮
     │        ╱               ╲  (decay)
     │      ╱
     │    ╱
0    ┤──╱
     └──┬────┬───────────────┬──→ 스텝
        0   10%             100%
        ← 워밍업 →

총 스텝이 760이면 (95 steps/epoch × 8 epochs):
  워밍업 구간 = 760 × 0.1 = 76 스텝
```

#### `weight_decay: 0.01` — L2 정규화 계수

**짐을 가볍게 유지하는 규칙**. 등산 중 불필요한 짐(큰 가중치)을 조금씩 버리게 하여 민첩한 상태를 유지. 짐이 너무 많으면(가중치가 큼) 특정 경로에만 의존(과적합)하게 된다.

```
decay=0:     짐을 절대 안 버림 → 과적합 위험
decay=0.01:  적당히 정리 → 일반화 향상
decay=0.5:   너무 많이 버림 → 학습 자체가 어려워짐
```

#### `label_smoothing_factor: 0.1` — 라벨 스무딩 계수

시험 채점 시 "정답은 무조건 100점, 오답은 0점"이 아니라 **"정답은 90점, 나머지도 조금씩 점수를 줌"**. 모델이 겸손하게 학습하여 새로운 문제에서도 유연하게 대응한다.

```
smoothing=0.0:  정답=[0, 0, 1, 0, 0]      "이거 100% 확실해!"
smoothing=0.1:  정답=[0.025, 0.025, 0.9, 0.025, 0.025]  "이거 맞는 것 같은데, 다른 것도 약간..."
```

#### `eval_steps: 95` — 평가 주기

교과서 1번 정독할 때마다 **모의고사를 1회** 보는 것. 너무 자주 보면 공부 시간이 줄고, 너무 드물면 실력 변화를 파악 못한다.

```
1,519개 샘플 ÷ batch_size 8 ÷ accumulation 2 = 약 95 steps/epoch

스텝:  95       190       285       ...      760
       │        │         │                   │
       EVAL     EVAL      EVAL               EVAL
      (1에폭)  (2에폭)   (3에폭)            (8에폭)
```

#### `save_steps: 95` — 체크포인트 저장 주기

게임의 **자동 저장 포인트**. 모의고사를 볼 때마다 저장해 두면, 나중에 가장 성적이 좋았던 시점으로 돌아갈 수 있다 (`load_best_model_at_end=True`).

#### `save_total_limit: 3` — 최대 체크포인트 보관 수

게임 세이브 슬롯이 **3칸**. 새로 저장하면 가장 오래된 저장을 덮어쓴다. 디스크 공간 절약을 위해. (Whisper 체크포인트 1개 ≈ 수백 MB)

```
checkpoint-95  ← 삭제됨
checkpoint-190 ← 삭제됨
checkpoint-285 ← 유지
checkpoint-380 ← 유지
checkpoint-475 ← 유지 (최근 3개)
```

#### `logging_steps: 50` — 로그 출력 주기

운동 중 **50걸음마다 심박수를 확인**하는 것. 학습이 잘 되고 있는지(loss 감소) 중간 점검. eval보다 자주 기록하여 세밀한 추이를 관찰한다.

#### `fp16: true` — FP16 혼합 정밀도 학습

계산할 때 **소수점 이하 자릿수를 줄여서** 빠르게 암산하되, 최종 답안지에는 정밀한 값을 기록하는 것.

```
FP32: 3.141592653589793  (정밀하지만 메모리 2배, 연산 느림)
FP16: 3.140625           (약간 부정확하지만 메모리 절반, 연산 빠름)

⚠️ Apple Silicon(MPS)에서는 fp16: false로 설정 필요 (CUDA 전용)
```

#### `seed: 42` — 랜덤 시드

**주사위에 번호를 매겨놓는 것**. seed=42로 던지면 항상 같은 순서로 나온다. 실험 A와 B의 차이가 랜덤성 때문인지, 실제 설정 차이 때문인지 구분 가능.

#### `dataloader_num_workers: 0` — 데이터 로딩 워커 수

요리사(GPU)에게 재료를 가져다주는 **보조 인원 수**.

```
workers=0: 요리사가 직접 재료를 가지러 감 → 디버깅 쉬움, 속도 보통
workers=2: 보조 2명이 미리 재료 준비 → GPU 대기 시간 감소
workers=4: 보조 4명 → 더 빠르지만 CPU/메모리 추가 사용

on_the_fly 모드에서는 증강이 CPU에서 이루어지므로
workers=2~4로 올리면 GPU 유휴 시간을 줄일 수 있다.
```

#### 전체 파라미터 상호 관계도

```
[데이터 1,519개]
       │
       ▼
  batch_size=8 ──┐
                 ├─ accumulation=2 ──→ 유효 배치 16
                 │                          │
                 │                    learning_rate=1e-4
                 │                    warmup_ratio=0.1
                 │                    weight_decay=0.01
                 │                    max_grad_norm=1.0
                 │                    label_smoothing=0.1
                 │                          │
                 │                          ▼
                 │                   95 steps/epoch × 8 epochs = 760 total steps
                 │                          │
                 ├── logging: 매 50 steps   │
                 ├── eval:    매 95 steps   │
                 ├── save:    매 95 steps (최근 3개만)
                 │                          │
                 │                          ▼
                 │                   best model 선택 (eval_loss 최소)
                 └── fp16으로 메모리/속도 최적화
```

---

## Q20. 학습 로그(`loss`, `grad_norm`, `learning_rate`, `epoch`)는 어떻게 읽는가?

### A. 각 지표는 학습이 올바르게 수렴하고 있는지 판단하는 핵심 신호다.

#### 실제 학습 로그

```
{'loss': '5.43',  'grad_norm': '10.02', 'learning_rate': '9.854e-05', 'epoch': '0.53'}
{'loss': '3.626', 'grad_norm': '5.084', 'learning_rate': '8.392e-05', 'epoch': '1.05'}
{'loss': '3.278', 'grad_norm': '5.851', 'learning_rate': '6.93e-05',  'epoch': '1.58'}
{'loss': '3.189', 'grad_norm': '4.181', 'learning_rate': '5.468e-05', 'epoch': '2.11'}
{'loss': '3.037', 'grad_norm': '3.243', 'learning_rate': '4.006e-05', 'epoch': '2.63'}
{'loss': '2.995', 'grad_norm': '1.736', 'learning_rate': '2.544e-05', 'epoch': '3.16'}
{'loss': '2.94',  'grad_norm': '1.832', 'learning_rate': '1.082e-05', 'epoch': '3.68'}
```

#### loss (손실값) — "시험 점수의 역수"

모델 예측과 정답의 차이. **낮을수록 좋으며 감소 추세여야 정상이다.**

**비유**: 학생의 **오답률**. 처음에는 틀리는 것이 많지만(5.43), 공부할수록 줄어든다(2.94). 하락폭이 점점 작아지는 것은 "쉬운 문제는 다 풀었고, 남은 건 어려운 문제"라는 의미.

```
5.43 → 3.63 → 3.28 → 3.19 → 3.04 → 3.00 → 2.94   ← 정상 수렴

비유:
  첫 시험(에폭 0.5): 100문제 중 54개 틀림 → "기본기 부족"
  두 번째(에폭 1):   100문제 중 36개 틀림 → "패턴 파악 시작"
  마지막(에폭 3.7):  100문제 중 29개 틀림 → "대부분 이해, 어려운 문제만 남음"
```

| 상태 | 의미 | 비유 |
|------|------|------|
| 꾸준히 감소 | 정상 학습 | 공부할수록 성적 향상 |
| 감소 후 증가 | 과적합 시작 | 교과서를 외워버려서 응용 문제를 못 푸는 상태 |
| 거의 변화 없음 | 학습률 너무 작거나 수렴 완료 | 보폭이 너무 작아 앞으로 안 나감 |
| 폭발적 증가 | 학습률 너무 크거나 gradient 발산 | 보폭이 너무 커서 골짜기를 뛰어넘음 |

#### grad_norm (그레이디언트 노름) — "엑셀의 세기"

파라미터 업데이트 크기. **초반에 크다가 점차 안정화되는 것이 정상이다.**

**비유**: 자동차의 **엑셀 밟는 세기**. 처음에는 출발을 위해 엑셀을 세게 밟지만(10.02), 속도가 붙으면 살짝만 밟아도 된다(1.74). 갑자기 세게 밟으면(급등) 위험할 수 있지만, 곧 안정되면 정상이다.

```
10.02 → 5.08 → 5.85 → 4.18 → 3.24 → 1.74 → 1.83   ← 안정화

비유:
  에폭 0.5: 엑셀 세게 밟음 (10.02) → "출발! 방향을 크게 수정 중"
  에폭 1.6: 잠깐 급가속 (5.85)     → "코너를 돌면서 일시적 불안정"
  에폭 3.2: 살짝만 밟음 (1.74)     → "순항 중, 미세 조정만 필요"
```

`max_grad_norm: 1.0` 설정은 **속도 제한장치**와 같다. grad_norm이 1.0을 넘으면 강제로 잘라내어 폭주를 방지한다. 로그에 표시되는 값은 클리핑 전 측정값이다.

#### learning_rate (학습률 스케줄) — "보폭 조절기"

Warmup 후 cosine/linear 감쇠로 자동 조절된다.

**비유**: 산에서 내려올 때 **보폭 조절**. 처음에는 보폭을 서서히 넓히고(워밍업), 정상에서 멀어질수록 보폭을 줄여서 목적지(최적점)를 정밀하게 찾는다.

```
학습률
  │
1e-4 ─┤     ╭───────╮
      │    ╱         ╲
      │   ╱            ╲
      │  ╱               ╲
  0 ──┼─╱─────────────────╲──→ step
      ←warmup→←── cosine decay ──→
     (10%)

9.854e-05 → 8.392e-05 → 6.93e-05 → ... → 1.082e-05

비유:
  에폭 0.5: 보폭 9.8cm  → "아직 큰 걸음으로 방향 탐색"
  에폭 2.1: 보폭 5.5cm  → "대략적 방향을 잡았으니 보폭 축소"
  에폭 3.7: 보폭 1.1cm  → "골짜기 근처, 정밀 탐색 모드"
```

#### epoch (에폭) — "교과서 정독 횟수"

전체 학습 데이터를 몇 번 반복했는지.

**비유**: 교과서를 **몇 번째 읽고 있는지**. 0.53은 "첫 번째 정독의 53% 지점", 3.68은 "네 번째 정독의 68% 지점".

```
1,519 샘플 ÷ 배치8 ÷ gradient_accumulation2 = 95 steps/에폭
4 에폭 × 95 = 380 steps 총합
```

#### 에폭별 Evaluation 시뮬레이션

실제 eval 데이터는 없으므로, train loss 패턴과 Whisper 파인튜닝의 일반적 경향을 기반으로 시뮬레이션한다.

```
┌────────┬────────────┬────────────┬────────────┬──────────────────────────────┐
│ 에폭   │ train_loss │ eval_loss  │ eval_wer   │ 비유                         │
├────────┼────────────┼────────────┼────────────┼──────────────────────────────┤
│ 1      │ 3.626      │ 2.85       │ 45.2%      │ 첫 시험: 절반 가까이 틀림     │
│ 2      │ 3.189      │ 2.42       │ 32.7%      │ 기본기 완성: 쉬운 문제 해결   │
│ 3      │ 2.995      │ 2.18       │ 26.4%      │ 실력 향상: 중간 난이도 정복   │
│ 4      │ 2.94       │ 2.05       │ 23.1%      │ 수렴 접근: 어려운 문제만 남음  │
└────────┴────────────┴────────────┴────────────┴──────────────────────────────┘
```

**왜 eval_loss가 train_loss보다 낮은가?**

`label_smoothing_factor=0.1`이 적용되어 train_loss에는 스무딩 패널티가 포함되어 있다. eval은 순수 Cross-Entropy만 측정하므로 더 낮게 나온다.

**비유**: 훈련 시에는 채점 기준이 더 엄격하고(smoothing으로 100점 만점이 90점 만점으로 축소), 모의고사(eval)에서는 일반 채점 기준을 적용하는 것과 같다.

#### 시각화

```
Loss 추이:
  │
5.4 ┤ ●                             ← train_loss (엄격한 채점)
    │   ╲
3.6 ┤     ●
3.2 ┤       ●
3.0 ┤         ●  ●
2.9 ┤               ●  ●            ← 수렴 중
    │
2.8 ┤     ○                         ← eval_loss (일반 채점)
2.4 ┤         ○
2.2 ┤            ○
2.0 ┤               ○               ← eval_loss가 더 낮음
    └──┬──┬──┬──┬──┬──┬──┬──→ epoch
       0.5 1  1.5 2  2.5 3  3.5 4

WER 추이 (단어 오류율):
  │
45%┤ ○   "절반 가까이 틀림"
   │   ╲
33%┤     ○   "쉬운 단어 해결"
26%┤        ○   "중간 난이도 정복"
23%┤           ○   "어려운 단어만 남음"
   └──┬──────┬──────┬──────┬──→ epoch
      1      2      3      4
```

#### 종합 판단 — "건강 진단서"

| 지표 | 상태 | 판정 | 비유 |
|------|------|------|------|
| train_loss | 꾸준히 감소 | 정상 | 매 시험마다 성적 향상 |
| grad_norm | 초반 크고 후반 안정 | 정상 | 엑셀에서 발을 점차 뗌 |
| learning_rate | warmup 후 decay | 정상 | 보폭을 줄이며 정밀 탐색 |
| 전체 학습 | 4에폭 기준 수렴 접근 중 | 양호 | "교과서 4번 읽으니 거의 다 이해" |

#### 추가 에폭 시 예상 (4에폭 이후)

**비유**: 교과서를 4번 읽은 후 계속 더 읽으면?

```
에폭 5~6: eval_wer 소폭 개선 (23% → 21%)
          → "5번째 읽기에서 놓쳤던 세부사항 발견"
에폭 7~8: eval_loss 반등 시작, eval_wer 정체 또는 소폭 악화
          → "이제 교과서를 외워버리기 시작 → 새로운 문제에 약해짐"
```

현재 설정(4에폭)은 이 데이터셋 규모(1,519개)에 적절한 수준이다. 데이터가 더 많으면 에폭을 줄여도 되고, 데이터가 적으면 에폭을 늘리되 과적합 감시(`eval_loss` 반등 여부)가 필요하다.

---

## Q21. 데이터 로드 후 Padding, Mapping 하는 과정은 어떤 과정인가?

### A. Mapping은 원본 데이터를 모델 입력 형태로 변환하는 전처리이고, Padding은 배치 내 길이를 맞추는 정렬 과정이다.

#### 전체 흐름: 원본 데이터 → 모델 입력

```
원본 오디오/텍스트
     │
     ▼
 [1] Mapping (전처리)  ← Dataset.map()
     │  오디오 → log-mel spectrogram (80채널)
     │  텍스트 → 토큰 ID 시퀀스
     ▼
 [2] Padding (배치 정렬) ← DataCollator
     │  길이가 다른 샘플들을 동일 길이로 맞춤
     │  패딩 위치를 -100으로 마스킹
     ▼
  모델에 투입
```

#### 1단계: Mapping (전처리) — "재료 손질"

`Dataset.map()`으로 **모든 샘플을 모델이 이해하는 형태로 변환**하는 과정이다.

**비유**: 요리 전에 **재료를 손질**하는 것. 생선(원본 오디오)을 회(log-mel)로 썰고, 레시피(텍스트)를 번호 카드(토큰 ID)로 변환.

##### 오디오 → log-mel spectrogram

```
원본 오디오 파형 (1차원 시계열):
  [0.02, -0.01, 0.05, 0.03, -0.04, ...]   ← 16,000개/초

        ↓ WhisperProcessor (feature_extractor)

log-mel spectrogram (2차원 이미지):
  80채널 ┌─────────────────────────┐
  주파수 │ ░░▓▓██▓▓░░░░▓▓██████▓░ │   ← 시간 축 →
         │ ░▓██████▓░░▓██████████▓ │
         │ ▓████████▓▓████████████▓ │
         └─────────────────────────┘
         shape: (80, 3000)  ← 30초 기준
```

**비유**: 음성을 **악보로 변환**하는 것. 사람 귀로 듣는 소리(파형)를 주파수별 세기(스펙트로그램)로 펼쳐서 모델이 "읽을 수 있는" 형태로 만든다.

##### 텍스트 → 토큰 ID

```
"안녕하세요"
        ↓ WhisperTokenizer

[50364, 31015, 8930, 22993, 50257]
  ↑                              ↑
 시작토큰                      종료토큰
```

**비유**: 한국어 문장을 **숫자 암호표**로 변환. 모델은 글자를 읽지 못하고 숫자만 이해하므로, 단어/음절을 고유 번호로 바꿔준다.

##### Mapping 처리 방식

```
Dataset.map(prepare_dataset_with_processor, batched=True, batch_size=32)
                                             ↑
                                32개 샘플씩 묶어서 한 번에 처리 (효율적)
```

#### 2단계: Padding (배치 정렬) — "도시락 칸 맞추기"

DataCollator가 **배치 내 샘플들의 길이를 동일하게 맞추는** 과정이다.

**비유**: 도시락 칸이 모두 같은 크기여야 뚜껑이 닫히듯, GPU는 **같은 크기의 텐서**만 한꺼번에 처리할 수 있다. 짧은 반찬은 빈 칸(패딩)으로 채운다.

##### 왜 필요한가?

```
배치 내 3개 샘플의 라벨(토큰 ID):

샘플 A: [50364, 31015, 8930, 50257]           ← 4개 토큰
샘플 B: [50364, 22993, 7689, 4511, 50257]     ← 5개 토큰
샘플 C: [50364, 9012, 50257]                  ← 3개 토큰

        ↓ Padding (최대 길이 5에 맞춤)

샘플 A: [50364, 31015, 8930, 50257, <pad>]
샘플 B: [50364, 22993, 7689,  4511, 50257]
샘플 C: [50364,  9012, 50257, <pad>, <pad>]
```

##### -100 마스킹 — "채점하지 마세요 표시"

패딩된 위치는 **진짜 정답이 아니므로** loss 계산에서 제외해야 한다. PyTorch의 CrossEntropyLoss는 라벨이 `-100`인 위치를 자동으로 무시한다.

```
패딩 후:   [50364, 31015, 8930, 50257, <pad>]
                                        ↓
-100 적용: [50364, 31015, 8930, 50257,  -100]
                                         ↑
                              "이 칸은 채점하지 마세요"
```

**비유**: 시험지에서 빈 칸에 **"채점 제외"** 도장을 찍는 것. 학생이 안 쓴 부분을 오답으로 처리하면 불공평하므로.

##### decoder_input_ids (Shift-Right) — "힌트 카드"

Whisper는 Seq2Seq 모델이므로 디코더에 **이전 토큰을 힌트로** 제공하여 다음 토큰을 예측하게 한다.

```
labels (정답):           [31015, 8930, 22993, 50257]
                          ↓ shift-right
decoder_input_ids (힌트): [<|sot|>, 31015, 8930, 22993]

디코더 동작:
  <|sot|>  입력 → 31015 예측  ← "시작 토큰 보고 첫 단어 맞춰봐"
  31015   입력 → 8930  예측  ← "첫 단어 보고 두 번째 맞춰봐"
  8930    입력 → 22993 예측  ← "두 번째 보고 세 번째 맞춰봐"
  22993   입력 → 50257 예측  ← "세 번째 보고 끝 토큰 맞춰봐"
```

**비유**: 단어 이어말하기 게임에서 **이전 사람이 말한 단어를 듣고** 다음 단어를 말하는 것. 첫 번째 사람에게는 "시작!" 신호(sot)를 준다.

#### 전체 과정 요약

```
[원본 데이터]
  audio: 음성 파형 (길이 제각각)
  sentence: "안녕하세요"
       │
       ▼  ── Mapping (재료 손질) ──
  input_features: log-mel (80, 3000)     ← "음성을 악보로"
  labels: [50364, 31015, 8930, 50257]    ← "텍스트를 번호로"
       │
       ▼  ── Padding (도시락 칸 맞추기) ──
  input_features: (batch, 80, 3000)      ← 배치 단위 텐서
  labels: [50364, 31015, 8930, -100]     ← 패딩은 -100 마스킹
  decoder_input_ids: [<sot>, 50364, ...]  ← 힌트 카드 (shift-right)
       │
       ▼
  [모델에 투입 → 학습]
```

| 단계 | 수행 주체 | 시점 | 비유 |
|------|----------|------|------|
| Mapping | `Dataset.map()` | 학습 전 (전처리) | 재료 손질 |
| Padding | `DataCollator` | 학습 중 (매 배치) | 도시락 칸 맞추기 |

#### 구현 위치

- `preprocess_datasetdict_batch.py`: `prepare_dataset_with_processor()` — Mapping 함수
- `preprocess_datasetdict_batch.py`: `DataCollatorSpeechSeq2SeqWithPadding` — Padding 클래스
- `main.py`: `preprocess_datasetdict_batched()` 호출 및 `DataCollator` 생성

---

## Q22. Overfitting(과적합)과 Underfitting(과소적합)의 의미는?

### A. Overfitting은 학습 데이터를 외워버린 상태, Underfitting은 충분히 학습하지 못한 상태다.

#### Overfitting (과적합) — "교과서를 통째로 외워버린 학생"

학습 데이터를 **너무 잘 외워서** 새로운 데이터에 대응하지 못하는 상태.

```
학습 데이터: "사과는 빨갛다" "바나나는 노랗다" "포도는 보라색이다"

Overfitting 모델:
  "사과 색은?" → "빨갛다" ✅
  "바나나 색은?" → "노랗다" ✅
  "수박 색은?" → "???" ❌  ← 학습에 없던 것은 못 맞춤
```

**비유**: 수학 시험에서 **기출 문제 답을 통째로 외운 학생**. 같은 문제가 나오면 100점이지만, 숫자만 바뀌어도 못 푼다.

| 신호 | 설명 |
|------|------|
| train_loss ↓↓ | 학습 데이터에서는 loss가 매우 낮음 |
| eval_loss ↑ | 평가 데이터에서는 loss가 오히려 증가 |
| 격차 확대 | train과 eval의 gap이 점점 벌어짐 |

```
loss
  │
  │  ╲  train_loss (계속 감소)
  │    ╲──────────────
  │         ╱  eval_loss (반등 시작!)
  │    ────╱
  │
  └───────────────────→ epoch
          ↑
     여기서부터 overfitting
```

##### Overfitting 원인과 대처

| 원인 | 대처 |
|------|------|
| 데이터가 너무 적음 | 데이터 증강, 더 많은 데이터 확보 |
| 에폭이 너무 많음 | Early Stopping 적용 |
| 모델이 너무 큼 | 작은 모델 사용 또는 LoRA |
| 정규화 부족 | weight_decay 증가, label_smoothing 적용, dropout 추가 |

#### Underfitting (과소적합) — "수업을 대충 들은 학생"

학습 데이터조차 **충분히 학습하지 못한** 상태.

```
학습 데이터: "사과는 빨갛다" "바나나는 노랗다" "포도는 보라색이다"

Underfitting 모델:
  "사과 색은?" → "파랗다" ❌
  "바나나 색은?" → "빨갛다" ❌  ← 학습 데이터조차 못 맞춤
```

**비유**: **교과서를 한 번도 안 읽거나 대충 훑은 학생**. 기본 문제도 못 푼다.

| 신호 | 설명 |
|------|------|
| train_loss ↑ | 학습 데이터에서도 loss가 높음 |
| eval_loss ↑ | 평가 데이터에서도 당연히 높음 |
| 격차 없음 | 둘 다 나쁜 상태로 비슷 |

```
loss
  │
  │  ── train_loss (높은 채로 정체)
  │  ── eval_loss  (비슷하게 높음)
  │
  │
  └───────────────────→ epoch
     "아직 배우지 못한 상태"
```

##### Underfitting 원인과 대처

| 원인 | 대처 |
|------|------|
| 에폭이 너무 적음 | 에폭 수 증가 |
| 학습률이 너무 작음 | learning_rate 증가 |
| 모델이 너무 작음 | 더 큰 모델 사용 (tiny → small) |
| 정규화가 너무 강함 | weight_decay 감소 |

#### 비교 요약

```
성능
(낮은 WER)
  │
  │              ╭── 최적점 (Just Right)
  │            ╱    ╲
  │          ╱        ╲
  │        ╱            ╲
  │      ╱                ╲
  │    ╱                    ╲
  │──╱                        ╲──
  │
  └──┬────────────┬────────────┬──→ 학습 정도
   Underfitting  적절함     Overfitting
   "덜 배움"    "딱 좋음"   "너무 외움"
```

| | Underfitting | Just Right | Overfitting |
|--|-------------|------------|-------------|
| 비유 | 수업을 대충 들음 | 이해하며 공부 | 답을 통째로 암기 |
| train_loss | 높음 | 낮음 | 매우 낮음 |
| eval_loss | 높음 | 낮음 | 높음 (반등) |
| WER | 나쁨 | 좋음 | 학습데이터만 좋음 |
| 해결 | 더 배우게 | 유지 | 덜 외우게 |

#### 현재 프로젝트 기준

```
에폭 1~2:  Underfitting 구간 → loss가 빠르게 감소 중 (아직 배우는 중)
에폭 3~4:  최적 구간        → loss 감소폭이 줄어듦 (충분히 학습됨)
에폭 5~8:  Overfitting 위험 → eval_loss 반등 가능성 (외우기 시작)
```
