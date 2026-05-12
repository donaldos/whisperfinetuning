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
