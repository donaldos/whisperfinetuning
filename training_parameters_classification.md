# 학습 파라미터 7가지 카테고리 분류

> Whisper 파인튜닝을 비롯한 딥러닝 모델 학습 시 사용되는 파라미터들을 역할별로 분류한 정리 문서

---

## 학습 흐름 복습

모든 학습 파라미터는 아래 4단계 중 어느 지점에 작용하는지에 따라 분류할 수 있다.

```
1. Forward propagation  → 예측 ŷ 생성
2. Loss 계산            → L = loss(ŷ, y)
3. Backpropagation      → gradient ∇L 계산
4. Gradient descent     → W ← W - lr · ∇L  (weight 업데이트)
```

---

## 7가지 카테고리

### 1. 훈련 스케일링 파라미터
> "얼마나, 몇 번 학습할 것인가"

```yaml
num_epochs: 8
train_batch_size: 8
eval_batch_size: 8
gradient_accumulation_steps: 2
```

- 총 학습량과 한 번에 처리할 데이터 양을 결정
- `effective batch size = train_batch_size × gradient_accumulation_steps × num_gpus`
- 학습 스텝 수 계산의 기준

**계산 예시 (데이터 10,000개 기준)**

```
effective batch size = 8 × 2          = 16
steps per epoch      = 10,000 ÷ 16    = 625
total training steps = 625 × 8 epochs = 5,000
```

---

### 2. 가중치 업데이트 파라미터
> "weight를 어떻게, 얼마나 바꿀 것인가"

```yaml
learning_rate: 1.0e-4
warmup_ratio: 0.1
max_grad_norm: 1.0
weight_decay: 0.01
```

주로 **4단계 (Update)** 에 작용하는 파라미터들:

| 파라미터 | 역할 |
|---|---|
| `learning_rate` | 한 번의 update에서 weight를 얼마나 바꿀지 (보폭) |
| `warmup_ratio` | 학습 초반 lr을 0부터 천천히 끌어올리는 구간의 비율 |
| `max_grad_norm` | gradient가 비정상적으로 클 때 크기 제한 (gradient clipping) |
| `weight_decay` | weight를 0 방향으로 조금씩 끌어당기는 L2 정규화 |

추가로 포함되는 항목: `lr_scheduler_type`, `optimizer`, `adam_beta1/beta2/epsilon`

---

### 3. 손실 관련 파라미터
> "예측과 정답 차이를 어떻게 계산할 것인가"

```yaml
label_smoothing_factor: 0.1
```

**2단계 (Loss)** 에 작용:

- `label_smoothing_factor`: 정답 라벨을 one-hot이 아니라 부드럽게 만들어, 모델의 과도한 확신(overconfidence)을 방지
  - 예: `[0, 0, 1, 0, 0]` → `[0.025, 0.025, 0.9, 0.025, 0.025]`

추가로 포함되는 항목: `loss_type`, task별 가중치 등

---

### 4. 로깅·평가·저장 파라미터
> "학습 과정을 어떻게 추적·보존할 것인가"

```yaml
eval_steps: 95
save_steps: 95
save_total_limit: 3
logging_steps: 50
```

학습 자체에는 영향을 주지 않지만, **재현성과 디버깅·실험 관리에 결정적**:

| 파라미터 | 역할 |
|---|---|
| `eval_steps` | 몇 스텝마다 검증 평가를 수행할지 |
| `save_steps` | 몇 스텝마다 체크포인트를 저장할지 |
| `save_total_limit` | 디스크에 유지할 체크포인트 최대 개수 |
| `logging_steps` | 몇 스텝마다 train loss 등을 로깅할지 |

추가로 포함되는 항목: `evaluation_strategy`, `save_strategy`, `load_best_model_at_end`, `metric_for_best_model`, `report_to`

---

### 5. 연산 환경 파라미터
> "어떤 정밀도·하드웨어로 계산할 것인가"

```yaml
fp16: true
dataloader_num_workers: 0
seed: 42
```

학습의 "수학"이 아니라 **"물리적 실행 방식"** 을 정하는 그룹. 이상적으로는 연산 결과에 영향을 주지 않지만, 속도·메모리·재현성을 결정.

| 파라미터 | 역할 |
|---|---|
| `fp16` / `bf16` | 16비트 부동소수점 사용 (메모리 절반, 속도 1.5~2배) |
| `dataloader_num_workers` | 데이터 로딩 병렬화 워커 수 |
| `seed` | 무작위성 고정 (재현성 확보) |
| `gradient_checkpointing` | 메모리 절약을 위한 중간 활성화 재계산 |
| `torch_compile` | PyTorch 2.x JIT 컴파일 |
| `deepspeed` / `ddp_*` | 분산 학습 설정 |

---

### 6. 데이터 처리 파라미터
> "데이터를 어떻게 모델에 먹일 것인가"

**Forward 이전 단계**에서 작동. 보통 별도의 `data_config.yaml`로 분리하는 것이 좋다.

```yaml
max_input_length: 30           # 최대 음성 길이(초)
max_label_length: 448          # 최대 텍스트 토큰 길이
language: "ko"                 # 인식 언어
task: "transcribe"             # transcribe/translate
audio_sampling_rate: 16000     # 샘플링 레이트
remove_punctuation: false      # 전처리 옵션
augmentation: {...}            # SpecAugment 등
```

---

### 7. 추론·생성 파라미터
> "평가 시 어떻게 예측을 뽑을 것인가"

평가 단계에서 모델이 출력을 생성할 때 사용. **Eval 시점**에만 작용.

```yaml
predict_with_generate: true    # 평가 시 generate() 사용
generation_max_length: 225
generation_num_beams: 1        # beam search 빔 수
```

- 훈련 자체에는 영향 없지만 **검증 시 WER이 측정되는 방식**을 결정
- beam을 늘리면 정확도는 올라가지만 평가 속도가 느려짐

---

## 전체 요약표

| # | 카테고리 | 결정하는 것 | 학습 단계 |
|---|---|---|---|
| 1 | **훈련 스케일링** | 총 학습량, 한 번에 처리할 데이터 양 | Forward 입력 단위 |
| 2 | **가중치 업데이트** | weight를 어떻게/얼마나 바꿀지 | Update |
| 3 | **손실** | 예측-정답 차이 측정 방식 | Loss |
| 4 | **로깅·평가·저장** | 학습 과정 추적·보존 | 학습 외부 |
| 5 | **연산 환경** | 정밀도, 병렬화, 재현성 | 전 단계 (실행 방식) |
| 6 | **데이터 처리** | 모델에 들어가기 전 데이터 형태 | Forward 이전 |
| 7 | **추론·생성** | 평가 시 예측 생성 방식 | Eval 시 |

---

## 학습 흐름에 매핑

```
[데이터 준비 — 6번]
        ↓
1. Forward     ← 1번 (배치 크기), 5번 (fp16 등)
        ↓
2. Loss        ← 3번 (label smoothing 등)
        ↓
3. Backward    ← 1번 (gradient accumulation), 5번 (loss scaling)
        ↓
4. Update      ← 2번 (lr, warmup, clipping, weight decay)
        ↓
[일정 스텝마다]
   - 로깅      ← 4번
   - 평가      ← 4번 + 7번 (어떻게 생성할지)
   - 체크포인트 저장 ← 4번
```

---

## 실무 팁

- **1·2·3번은 학습 결과(성능)를 직접 결정**한다. 실험할 때 주로 만지는 영역.
- **5번은 결과를 거의 안 바꾸지만 효율을 크게 바꾼다.** GPU 메모리 부족할 때 첫 번째로 손대는 곳.
- **6·7번은 별도 yaml로 분리하는 게 좋은 아키텍처**다. 모델/데이터/학습 설정의 책임을 분리하면 실험 관리가 훨씬 깔끔해진다. (Hydra, OmegaConf 같은 도구가 도움됨)
- **4번은 결과에 영향 없지만 디버깅·재현성에 결정적**이다. `logging_steps`가 너무 듬성듬성하면 학습 곡선 분석이 어렵다.

> 이 7가지 분류 체계는 PyTorch Lightning, HuggingFace Trainer, Accelerate 같은 프레임워크 설계에도 거의 그대로 반영되어 있다. HuggingFace `TrainingArguments`를 보면 위 7개 그룹이 섹션처럼 묶여 있음을 확인할 수 있다.

---

## 부록: 전체 yaml 예시 (Whisper 파인튜닝 기준)

```yaml
# === 1. 훈련 스케일링 ===
num_epochs: 8
train_batch_size: 8
eval_batch_size: 8
gradient_accumulation_steps: 2

# === 2. 가중치 업데이트 ===
learning_rate: 1.0e-4
warmup_ratio: 0.1
max_grad_norm: 1.0
weight_decay: 0.01

# === 3. 손실 ===
label_smoothing_factor: 0.1

# === 4. 로깅·평가·저장 ===
eval_steps: 95
save_steps: 95
save_total_limit: 3
logging_steps: 50

# === 5. 연산 환경 ===
fp16: true
seed: 42
dataloader_num_workers: 0

# === 6. 데이터 처리 (별도 파일 권장) ===
# language: "ko"
# task: "transcribe"
# max_input_length: 30

# === 7. 추론·생성 ===
# predict_with_generate: true
# generation_num_beams: 1
```
