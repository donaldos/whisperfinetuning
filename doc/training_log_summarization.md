# Whisper-tiny LoRA 파인튜닝 학습 로그 분석 및 개선 방향

> 90분 LoRA 파인튜닝 학습 로그 전체 분석과, "epoch를 늘리면 WER이 개선될까?"에 대한 진단

---

# Part 1. 학습 로그 전체 분석

## 1. 학습 개요

| 항목 | 값 |
|---|---|
| 모델 | Whisper-tiny (영어) |
| 파인튜닝 방식 | LoRA (rank=32, alpha=64) |
| 학습 가능 파라미터 | 2.2M / 전체 39.9M (**5.5%**) |
| 타겟 모듈 | q_proj, k_proj, v_proj, out_proj, fc1, fc2 (어텐션 + FFN 전체) |
| GPU | NVIDIA GeForce GTX 1660 SUPER (6GB) |
| 총 학습 시간 | **1시간 30분 54초** (19:37:55 → 21:08:50) |
| 총 학습 스텝 | 760 (95 steps × 8 epochs) |
| Effective batch size | 16 (8 × gradient_accumulation 2) |

---

## 2. Epoch별 검증 메트릭 추이 (핵심 결과)

| Epoch | 학습 step | eval_loss | **eval_wer** | Δ WER | 의미 |
|---|---|---|---|---|---|
| 1 | 95/760 | 1.892 | **41.91%** | (baseline) | 첫 검증, 학습 시작점 |
| 2 | 190/760 | 1.768 | **33.18%** | **-8.73%p** | 가장 큰 폭의 개선 |
| 3 | 285/760 | 1.747 | **30.49%** | -2.69%p | 빠른 개선 지속 |
| 4 | 380/760 | 1.737 | **27.91%** | -2.58%p | 안정적 개선 |
| 5 | 475/760 | 1.731 | **27.05%** | -0.86%p | 개선 폭 감소 시작 |
| 6 | 570/760 | 1.728 | **26.38%** | -0.67%p | 수렴 구간 진입 |
| 7 | 665/760 | 1.727 | **26.12%** | -0.26%p | 거의 평탄 |
| 8 | 760/760 | 1.726 | **26.27%** | **+0.15%p** | ⚠️ 약간 반등 |

### 핵심 관찰

- **총 개선폭**: WER 41.91% → 26.12% (epoch 7 최저점), **15.79%p 감소** ≈ 38% 상대 개선
- **수렴 패턴**: epoch 1~4 가파른 학습 → epoch 5~6 둔화 → epoch 7~8 평탄
- **Best 체크포인트**: **epoch 7 (WER 26.12%)** 이 최고 성능
- **마지막 epoch 8 반등**: 26.12% → 26.27% — **과적합 시작 신호**

---

### eval_steps와 eval_loss 이해

#### 평가에 사용되는 데이터

`eval_steps=95`마다 수행되는 평가는 **validation 데이터셋**을 사용한다. HuggingFace `Trainer`는 train/eval 데이터셋을 분리해서 관리하며, 평가 시점에는 **weight 업데이트 없이** eval_dataset 전체를 forward pass만 수행한다.

```python
trainer = Trainer(
    train_dataset=train_dataset,   # 학습에만 사용
    eval_dataset=eval_dataset,     # eval_steps마다 사용 (weight 업데이트 없음)
)
```

#### eval_loss의 의미

**eval_loss = validation 데이터셋에 대한 평균 Cross-Entropy Loss**

```
eval_loss = -1/N × Σ log P(정답 토큰 | 이전 토큰들, 음성)
```

| 구분 | 설명 |
|---|---|
| **train_loss** | 현재 학습 배치의 loss (weight 업데이트에 사용) |
| **eval_loss** | 학습에 쓰이지 않은 데이터에 대한 loss (일반화 성능 지표) |

#### eval_loss로 과적합 판단

```
train_loss ↓↓  eval_loss ↓↓          → 정상 학습
train_loss ↓↓  eval_loss → 또는 ↑    → 과적합 시작 신호
```

이번 학습에서 epoch 8이 정확히 이 패턴을 보였다:

| Epoch | train_loss | eval_loss | eval_wer |
|---|---|---|---|
| 7 | 3.494 | 1.727 | **26.12%** ← best |
| 8 | 3.489 ↓ | 1.726 ↓ | 26.27% ↑ ← **과적합** |

> eval_loss 자체는 미세하게 계속 줄었지만 실제 지표인 eval_wer가 반등했다. 이는 loss 감소가 실제 음성 인식 성능 개선으로 이어지지 않는 과적합 초기 단계임을 의미한다. **`load_best_model_at_end=True` 설정 시 eval_loss 또는 eval_wer 기준으로 best 체크포인트를 자동 선택**하므로, `metric_for_best_model: eval_wer` 명시를 권장한다.

---

## 3. 학습 메트릭 (Training Loss) 추이

| 시점 | epoch | train_loss | grad_norm | learning_rate |
|---|---|---|---|---|
| step 22 | 0.53 | 7.384 | 4.37 | 6.05e-05 (warmup ↑) |
| step ~100 | 1.05 | 4.969 | 2.23 | **9.71e-05** (lr 최고점) |
| step ~150 | 1.58 | 4.014 | 2.34 | 8.98e-05 (decay 시작) |
| step ~200 | 2.11 | 3.898 | 2.08 | 8.25e-05 |
| step ~250 | 2.63 | 3.786 | 1.86 | 7.52e-05 |
| step ~300 | 3.16 | 3.713 | 1.86 | 6.78e-05 |
| step ~350 | 3.68 | 3.678 | 1.87 | 6.05e-05 |
| step ~400 | 4.21 | 3.625 | 2.08 | 5.32e-05 |
| step ~450 | 4.74 | 3.608 | 2.01 | 4.59e-05 |
| step ~500 | 5.26 | 3.558 | 1.90 | 3.86e-05 |
| step ~550 | 5.79 | 3.548 | 1.67 | 3.13e-05 |
| step ~600 | 6.32 | 3.520 | 1.61 | 2.40e-05 |
| step ~650 | 6.84 | 3.509 | 1.60 | 1.67e-05 |
| step ~700 | 7.37 | 3.494 | 1.69 | 9.36e-06 |
| step ~750 | 7.90 | 3.489 | 1.69 | **2.05e-06** (lr 최저점) |

**최종 train_loss: 3.979** (전체 평균)

### 메트릭 개념 정리

#### train_loss

**현재 학습 배치에서 모델 예측이 정답과 얼마나 다른지를 나타내는 값**

```
train_loss = -1/N × Σ log P(정답 토큰 | 모델 예측)
```

- 값이 작을수록 현재 배치를 잘 맞추고 있다는 의미
- 매 `logging_steps`마다 기록되며, **배치 단위의 순간 loss**이므로 노이즈가 있음
- 전체 학습 경향은 이동 평균으로 보는 게 정확하다
- **단독으로는 과적합 판단 불가** — 반드시 eval_loss 또는 eval_wer와 함께 봐야 함

#### grad_norm (Gradient Norm)

**현재 스텝에서 모든 학습 파라미터의 gradient 벡터 크기 (L2 norm)**

```
grad_norm = √( Σ ||∇W_i||² )   (모든 레이어의 gradient를 이어 붙인 벡터의 크기)
```

- **크다** → 파라미터가 빠르게 변해야 한다는 신호 (손실 지형이 가파름)
- **작다** → 파라미터 변화가 미미함 (수렴 중이거나 평탄한 지형)
- `max_grad_norm=1.0` 설정 시 이 값이 1.0을 넘으면 **gradient clipping** 적용:
  ```
  실제 적용 gradient = gradient × (1.0 / grad_norm)   if grad_norm > 1.0
  ```
  로그에 찍히는 grad_norm은 **클리핑 전 원본 값**이므로, 이 값이 줄어드는 것은 모델이 수렴 중이라는 의미

---

### 세 메트릭의 동역학

**Learning Rate 곡선** — 의도한 스케줄대로 정확히 작동

```
6.05e-5 → 9.71e-5 (최고점, warmup 종료) → 점진적 감소 → 2.05e-6 (선형 decay 종료)
        ↑ warmup ↑                        ↑ linear decay ↑
```

**Grad Norm 안정화 추이**

```
4.37 → 2.2~2.3 → 1.6~1.9
초기 불안정    안정 구간    학습 후반 추가 안정화
```

`max_grad_norm=1.0`으로 항상 클리핑되지만, **클리핑 직전 원본 값이 점차 작아진다는 것은 모델이 안정적인 수렴 영역에 진입했다는 신호**.

**Train Loss 감소 패턴**

```
7.38 → 4.97 → 4.01 → 3.90 → 3.79 → 3.71 → ... → 3.49
가파른 감소  ─────────  완만한 감소  ─────────────  평탄화
```

---

### 메트릭 변화 패턴과 과적합 신호 감지

학습 상태는 train_loss, eval_loss(또는 eval_wer), grad_norm 세 가지를 **조합**해서 판단한다.

#### 패턴 분류표

| 패턴 | train_loss | eval_loss | grad_norm | 해석 |
|---|---|---|---|---|
| **정상 학습** | ↓ | ↓ | 점진적 안정 | 모델이 일반화 가능한 패턴을 학습 중 |
| **과적합 초기** | ↓ | → 또는 ↑ | 낮고 안정 | 훈련 데이터 외우기 시작 |
| **과적합 심화** | ↓↓ | ↑↑ | 낮고 안정 | 완전한 과적합, 즉시 중단 필요 |
| **학습률 과대** | 진동 | 진동 | 크고 불안정 | lr이 너무 높음, lr을 낮춰야 함 |
| **학습 정체** | → | → | 매우 작음 | 수렴 완료 또는 local minimum 진입 |
| **gradient 폭발** | 급등 또는 NaN | - | 갑자기 매우 큼 | lr 과대 또는 초기화 문제 |
| **gradient 소실** | 거의 안 변함 | 거의 안 변함 | 거의 0 | 깊은 층에 gradient가 전달 안 됨 |

#### 이번 학습에서 실제로 나타난 신호

```
Epoch 1~4:  train_loss ↓↓  eval_wer ↓↓  grad_norm 4.37→1.87  → 정상 학습
Epoch 5~7:  train_loss ↓   eval_wer ↓   grad_norm 1.90→1.60  → 수렴 진입
Epoch 8:    train_loss ↓   eval_wer ↑   grad_norm 1.69        → ⚠️ 과적합 초기
```

특이사항: eval_loss(1.727→1.726)는 미세하게 줄었지만 **eval_wer(26.12%→26.27%)는 반등**. loss 감소가 실제 성능 개선으로 이어지지 않는 단계임을 의미하므로, **primary metric은 task 지표(WER)로 설정하는 것이 중요**.

#### 조기 개입 기준 (실무 가이드)

| 조건 | 권장 조치 |
|---|---|
| eval_loss가 2 epoch 연속 증가 | Early stopping 발동 (`patience=2`) |
| eval_wer가 반등 + train_loss는 감소 | Best 체크포인트 저장 후 학습 종료 |
| grad_norm이 갑자기 5× 이상 급증 | lr 낮추거나 `max_grad_norm` 축소 |
| train_loss가 NaN | lr을 10× 낮추고 재시작 |
| grad_norm이 수 epoch 동안 거의 0 | lr 너무 낮거나 모델 구조 문제 확인 |

---

## 4. 단계별 의미 정리

### Stage 1: Warmup (step 0 ~ 76, epoch 0 ~ 0.8)
- lr이 0에서 1e-4로 선형 증가하는 구간
- 사전학습된 가중치 + 새로 추가된 LoRA 어댑터 모두를 충격으로부터 보호
- step 22 시점 grad_norm=4.37로 크게 나타남 → warmup이 없었다면 학습 초반 불안정 가능성 ↑

### Stage 2: 가파른 학습 (epoch 1 ~ 4)
- **WER 41.91% → 27.91%, 14%p 감소** (전체 개선의 88%가 여기서 발생)
- lr이 최고점 부근(9.7e-5 ~ 5.3e-5)에서 가장 활발한 업데이트
- LoRA 어댑터가 빠르게 도메인 특성 학습
- train_loss 4.97 → 3.62 (-27%)

### Stage 3: 미세 조정 (epoch 5 ~ 7)
- WER 27.05% → 26.12%, **단 0.93%p 개선** (개선 폭 급감)
- lr이 decay되며 weight 변화 폭 점점 작아짐
- 모델이 데이터로부터 짜낼 수 있는 정보를 거의 다 추출한 상태
- grad_norm도 1.6~1.7로 안정

### Stage 4: 수렴/약한 과적합 (epoch 8)
- WER 26.12% → 26.27%로 **소폭 반등**
- train_loss는 계속 감소(3.51 → 3.49)하는데 eval_wer는 악화 → 전형적인 과적합 초기 신호
- lr이 거의 0(2e-6)까지 떨어져 마지막 미세 조정 시도했으나 효과 없음

---

## 5. 검증 사이클 정보

- **검증셋 크기**: 약 880 샘플 (3.94 samples/sec × 224초 평균)
- **검증 1회 소요**: 약 3분 45초 (110 batches × 2초)
- **총 검증 횟수**: 8회 (`eval_steps=95` 설정)
- 검증 시점마다 체크포인트 저장 + Hugging Face config HEAD 요청 발생

---

## 6. 결과물 — 저장된 체크포인트

```
./checkpoints/whisper-tiny-en/
├── checkpoint-???/  (save_total_limit=3으로 최근 3개만 유지)
├── checkpoint-???/
├── checkpoint-???/
└── final_adapter/   ← 최종 LoRA 어댑터 (마지막 epoch 8 가중치)
```

추론 시 사용 방법(로그에서 안내):

```python
PeftModel.from_pretrained(base_model, "checkpoints/whisper-tiny-en/final_adapter")
    .merge_and_unload()
```

> ⚠️ **주의**: `final_adapter`는 epoch 8(WER 26.27%) 가중치다. 만약 학습 시 `load_best_model_at_end=True` 설정이 있었다면 best(epoch 7, WER 26.12%) 가중치가 들어갔을 텐데, 로그에서 명시적 확인이 안 되니 체크포인트 디렉토리를 확인해보는 게 좋다.

---

## 7. 전체 평가 — 잘 된 점과 개선 여지

### ✅ 잘 된 점

1. **학습이 매우 안정적**: loss·grad_norm·lr 모든 지표가 의도된 방향으로 일관되게 움직임
2. **LoRA 효과 입증**: 5.5% 파라미터만 학습해서 WER 38% 상대 개선 달성
3. **6GB GPU에서 완주**: GTX 1660 SUPER 같은 소비자 GPU로도 LoRA + fp16 + grad accumulation 조합 덕에 90분에 완료
4. **하이퍼파라미터 적절**: lr=1e-4, rank=32, warmup_ratio=0.1 모두 LoRA에 적합한 설정

### ⚠️ 개선 여지

1. **Early stopping 미적용**: epoch 7이 best였는데 8까지 진행해서 약간의 과적합 발생. `EarlyStoppingCallback(patience=2)` 추가 권장
2. **Epoch 수 과다 가능성**: 마지막 1~2 epoch의 marginal gain이 거의 0. **6 epoch도 충분**했을 듯 (시간 18% 절약)
3. **WER 26%는 여전히 높음**: 추가 개선 여지 — 데이터 추가, rank 증가(64), 또는 더 큰 베이스 모델(small/medium)로 업그레이드 고려
4. **`load_best_model_at_end=True` 확인 필요**: 설정 안 되어 있다면 epoch 7 체크포인트를 수동 선택 필요

---

## Part 1 한 줄 결론

**Whisper-tiny에 LoRA를 붙여 영어 음성 도메인에 적응시키는 데 성공한 정석적인 학습이다.** WER을 절대값 약 16%p (41.91 → 26.12) 줄였고, 모든 학습 지표가 교과서적인 패턴을 보였다. 다음 실험에서는 (1) early stopping 추가, (2) epoch 6으로 축소, (3) 더 큰 베이스 모델로 업그레이드를 고려해볼 만하다.

---

# Part 2. Epoch를 늘리면 WER이 개선될까?

> **결론부터**: 현재 설정에서 epoch를 늘리는 것만으로는 WER 개선 가능성이 낮다. 오히려 과적합이 심해질 가능성이 더 높다.

## 1. 로그가 말해주는 신호 — "이미 수렴했다"

epoch별 WER 개선 폭(Δ)을 다시 보면:

```
Epoch 1 → 2:  -8.73%p   ████████████████████
Epoch 2 → 3:  -2.69%p   ██████
Epoch 3 → 4:  -2.58%p   ██████
Epoch 4 → 5:  -0.86%p   ██
Epoch 5 → 6:  -0.67%p   █▌
Epoch 6 → 7:  -0.26%p   ▌
Epoch 7 → 8:  +0.15%p   ← 반등 ⚠️
```

**개선 곡선이 epoch 4부터 명확히 평탄화**되고, **epoch 8에서 이미 반등**이 시작됐다. 이건 더 학습해도 좋아질 가능성보다 나빠질 가능성이 큰 상태다.

## 2. train_loss vs eval_wer의 결정적 단서

이 부분이 진단의 핵심이다:

| Epoch | train_loss | eval_wer |
|---|---|---|
| 7 | 3.494 | 26.12% ← best |
| 8 | 3.489 | 26.27% ← **악화** |

**train_loss는 계속 감소하는데 eval_wer가 다시 올라가기 시작**했다는 건 교과서적인 **과적합 시작 신호**다. 즉 모델이 더 이상 "일반화 가능한 패턴"이 아니라 "훈련 데이터만의 특성"을 학습하기 시작했다. epoch를 더 늘리면:

```
train_loss:  계속 감소  ↓↓↓
eval_wer:    계속 증가  ↑↑↑  ← 더 나빠짐
```

이 발산이 가속화될 가능성이 높다.

## 3. Learning Rate 측면에서도 epoch 추가가 무의미한 이유

현재 학습 종료 시점의 lr:

```
step 750 (마지막 로그):  lr = 2.05e-06
```

이미 거의 0에 가까운 값이다. 현재 스케줄러는 760 스텝에 맞춰 1e-4 → 0으로 decay되도록 설정되어 있어서, **epoch만 늘리면 lr이 0인 채로 추가 스텝을 도는 셈**이다. 이건 사실상 학습이 안 되는 상태다.

epoch를 의미 있게 늘리려면 lr 스케줄도 함께 재설계해야 한다:

```yaml
# 단순히 epoch만 늘리면 ❌
num_epochs: 12   # lr 스케줄은 여전히 8 epoch 기준 → 마지막 4 epoch는 lr≈0

# 같이 조정해야 함 ✅
num_epochs: 12
lr_scheduler_type: cosine_with_restarts  # 또는 스케줄 재설정
```

## 4. WER을 더 개선하려면? — 영향력 순

epoch 증가는 우선순위가 매우 낮다. 효과가 큰 것부터:

### 🥇 1순위: 더 큰 베이스 모델

```
whisper-tiny  (39M)  → 현재 WER 26%
whisper-small (244M) → 보통 WER 10~15%대 가능
whisper-medium(769M) → 보통 WER 8~12%대 가능
```

LoRA 어댑터가 아무리 좋아도 **베이스 모델의 capacity 한계**가 있다. tiny에서 26%는 모델 크기 대비 거의 한계치일 수 있다.

### 🥈 2순위: 데이터 추가

현재 약 1.5K 샘플(주석 기준)에서 학습. **데이터를 2~3배로 늘리면** WER 개선 폭이 가장 크다. 특히:
- 검증셋에서 자주 틀리는 패턴(긴 발화, 특정 발음, 도메인 용어) 보강
- Data augmentation 강화 (현재 SpecAugment는 적용 중이지만 추가로 noise injection, speed perturbation 등)

### 🥉 3순위: LoRA capacity 증가

```yaml
lora_r: 32 → 64      # 학습 가능 파라미터 2배
lora_alpha: 64 → 128 # 비율 유지
```

현재 2.2M 학습 파라미터를 4.4M으로 늘려 표현력 증가. **데이터가 충분하다는 전제 하에** 효과 있음.

### 4순위: 학습 전략 개선

- `load_best_model_at_end=True` + `EarlyStoppingCallback(patience=3)`
- Cosine annealing with warm restarts (lr 재시작으로 local minimum 탈출 시도)
- Label smoothing factor 조정 (0.1 → 0.05)

### 5순위 (마지막): Epoch 조정

**그래도 epoch를 늘리고 싶다면** 다음 조합으로:

```yaml
num_epochs: 12               # 8 → 12
learning_rate: 5.0e-5        # 1e-4 → 5e-5로 낮춰서 천천히 학습
lr_scheduler_type: cosine    # linear → cosine (후반 lr이 너무 빨리 0되지 않음)
warmup_ratio: 0.05           # 0.1 → 0.05 (warmup 짧게)

# 그리고 반드시 추가
load_best_model_at_end: true
metric_for_best_model: eval_wer
greater_is_better: false
# EarlyStoppingCallback(early_stopping_patience=3)
```

이 조합이면 더 긴 학습 동안 과적합을 방지하면서 best 시점을 자동 저장한다. 하지만 **개선 폭은 0.5~1%p 정도일 가능성**이 높다. tiny 모델의 한계 때문이다.

## 5. 실험 우선순위 추천

같은 시간/비용을 투자한다면 효과 큰 순서로:

| 우선순위 | 실험 | 예상 WER | 비고 |
|---|---|---|---|
| 1 | **whisper-small + 동일 LoRA 설정으로 재학습** | 10~15% | 1660 SUPER에서도 batch_size=4로 돌릴 수 있음. 시간 2~3배지만 개선 폭 압도적 |
| 2 | **현재 tiny 유지 + 데이터 2배 + 6 epoch + early stopping** | 23~25% | 데이터 수집이 가능하다면 가장 가성비 좋음 |
| 3 | **rank=64로 늘려서 재학습** | 25~26% | 코드 한두 줄 변경으로 빠른 실험 |
| 4 | **epoch만 12로 늘리기** | 26~27% | 개선 보장 없음, 권장하지 않음 |

---

## Part 2 한 줄 결론

**현재 로그는 "tiny 모델로 1.5K 데이터에서 짜낼 수 있는 거의 한계치(WER 26%)에 도달했다"고 말하고 있다.** epoch 추가는 이 한계를 못 넘는다. WER을 의미 있게 개선하려면 **베이스 모델을 키우거나 데이터를 늘리는 게** 정공법이다. 둘 다 어렵다면 LoRA rank를 늘려 capacity를 키우는 게 차선책이고, epoch 증가는 마지막 옵션이다. 그마저도 lr 스케줄 재설계 + early stopping이 함께 가야 효과가 있다.

---

# 종합 핵심 메시지

1. **이번 학습은 성공적**이다. WER 41.91% → 26.12%로 약 38% 상대 개선 달성.
2. **모델은 이미 수렴**했다. epoch 4 이후 marginal gain 급감, epoch 8에서 과적합 시작.
3. **다음 실험의 정공법은 epoch 증가가 아닌 모델 업그레이드 또는 데이터 추가**다.
4. **즉시 적용 가능한 개선**: `load_best_model_at_end=True` + `EarlyStoppingCallback(patience=2)` 추가로 best 체크포인트 자동 저장 + 학습 시간 절약.
