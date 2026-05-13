# Whisper Precision: Whisper 모델 파인튜닝 및 하이퍼파라미터 튜닝 가이드 (한글 요약)

원문:
- Medium 기사: https://medium.com/@chris.xg.wang/a-guide-to-fine-tune-whisper-model-with-hyper-parameter-tuning-c13645ba2dba
- 참고 HuggingFace 블로그: https://huggingface.co/blog/fine-tune-whisper

---

## 개요

이 문서는 Whisper 음성인식 모델을 특정 도메인 데이터에 맞게 파인튜닝(Fine-tuning)하는 과정을 한국어로 정리한 요약본이다.

주요 내용:
- WAV 음성 데이터셋 구성
- Whisper 입력 Feature 생성
- Label Tokenizing
- HuggingFace Trainer 기반 학습
- TensorBoard 기반 모니터링
- Optuna + Ray 기반 하이퍼파라미터 튜닝

---

# 1. 데이터셋 준비

작성자는 CSV 파일 기반으로 음성 경로와 정답 텍스트를 로딩한다.

예시:
```python
from datasets import load_dataset, DatasetDict
df = pd.read_csv('./data/audio_list.csv')
dataset = Dataset.from_pandas(df)
```

CSV 예시:

| file_path | whole_sentence |
|---|---|
| audio1.wav | 안녕하세요 |
| audio2.wav | Whisper 학습 예제입니다 |

---

# 2. 오디오 Feature 추출

Whisper는 기본적으로 16kHz 입력을 사용한다.

따라서 모든 음성을 16kHz로 리샘플링한다.

예시:
```python
def extract_features(batch):
    waveform, sr = librosa.load(batch["file_path"], sr=16000)
    batch["speech"] = waveform
    return batch
```

핵심 포인트:
- Whisper 입력 샘플레이트는 16000Hz
- librosa를 이용해 waveform 추출
- 음성을 mel feature 형태로 변환

---

# 3. 텍스트 Tokenize

정답 문장을 Whisper tokenizer로 토큰화한다.

예시:
```python
def tokenize_label(batch):
    audio_inputs = processor.feature_extractor(
        batch["speech"],
        sampling_rate=16000,
        return_tensors="pt"
    )

    batch["input_features"] = audio_inputs.input_features[0]

    labels = processor.tokenizer(
        batch["whole_sentence"],
        return_tensors="pt",
        padding="longest"
    )

    batch["labels"] = labels.input_ids[0]
    return batch
```

핵심:
- 입력: 음성 feature
- 출력: 토큰 ID sequence

---

# 4. Train / Validation / Test 분리

데이터셋을 학습/검증/테스트로 분리한다.

예시:
```python
dataset = dataset.train_test_split(test_size=0.2)

train_val_split = dataset['train'].train_test_split(test_size=0.1)

train_dataset = train_val_split['train']
val_dataset = train_val_split['test']
test_dataset = dataset['test']
```

일반적인 권장 비율:
- Train: 70~80%
- Validation: 10~15%
- Test: 10~15%

---

# 5. Data Collator 구성

Whisper 학습 시 길이가 서로 다른 sequence를 padding 처리해야 한다.

예시:
```python
class DataCollator:
    def __call__(self, features):
        input_features = [
            torch.tensor(feature["input_features"]).squeeze(0)
            for feature in features
        ]

        labels = [
            torch.tensor(feature["labels"]).squeeze(0)
            for feature in features
        ]

        input_features_padded = pad_sequence(
            input_features,
            batch_first=True
        )

        labels_padded = pad_sequence(
            labels,
            batch_first=True
        )

        return {
            "input_features": input_features_padded,
            "labels": labels_padded
        }
```

---

# 6. Whisper 학습 설정

HuggingFace `Seq2SeqTrainingArguments`를 사용한다.

예시 주요 설정:

```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small",
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=400,
    eval_steps=50,
    logging_steps=100,
    gradient_checkpointing=True,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False
)
```

핵심 옵션 설명:

| 옵션 | 설명 |
|---|---|
| learning_rate | 학습률 |
| fp16 | mixed precision 학습 |
| gradient_checkpointing | 메모리 절약 |
| metric_for_best_model | WER 기준 최적 모델 선택 |
| generation_max_length | 최대 디코딩 길이 |

---

# 7. Trainer 구성

```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DataCollator,
    tokenizer=processor.feature_extractor,
)
```

학습 시작:
```python
trainer.train()
```

---

# 8. 모델 저장

```python
model.save_pretrained("./whisper_finetuned")
processor.save_pretrained("./whisper_finetuned")
```

저장 결과:
- 모델 weight
- tokenizer
- feature extractor
- config

---

# 9. TensorBoard 사용

학습 로그를 시각화한다.

```python
%load_ext tensorboard
%tensorboard --logdir ./logs
```

확인 가능한 항목:
- loss
- learning rate
- WER
- validation score

---

# 10. Optuna 기반 Hyperparameter Tuning

Optuna를 사용하여 자동 하이퍼파라미터 탐색 수행.

튜닝 대상:
- learning_rate
- batch_size
- epoch 수

예시:
```python
learning_rate = trial.suggest_loguniform(
    'learning_rate',
    1e-5,
    1e-4
)

batch_size = trial.suggest_categorical(
    'per_device_train_batch_size',
    [2, 5, 10]
)
```

목표:
- Validation WER 최소화

---

# 11. Ray Dashboard 활용

Ray + Optuna 조합으로 병렬 탐색 가능.

구성:
- Optuna: 탐색 알고리즘
- Ray Tune: 분산 실행
- ASHA Scheduler: early stopping

예시:
```python
analysis = tune.run(
    objective,
    resources_per_trial={"cpu": 1, "gpu": 1},
    metric="eval_wer",
    search_alg=optuna_search,
    scheduler=scheduler,
    num_samples=20,
)
```

---

# 12. 실무 관점 핵심 정리

## 중요한 포인트

### 1) 데이터 품질이 가장 중요
Whisper 성능은 데이터 품질 영향을 크게 받는다.

특히:
- 정답 오타
- 발화 누락
- alignment 오류
- 노이즈

등은 WER 악화 원인이 된다.

---

### 2) 작은 데이터셋은 과적합 위험
수 시간 수준 데이터만으로 과도한 파인튜닝 시:
- 일반 성능 저하
- hallucination 증가
- 특정 화자 편향

이 발생 가능.

---

### 3) 도메인 특화가 핵심
금융/의료/콜센터 등 특정 도메인에서는:
- 용어
- 숫자
- 고유명사
- 문장 패턴

적응이 중요하다.

---

### 4) Whisper Large 무조건 정답 아님
실제 환경에서는:
- small
- medium

모델이 더 안정적인 경우도 존재.

---

# 13. 추가적으로 고려할 부분

## 추천 전략

### LM Rescoring
Whisper 후단에 KenLM 같은 언어모델 적용.

효과:
- 숫자 인식 향상
- 도메인 용어 강화
- 문장 안정성 개선

---

### LoRA / PEFT
전체 weight 학습 대신 일부 adapter만 학습.

장점:
- GPU 메모리 절감
- 빠른 학습
- checkpoint 경량화

---

### 데이터 증강
합성 음성(TTS) 기반 augmentation 가능.

예:
- 잡음 추가
- speed perturbation
- reverberation
- TTS synthetic speech

---

# 14. 참고 자료

## HuggingFace 공식 가이드
https://huggingface.co/blog/fine-tune-whisper

## OpenAI Whisper
https://github.com/openai/whisper

## Transformers
https://github.com/huggingface/transformers

## PEFT
https://github.com/huggingface/peft

## Ray Tune
https://docs.ray.io/en/latest/tune/index.html

## Optuna
https://optuna.org/

---

# 요약

이 글은 Whisper 기반 음성인식 모델을:
- 데이터셋 구축
- Feature 생성
- Trainer 기반 학습
- WER 평가
- Hyperparameter tuning
- Ray 기반 분산 탐색

까지 전체 파이프라인 관점에서 설명한다.

실제 실무에서는:
- 데이터 품질
- 도메인 적합성
- decoding 전략
- LM rescoring
- PEFT/LoRA

가 성능 개선에 매우 큰 영향을 준다.
