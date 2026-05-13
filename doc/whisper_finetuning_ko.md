# Whisper 파인튜닝 가이드 (Hugging Face 블로그 한국어 정리)

원문: https://huggingface.co/blog/fine-tune-whisper

---

# 1. 개요

## Whisper란?

Whisper 는 OpenAI가 공개한 Transformer 기반 음성인식(ASR) 모델이다.

특징:

- 약 68만 시간(680k hours) 규모의 라벨링된 음성 데이터로 사전학습
- 96개 이상의 언어 지원
- multilingual ASR 성능 우수
- low-resource language(데이터가 적은 언어)에서도 강력한 성능
- Encoder-Decoder 구조의 Seq2Seq 모델 사용

---

# 2. Whisper 구조

Whisper는 다음 구조를 가진다.

```text
Raw Audio
   ↓
Log-Mel Spectrogram
   ↓
Transformer Encoder
   ↓
Hidden States
   ↓
Transformer Decoder
   ↓
Text Tokens
```

## 핵심 특징

### Encoder
- 음성 특징(feature) 추출

### Decoder
- 이전 토큰과 Encoder hidden state를 기반으로
- autoregressive 방식으로 텍스트 생성

### 학습 방식
- Cross Entropy Loss 사용
- End-to-End Speech-to-Text 학습

---

# 3. Whisper 모델 크기

| 모델 | 파라미터 |
|---|---|
| tiny | 39M |
| base | 74M |
| small | 244M |
| medium | 769M |
| large | 1550M |

---

# 4. Whisper 파인튜닝 전체 흐름

```text
Dataset 준비
   ↓
Audio Resampling
   ↓
Feature Extraction
   ↓
Tokenizer Encoding
   ↓
Whisper Model Load
   ↓
Trainer 구성
   ↓
Fine-tuning
   ↓
WER 평가
   ↓
Inference / Demo
```

---

# 5. 환경 구성

```bash
pip install --upgrade pip
pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio
```

---

# 6. 데이터셋 준비

예제에서는 Mozilla Common Voice 데이터셋 사용.

```python
{
  "audio": audio_file,
  "sentence": transcription
}
```

---

# 7. Audio 전처리

Whisper는 입력 음성을:

- 16kHz로 리샘플링
- 최대 30초 길이로 맞춤
- Log-Mel Spectrogram 변환

을 수행한다.

---

# 8. Feature Extractor & Tokenizer

## Feature Extractor

```python
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-small"
)
```

## Tokenizer

```python
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Korean",
    task="transcribe"
)
```

## Processor 통합

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Korean",
    task="transcribe"
)
```

---

# 9. 데이터 전처리 함수

```python
def prepare_dataset(batch):

    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = tokenizer(
        batch["sentence"]
    ).input_ids

    return batch
```

---

# 10. Dataset map 처리

```python
dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names["train"],
    num_proc=4
)
```

---

# 11. 모델 로딩

```python
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
)
```

---

# 12. 평가 지표 (WER)

WER 공식:

```text
WER = (S + D + I) / N
```

| 기호 | 의미 |
|---|---|
| S | Substitution |
| D | Deletion |
| I | Insertion |
| N | 전체 단어 수 |

---

# 13. TrainingArguments

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-ko",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    predict_with_generate=True,
    generation_max_length=225,
)
```

---

# 14. 한국어 Whisper 파인튜닝 시 고려사항

## 숫자 인식 강화

금융/교육 도메인에서는 숫자 인식 오류가 많으므로:

- 숫자 포함 데이터 강화
- LM Rescoring 병행

추천.

---

# 15. LoRA 기반 Whisper 파인튜닝

최근에는:

- LoRA
- QLoRA
- PEFT

기반 학습이 많이 사용된다.

장점:

- VRAM 절약
- 빠른 학습
- Mac/MPS 환경 가능

---

# 16. 실제 현업 팁

## 데이터가 적을 때
- small 모델 추천
- augmentation 적극 활용

## 도메인 특화
- 금융
- 의료
- 교육
- 콜센터

등은 fine-tuning 효과 큼

## 긴 음성 처리
- VAD
- chunking
- LM rescoring

조합 필요

---

# 17. 추가 참고 자료

- Hugging Face Whisper Fine-tuning Blog
- Distil-Whisper
- WhisperX
- PEFT / LoRA
- LM Rescoring
