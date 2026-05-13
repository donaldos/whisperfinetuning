# 참고 자료

Whisper 파인튜닝 학습 설정에 관한 주요 참고 사이트 모음.

---

## 1. HuggingFace 공식 블로그 (가장 추천)

**[Fine-Tune Whisper For Multilingual ASR with Transformers](https://huggingface.co/blog/fine-tune-whisper)**

- Whisper 파인튜닝의 **공식 가이드**에 해당
- 데이터 준비 → 전처리 → `Seq2SeqTrainingArguments` 설정 → 학습까지 전 과정을 코드와 함께 설명
- learning_rate, warmup, fp16, gradient_checkpointing 등 핵심 파라미터 설명 포함

## 2. HuggingFace Audio Course

**[Fine-tuning the ASR model](https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning)**

- HuggingFace의 **오디오 과정 Chapter 5**
- 교육 목적으로 구성되어 초보자에게 친절한 설명
- Whisper 모델 구조부터 학습 설정까지 단계별 학습 가능

## 3. Whisper Hyperparameter Tuning 가이드

**[Whisper Precision: A Guide to Fine-Tune with Hyperparameter Tuning](https://medium.com/@chris.xg.wang/a-guide-to-fine-tune-whisper-model-with-hyper-parameter-tuning-c13645ba2dba)**

- **하이퍼파라미터 튜닝**에 특화된 가이드
- learning_rate, batch_size, dropout 등 파라미터별 최적값 탐색 방법 설명
- 실험 결과 기반으로 파라미터 영향도 분석

## 4. LearnOpenCV 튜토리얼

**[Fine Tuning Whisper on Custom Dataset](https://learnopencv.com/fine-tuning-whisper-on-custom-dataset/)**

- 커스텀 데이터셋 기반 파인튜닝에 초점
- 시각적 자료가 풍부하고 코드 설명이 상세

## 5. OpenAI Whisper 원본 논문

**[Robust Speech Recognition via Large-Scale Weak Supervision (PDF)](https://cdn.openai.com/papers/whisper.pdf)**

- Whisper 모델의 **설계 철학과 학습 전략** 원본
- 학습 파라미터 선택의 근거를 이해하는 데 유용

## 6. HuggingFace 공식 문서

**[Whisper Model Documentation](https://huggingface.co/docs/transformers/model_doc/whisper)**

- `WhisperForConditionalGeneration`, `WhisperProcessor` 등 API 레퍼런스
- 파라미터 기본값과 설명이 정확

---

## 목적별 추천

| 목적 | 추천 사이트 |
|------|-----------|
| 처음 시작하기 | HuggingFace Audio Course |
| 실전 코드 따라하기 | HuggingFace 공식 블로그 |
| 파라미터 튜닝 깊이 이해 | Medium 하이퍼파라미터 가이드 |
| 이론적 근거 | OpenAI 원본 논문 |
| API 레퍼런스 | HuggingFace 공식 문서 |
