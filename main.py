"""
main.py - Whisper 파인튜닝 메인 실행 스크립트

config.yaml을 통해 다음을 선택적으로 구성할 수 있다:
  - 데이터 소스: 로컬 파일(wav/txt) 또는 HuggingFace Hub
  - 증강 방식: offline(사전 생성+저장) 또는 on_the_fly(매 에폭 실시간)

전체 파이프라인:
  1단계: 데이터 로드 (로컬 CSV manifest 또는 HF Hub)
  2단계: 불필요 컬럼 제거
  3단계: 데이터 증강 (offline 또는 on-the-fly)
  4단계: 전처리 (log-mel + 토크나이징)
  5단계: 모델 초기화 및 학습
"""

import os
from pathlib import Path

from config_loader import load_config
from load_speechdb import build_and_load_enuma_dataset_basedon_csv, load_from_huggingface
from preprocess_datasetdict_batch import (
    remove_columns_from_datasetdict,
    preprocess_datasetdict_batched,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from augumented_audio import (
    augment_example,
    load_noise_from_dir,
    load_rir_from_dir,
    make_on_the_fly_transform,
    make_eval_transform,
)
from access_whisperobj import get_tokenizer, get_processor, get_model
from loggerinterface import get_logger
from datasets import DatasetDict, load_from_disk
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

logger = get_logger(__name__)

DISK_ROOT = Path.cwd() / "storage"
os.makedirs(DISK_ROOT, exist_ok=True)


def make_compute_metrics(tokenizer):
    """WER 평가 메트릭 클로저를 반환한다."""
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ============================================================
# 1단계: 데이터 로드 (config.data.source에 따라 분기)
# ============================================================

def load_dataset_from_config(cfg: dict) -> DatasetDict:
    """config의 data.source 설정에 따라 로컬 또는 HuggingFace에서 데이터를 로드한다."""
    source = cfg["data"]["source"]

    if source == "local":
        local_cfg = cfg["data"]["local"]
        db_path = Path(local_cfg["db_path"])
        raw_cache = Path(local_cfg["raw_cache_dir"])
        sr = local_cfg["sampling_rate"]

        if not raw_cache.exists():
            logger.info(f"로컬 데이터 로드: {db_path}")
            dsd = build_and_load_enuma_dataset_basedon_csv(str(db_path), str(raw_cache), sr)
            dsd, _ = remove_columns_from_datasetdict(dsd)
            dsd.save_to_disk(str(raw_cache))
            logger.info(f"DatasetDict 저장 완료: {raw_cache}")
        else:
            dsd = load_from_disk(str(raw_cache))
            logger.info(f"캐시된 DatasetDict 로드: {raw_cache}")
        return dsd

    elif source == "huggingface":
        hf_cfg = cfg["data"]["huggingface"]
        dsd = load_from_huggingface(
            dataset_id=hf_cfg["dataset_id"],
            subset=hf_cfg.get("subset"),
            audio_column=hf_cfg.get("audio_column", "audio"),
            text_column=hf_cfg.get("text_column", "sentence"),
            sampling_rate=hf_cfg.get("sampling_rate", 16000),
        )
        dsd, _ = remove_columns_from_datasetdict(dsd)
        return dsd


# ============================================================
# 2단계: 증강 + 전처리 (config.augmentation.mode에 따라 분기)
# ============================================================

def prepare_datasets_offline(dsd: DatasetDict, cfg: dict, processor) -> DatasetDict:
    """
    오프라인 증강: train 데이터에 증강을 사전 적용하고 디스크에 저장한다.
    이후 모든 split에 log-mel + 토크나이징 전처리를 적용한다.
    """
    aug_cfg = cfg["augmentation"]
    cache_dir = Path(aug_cfg["offline_cache_dir"])

    if not cache_dir.exists():
        logger.info("오프라인 증강 시작 (train only)...")
        dataset_dict_aug = DatasetDict({
            "train": dsd["train"].map(
                augment_example,
                num_proc=aug_cfg.get("num_proc", 4),
            ),
            "validation": dsd["validation"],
            "test": dsd["test"],
        })
        dataset_dict_aug.save_to_disk(str(cache_dir))
        logger.info(f"증강 데이터 저장 완료: {cache_dir}")
    else:
        dataset_dict_aug = load_from_disk(str(cache_dir))
        logger.info(f"캐시된 증강 데이터 로드: {cache_dir}")

    # 전체 split에 log-mel + 토크나이징 전처리 적용
    prep_cfg = cfg["preprocessing"]
    dsd_proc = preprocess_datasetdict_batched(
        dataset_dict_aug,
        processor=processor,
        audio_col="audio",
        text_col="sentence",
        num_proc=prep_cfg.get("num_proc"),
        batch_size=prep_cfg.get("batch_size", 32),
        remove_others=False,
        feature_dtype=prep_cfg.get("feature_dtype", "float32"),
        max_target_len=prep_cfg.get("max_target_len"),
        truncation=prep_cfg.get("truncation", True),
    )
    return dsd_proc


def prepare_datasets_on_the_fly(dsd: DatasetDict, cfg: dict, processor) -> DatasetDict:
    """
    On-the-fly 증강: train 데이터에 set_transform()을 설정하여
    매 샘플 접근 시 랜덤 증강 + 전처리를 실시간으로 적용한다.
    validation/test는 증강 없이 전처리만 적용한다.
    """
    aug_cfg = cfg["augmentation"]

    # 노이즈 파일 사전 로드 (noise 증강이 활성화된 경우)
    loaded_noise_bank = []
    if aug_cfg["noise"]["enabled"]:
        noise_dir = aug_cfg["noise"]["noise_dir"]
        if os.path.isdir(noise_dir):
            loaded_noise_bank = load_noise_from_dir(noise_dir)
            logger.info(f"노이즈 뱅크 로드 완료: {len(loaded_noise_bank)}개 파일")
        else:
            logger.warning(f"노이즈 디렉토리가 없습니다: {noise_dir} (노이즈 증강 스킵)")

    # RIR 파일 사전 로드 (reverb 증강이 활성화된 경우)
    loaded_rir_bank = []
    reverb_cfg = aug_cfg.get("reverb", {})
    if reverb_cfg.get("enabled", False):
        rir_dir = reverb_cfg["rir_dir"]
        if os.path.isdir(rir_dir):
            loaded_rir_bank = load_rir_from_dir(rir_dir)
            logger.info(f"RIR 뱅크 로드 완료: {len(loaded_rir_bank)}개 파일")
        else:
            logger.warning(f"RIR 디렉토리가 없습니다: {rir_dir} (잔향 증강 스킵)")

    # train: on-the-fly transform 설정 (증강 + log-mel + 토크나이징)
    train_transform = make_on_the_fly_transform(processor, aug_cfg, loaded_noise_bank, loaded_rir_bank)
    dsd["train"].set_transform(train_transform)
    logger.info("On-the-fly 증강 transform 설정 완료 (train)")

    # validation/test: 증강 없이 전처리만 (set_transform으로 실시간 처리)
    eval_transform = make_eval_transform(processor)
    dsd["validation"].set_transform(eval_transform)
    dsd["test"].set_transform(eval_transform)
    logger.info("Eval transform 설정 완료 (validation, test)")

    return dsd


# ============================================================
# 3단계: 모델 초기화
# ============================================================

def setup_model(cfg: dict, processor):
    """Whisper 모델을 초기화하고 generation config를 설정한다."""
    model_cfg = cfg["model"]
    model = get_model(model_cfg["name"])

    # 추론/평가 시 디코더 설정
    model.generation_config.language = model_cfg["language"]
    model.generation_config.task = model_cfg["task"]
    model.generation_config.forced_decoder_idx = processor.get_decoder_prompt_ids(
        language=model_cfg["language"], task=model_cfg["task"]
    )

    # pad_token 설정
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # decoder_start_token_id 설정
    sot_id = tok.convert_tokens_to_ids("<|startoftranscript|>")
    model.config.decoder_start_token_id = sot_id

    return model


# ============================================================
# 메인 실행
# ============================================================

if __name__ == '__main__':
    # 설정 파일 로드
    cfg = load_config("config.yaml")

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # Whisper tokenizer, processor 로드
    success, tokenizer = get_tokenizer(model_cfg["name"], model_cfg["language"], model_cfg["task"])
    if not success:
        raise RuntimeError(f"Tokenizer 로드 실패: {tokenizer}")

    success, processor = get_processor(model_cfg["name"], model_cfg["language"], model_cfg["task"])
    if not success:
        raise RuntimeError(f"Processor 로드 실패: {processor}")
    logger.info("Tokenizer, Processor 로드 성공")

    # 1단계: 데이터 로드
    dsd = load_dataset_from_config(cfg)

    # 2단계: 증강 + 전처리 (config에 따라 분기)
    aug_mode = cfg["augmentation"]["mode"]
    if aug_mode == "offline":
        dsd_proc = prepare_datasets_offline(dsd, cfg, processor)
        train_dataset = dsd_proc["train"]
        eval_dataset = dsd_proc["test"]
    elif aug_mode == "on_the_fly":
        dsd_proc = prepare_datasets_on_the_fly(dsd, cfg, processor)
        train_dataset = dsd_proc["train"]
        eval_dataset = dsd_proc["test"]

    logger.info(f"데이터 준비 완료 (mode={aug_mode})")

    # 3단계: 모델 초기화
    model = setup_model(cfg, processor)

    # DataCollator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 4단계: 학습 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        label_smoothing_factor=train_cfg["label_smoothing_factor"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        logging_strategy="steps",
        logging_steps=train_cfg["logging_steps"],
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        dataloader_prefetch_factor=None,
        dataloader_persistent_workers=False,
        group_by_length=False,
        length_column_name="input_length",
        remove_unused_columns=False,
        fp16=train_cfg.get("fp16", True),
        seed=train_cfg.get("seed", 42),
    )

    # 5단계: Trainer 구성 및 학습 실행
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
        processing_class=processor,
    )

    logger.info("학습 시작")
    trainer.train()
