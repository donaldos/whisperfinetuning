from load_speechdb import build_and_load_enuma_dataset_basedon_csv
from preprocess_datasetdict_batch import remove_columns_from_datasetdict
from augumented_audio import augment_example
from loggerinterface import get_logger
from datasets import DatasetDict, load_from_disk
import os
from access_whisperobj import get_tokenizer, get_processor, get_model
from preprocess_datasetdict_batch import remove_columns_from_datasetdict,apply_audio_text_filters, preprocess_datasetdict_batched, DataCollatorSpeechSeq2SeqWithPadding
from transformers import (
    PreTrainedTokenizerBase,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    IntervalStrategy,
    Seq2SeqTrainer,
)
import evaluate
from pathlib import Path


logger = get_logger(__name__)
DISK_ROOT = Path.cwd() / "storage"
os.makedirs(DISK_ROOT, exist_ok=True)


model_name = "openai/whisper-tiny"
lang_code = 'en'
task = 'transcribe'

def make_compute_metrics(tokenizer: WhisperTokenizer):
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


if __name__=='__main__':

    DB_PATH = DISK_ROOT / 'EnumaSpeech_01'    
    RAW_DATA_PATH = DISK_ROOT / 'raw_enuma_speech'
    AUGUMENT_DATA_PATH = DISK_ROOT / 'augument_enuma_speech'

    # 일반 음성데이터베이스를 기반으로 CSV 파일 만들고 DatasetDict 생성
    if not os.path.exists(RAW_DATA_PATH):
        # 읿반적인 음성데이터와 그 전사데이터를 csv 파일을 생성 DatasetDict 형태로 변환
        enuma_voice_dict = build_and_load_enuma_dataset_basedon_csv(DB_PATH,RAW_DATA_PATH,16000)
        
        # DatasetDict 에서 필요없는 컬럼 제거
        enuma_voice_datasetdict, _  = remove_columns_from_datasetdict(enuma_voice_dict)
        enuma_voice_datasetdict.save_to_disk(RAW_DATA_PATH)
        logger.info(f"save_to_disk:{enuma_voice_datasetdict}")
    else:
        enuma_voice_datasetdict = load_from_disk(RAW_DATA_PATH)
        logger.info(f"laod_from_dataset:{enuma_voice_datasetdict}")


    if not os.path.exists(AUGUMENT_DATA_PATH):
    # 1) 증강 적용
        dataset_dict_aug = DatasetDict({
            "train": enuma_voice_datasetdict["train"].map(
                augment_example,
                num_proc=12,           # CPU 코어 수에 맞게 조절
            ),
            "validation": enuma_voice_datasetdict["validation"],  # 그대로 사용
            "test": enuma_voice_datasetdict["test"],
        })
        # 2) 디스크에 저장 (폴더 경로는 원하는대로)
        dataset_dict_aug.save_to_disk(AUGUMENT_DATA_PATH)
    else:
        dataset_dict_aug = load_from_disk(AUGUMENT_DATA_PATH)
        logger.info("laod_from_dataset")

    # 3단계: tokenizer, processor
    # 3) 모델/프로세서
    success, tokenizer = get_tokenizer(model_name, lang_code, task)
    if not success:
        logger.error(f"Tokenizer 로드 실패: {tokenizer}")
    success, processor = get_processor(model_name, lang_code, task)
    if not success:
        logger.error(f"Processor 로드 실패: {processor}")
    logger.info(f"tokenizer, processor 로드 성공")

    enuma_voice_proc = preprocess_datasetdict_batched(dataset_dict_aug,
                                                    processor=processor,
                                                    audio_col="audio",
                                                    text_col="sentence",
                                                    num_proc=4,            # I/O 병목이면 2~4 권장
                                                    batch_size=32,
                                                    remove_others=False,   # 디버깅 위해 원본 칼럼 유지하려면 False
                                                    feature_dtype="float32",
                                                    max_target_len=None,   # 필요 시 448 같은 값으로 제한
                                                    truncation=True
                                                    )
    

    
    # 6) 모델/콜레이터/아규먼트/트레이너
    model = get_model(model_name)

    # 추론/평가시 유용
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    #model.generation_config.forced_decoder_ids = None
    model.generation_config.forced_decoder_idx = processor.get_decoder_prompt_ids(language='english',task='transcribe')
    
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    #model.config.pad_token_id = tok.pad_token_id
    #model.config.decoder_start_token_id = tok.pad_token_id  # Wh    isper는 보통 eos==pad
    sot_id = tok.convert_tokens_to_ids("<|startoftranscript|>")
    model.config.decoder_start_token_id = sot_id

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    logger.info('success')
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/whisper-tiny-en",
        
        # ✅ 배치/에폭        
        num_train_epochs=4,                   # 3~5 권장; early stopping과 함께 사용 권장
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,        # 유효 배치 16
        max_grad_norm=1.0,
        
        # ✅ 옵티마이저/스케줄
        learning_rate=1e-4,                   # 과적합 시 5e-5로
        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing_factor=0.1,                
        
        # ✅ 평가/저장/로그: 에폭 단위로 할 것인가? 스텝단위   
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=50,
        report_to=["tensorboard"],
        
        # ✅ 베스트 모델 로딩        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        

        # ✅ 생성 평가
        predict_with_generate=True,
        # (권장) 모델의 generation_config로 관리:
        # model.generation_config.max_length = 128
        # model.generation_config.num_beams = 4
        
        
        # ✅ DataLoader        
        dataloader_num_workers=0,             # 환경 따라 0/2/4 테스트
        dataloader_prefetch_factor=None,
        dataloader_persistent_workers=False,
        
        group_by_length=False, # True --> False
        length_column_name="input_length",
        remove_unused_columns=False,

        # ✅ 정밀도 / 로깅
        fp16=True,                            # CUDA일 때 유효; MPS면 False 권장        
        seed=42,
    )   
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=enuma_voice_proc["train"],
        eval_dataset=enuma_voice_proc["test"],
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
        processing_class=processor,  # tokenizer 대신 processor
    )    

    logger.info("start training")
    trainer.train()
    

