"""
augumented_audio.py - 오디오 데이터 증강 모듈

음성 인식 모델의 일반화 성능 향상을 위해 학습 데이터에 다양한 변형을 가한다.
지원하는 증강 기법:
  1. Speed Perturbation: 재생 속도를 랜덤하게 변경 (0.9x ~ 1.1x)
  2. Volume Perturbation: 볼륨을 랜덤하게 변경 (-6dB ~ +6dB)
  3. Noise Injection: SNR 기반으로 배경 노이즈를 혼합 (5~20dB)
  4. Reverb (RIR Convolution): Room Impulse Response로 잔향/울림 효과 시뮬레이션
  5. SpecAugment: log-mel 스펙트로그램에 주파수/시간 마스킹 적용

각 증강은 독립적으로 확률 기반으로 적용되며, 여러 기법이 동시에 적용될 수 있다.
config.yaml의 augmentation 섹션에서 개별 활성화/비활성화 및 파라미터 조정 가능.

기법별 처리 도메인:
  - Speed / Volume / Noise / Reverb : 파형(waveform) 도메인 — log-mel 변환 이전
  - SpecAugment                      : 스펙트로그램 도메인  — log-mel 변환 이후
"""

import random
import torch
import torchaudio
import glob

# 노이즈 wav 파일을 메모리에 미리 로드하여 재사용하는 글로벌 리스트
noise_bank = []


def load_noise():
    """
    노이즈 wav 파일들을 로드하여 noise_bank에 저장한다.
    모든 노이즈는 16kHz로 리샘플링되어 일관된 샘플레이트를 보장한다.

    사용 전 NOISE_DIR 경로를 실제 노이즈 파일이 있는 디렉토리로 변경해야 한다.
    """
    NOISE_DIR = "/path/to/noise_wavs"   # 사용자 환경에 맞게 노이즈 디렉토리 경로 수정 필요
    noise_paths = glob.glob(f"{NOISE_DIR}/*.wav")

    noise_bank = []
    for p in noise_paths:
        noise_waveform, noise_sr = torchaudio.load(p)  # (channels, T) 형태로 로드
        # 샘플레이트가 16kHz가 아니면 리샘플링
        if noise_sr != 16000:
            noise_waveform = torchaudio.functional.resample(noise_waveform, noise_sr, 16000)
        noise_bank.append(noise_waveform)


'''
# sox_effects 기반 속도 변환 (sox 백엔드 필요 - macOS/Linux에서 사용 가능)
def speed_perturb(waveform: torch.Tensor,
                  sample_rate: int,
                  min_speed: float = 0.9,
                  max_speed: float = 1.1):
    speed = random.uniform(min_speed, max_speed)
    effects = [
        ["speed", f"{speed}"],
        ["rate", f"{sample_rate}"],
    ]
    augmented, _ = sox_effects.apply_effects_tensor(
        waveform, sample_rate, effects
    )
    return augmented, sample_rate
'''


def speed_perturb_simple(waveform, sample_rate, min_speed=0.9, max_speed=1.1):
    """
    리샘플링 기반의 간단한 속도 변환.

    원리: 원본 SR → (SR * speed)로 리샘플링 후 다시 원본 SR로 복원하면,
    waveform의 시간축이 speed 비율만큼 변한다.
    sox 백엔드 없이 동작하므로 환경 의존성이 적다.

    Args:
        waveform: (1, T) 형태의 오디오 텐서
        sample_rate: 원본 샘플레이트
        min_speed: 최소 속도 배율 (0.9 = 10% 느리게)
        max_speed: 최대 속도 배율 (1.1 = 10% 빠르게)

    Returns:
        속도가 변환된 waveform 텐서
    """
    speed = random.uniform(min_speed, max_speed)
    new_length = int(waveform.shape[-1] / speed)
    # 2단계 리샘플링: 원본SR → 변환SR → 원본SR (시간축 스케일 효과)
    augmented = torchaudio.functional.resample(waveform, sample_rate, int(sample_rate * speed))
    augmented = torchaudio.functional.resample(augmented, int(sample_rate * speed), sample_rate)
    return augmented


import math

def volume_perturb(waveform: torch.Tensor,
                   min_gain_db: float = -6.0,
                   max_gain_db: float = 6.0):
    """
    랜덤 볼륨 변환 (dB 단위).

    dB → linear 변환 공식: gain_linear = 10^(gain_dB / 20)
    예: +6dB ≈ 2배, -6dB ≈ 0.5배

    Args:
        waveform: (1, T) 형태의 오디오 텐서
        min_gain_db: 최소 게인 (dB) — 음수면 볼륨 감소
        max_gain_db: 최대 게인 (dB) — 양수면 볼륨 증가

    Returns:
        볼륨이 조정된 waveform 텐서
    """
    gain_db = random.uniform(min_gain_db, max_gain_db)
    gain = 10 ** (gain_db / 20)  # dB → linear 스케일로 변환
    return waveform * gain


def add_noise_with_snr(clean: torch.Tensor,
                       noise_bank,
                       sample_rate: int = 16000,
                       min_snr_db: float = 5.0,
                       max_snr_db: float = 20.0):
    """
    SNR(Signal-to-Noise Ratio) 기반 노이즈 혼합.

    지정된 SNR 범위 내에서 랜덤 값을 선택하고, 해당 SNR을 달성하도록
    노이즈의 스케일을 조정하여 원본 오디오에 더한다.

    SNR 스케일 계산:
      clean_power / (scale^2 * noise_power) = snr
      → scale = sqrt(clean_power / (noise_power * snr))

    Args:
        clean: (1, T) 형태의 깨끗한 오디오 텐서
        noise_bank: [(1, T_noise), ...] 노이즈 텐서 리스트
        sample_rate: 샘플레이트 (현재 미사용, 향후 확장용)
        min_snr_db: 최소 SNR (dB) — 낮을수록 노이즈가 강함
        max_snr_db: 최대 SNR (dB) — 높을수록 원본에 가까움

    Returns:
        노이즈가 혼합된 waveform 텐서 ([-1, 1] 범위로 클리핑)
    """
    if len(noise_bank) == 0:
        return clean

    # 노이즈 뱅크에서 랜덤으로 하나 선택
    noise = random.choice(noise_bank)  # (1, Tn)

    # 노이즈와 원본 오디오의 길이를 맞춤
    T = clean.shape[1]    # 원본 오디오 길이
    Tn = noise.shape[1]   # 노이즈 길이

    if Tn < T:
        # 노이즈가 짧으면 반복(tile)하여 길이 맞춤
        repeat_factor = (T // Tn) + 1
        noise = noise.repeat(1, repeat_factor)[:, :T]
    elif Tn > T:
        # 노이즈가 길면 랜덤 위치에서 잘라냄(crop)
        start = random.randint(0, Tn - T)
        noise = noise[:, start:start+T]

    # 원본과 노이즈의 평균 파워 계산
    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()

    # 무음 노이즈는 혼합 불가 → 원본 그대로 반환
    if noise_power == 0:
        return clean

    # 목표 SNR 달성을 위한 노이즈 스케일 팩터 계산
    snr_db = random.uniform(min_snr_db, max_snr_db)
    snr = 10 ** (snr_db / 10)  # dB → linear
    scale = torch.sqrt(clean_power / (noise_power * snr))

    # 원본 + 스케일링된 노이즈 혼합
    noisy = clean + scale * noise

    # [-1, 1] 범위로 클리핑하여 오디오 오버플로우 방지
    noisy = torch.clamp(noisy, -1.0, 1.0)
    return noisy


def add_reverb(waveform: torch.Tensor,
               rir_bank: list,
               sample_rate: int = 16000):
    """
    RIR(Room Impulse Response) 컨볼루션으로 잔향/울림 효과를 시뮬레이션한다.

    홀, 교실, 회의실 등 실제 공간의 음향 특성을 재현한다.
    RIR은 특정 공간에서 임펄스(충격음)를 재생하고 마이크로 녹음한 응답 신호로,
    이를 깨끗한 음성과 컨볼루션하면 해당 공간에서 녹음한 것 같은 효과가 난다.

    처리 과정:
      1. RIR 뱅크에서 랜덤으로 하나 선택
      2. RIR을 L2 정규화 (에너지 보존)
      3. FFT 기반 컨볼루션 수행 (긴 RIR에서도 효율적)
      4. 원본과 같은 길이로 잘라내고 진폭 정규화

    Args:
        waveform: (1, T) 형태의 깨끗한 오디오 텐서
        rir_bank: [(1, R), ...] RIR 텐서 리스트
        sample_rate: 샘플레이트 (향후 RIR 리샘플링 확장용)

    Returns:
        잔향이 적용된 waveform 텐서
    """
    if len(rir_bank) == 0:
        return waveform

    # RIR 뱅크에서 랜덤 선택
    rir = random.choice(rir_bank)  # (1, R)

    # 모노가 아닌 경우 첫 번째 채널만 사용
    if rir.shape[0] > 1:
        rir = rir[:1]

    # L2 정규화: RIR의 에너지를 1로 맞춰 원본 음량을 유지
    rir_norm = rir / rir.norm(p=2).clamp(min=1e-8)

    # FFT 기반 컨볼루션 (시간 도메인 conv보다 긴 신호에서 훨씬 빠름)
    reverbed = torchaudio.functional.fftconvolve(waveform, rir_norm)

    # 컨볼루션 결과는 원본보다 길어지므로 원본 길이로 잘라냄
    reverbed = reverbed[:, :waveform.shape[1]]

    # 진폭 정규화: 최대 절대값이 1을 넘지 않도록 스케일링
    peak = reverbed.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        reverbed = reverbed / peak

    return reverbed


def load_rir_from_dir(rir_dir: str) -> list:
    """
    지정 디렉토리에서 RIR(Room Impulse Response) wav 파일들을 로드한다.

    모든 RIR은 16kHz로 리샘플링되어 음성 데이터와 동일한 샘플레이트를 보장한다.

    공개 RIR 데이터셋:
      - MIT IR Survey: 다양한 실내/홀 환경 (271개)
      - OpenSLR-26 (RWCP): 소/중/대규모 방
      - OpenSLR-28 (simulated): 합성 RIR (60,000개)
      - BUT ReverbDB: 실측 RIR

    Args:
        rir_dir: RIR wav 파일이 있는 디렉토리 경로

    Returns:
        [(1, R), ...] 형태의 RIR 텐서 리스트
    """
    import os
    rir_paths = glob.glob(os.path.join(rir_dir, "*.wav"))
    bank = []
    for p in rir_paths:
        waveform, sr = torchaudio.load(p)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = waveform[:1]
        bank.append(waveform)
    return bank


def augment_waveform(waveform: torch.Tensor,
                     sample_rate: int,
                     noise_bank,
                     p_speed: float = 0.5,
                     p_volume: float = 0.5,
                     p_noise: float = 0.5,
                     rir_bank: list = None,
                     p_reverb: float = 0.0):
    """
    증강 기법들을 확률적으로 적용하는 통합 함수.

    각 증강은 독립적인 확률(p_*)로 적용되므로,
    0개~4개의 증강이 동시에 적용될 수 있다.

    적용 순서 (음향적으로 자연스러운 순서):
      1. Speed → 2. Volume → 3. Reverb → 4. Noise
      (잔향은 공간 효과이므로 노이즈 혼합 전에 적용해야 자연스럽다)

    Args:
        waveform: (1, T) 형태의 오디오 텐서
        sample_rate: 샘플레이트
        noise_bank: 노이즈 텐서 리스트
        p_speed: 속도 변환 적용 확률
        p_volume: 볼륨 변환 적용 확률
        p_noise: 노이즈 혼합 적용 확률
        rir_bank: RIR 텐서 리스트 (None이면 reverb 스킵)
        p_reverb: 잔향 적용 확률

    Returns:
        (augmented_waveform, sample_rate) 튜플
    """
    # 1. 속도 변환
    if random.random() < p_speed:
        waveform = speed_perturb_simple(waveform, sample_rate)

    # 2. 볼륨 변환
    if random.random() < p_volume:
        waveform = volume_perturb(waveform)

    # 3. 잔향(RIR 컨볼루션) — 노이즈보다 먼저 적용
    if rir_bank and random.random() < p_reverb:
        waveform = add_reverb(waveform, rir_bank, sample_rate)

    # 4. 노이즈 혼합 — 잔향 후 적용 (현실적인 녹음 환경 시뮬레이션)
    if random.random() < p_noise:
        waveform = add_noise_with_snr(waveform, noise_bank, sample_rate)

    return waveform, sample_rate


from datasets import DatasetDict


def make_augment_config(aug_cfg: dict) -> dict:
    """
    config.yaml의 augmentation 섹션으로부터 증강 파라미터를 추출한다.

    Args:
        aug_cfg: config.yaml의 augmentation 딕셔너리

    Returns:
        증강 함수들에 전달 가능한 파라미터 딕셔너리
    """
    # reverb 섹션이 없을 수 있으므로 안전하게 처리
    reverb_cfg = aug_cfg.get("reverb", {})
    return {
        "p_speed": aug_cfg["speed"]["prob"] if aug_cfg["speed"]["enabled"] else 0.0,
        "p_volume": aug_cfg["volume"]["prob"] if aug_cfg["volume"]["enabled"] else 0.0,
        "p_noise": aug_cfg["noise"]["prob"] if aug_cfg["noise"]["enabled"] else 0.0,
        "p_reverb": reverb_cfg.get("prob", 0.0) if reverb_cfg.get("enabled", False) else 0.0,
        "min_speed": aug_cfg["speed"].get("min_speed", 0.9),
        "max_speed": aug_cfg["speed"].get("max_speed", 1.1),
        "min_gain_db": aug_cfg["volume"].get("min_gain_db", -6.0),
        "max_gain_db": aug_cfg["volume"].get("max_gain_db", 6.0),
        "min_snr_db": aug_cfg["noise"].get("min_snr_db", 5.0),
        "max_snr_db": aug_cfg["noise"].get("max_snr_db", 20.0),
    }


def load_noise_from_dir(noise_dir: str) -> list:
    """
    지정 디렉토리에서 노이즈 파일을 로드하여 리스트로 반환한다.

    Args:
        noise_dir: 노이즈 wav 파일이 있는 디렉토리 경로

    Returns:
        [(1, T), ...] 형태의 노이즈 텐서 리스트
    """
    import os
    noise_paths = glob.glob(os.path.join(noise_dir, "*.wav"))
    bank = []
    for p in noise_paths:
        waveform, sr = torchaudio.load(p)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        bank.append(waveform)
    return bank


def make_on_the_fly_transform(processor, aug_cfg: dict, loaded_noise_bank: list,
                              loaded_rir_bank: list = None):
    """
    on-the-fly 증강용 transform 함수를 생성한다.

    HuggingFace Dataset의 set_transform()에 전달하여,
    매 샘플 접근 시 증강 → log-mel 추출 → 토크나이징을 실시간으로 수행한다.
    에폭마다 다른 랜덤 증강이 적용되어 데이터 다양성이 극대화된다.

    증강 적용 순서 (음향적으로 자연스러운 순서):
      Speed → Volume → Reverb → Noise

    Args:
        processor: WhisperProcessor (log-mel 추출 + 토크나이징)
        aug_cfg: config.yaml의 augmentation 딕셔너리
        loaded_noise_bank: 사전 로드된 노이즈 텐서 리스트
        loaded_rir_bank: 사전 로드된 RIR 텐서 리스트 (None이면 reverb 스킵)

    Returns:
        set_transform()에 전달할 transform 함수
    """
    import numpy as np

    params = make_augment_config(aug_cfg)
    rir_bank = loaded_rir_bank or []

    def transform(batch):
        result = {"input_features": [], "labels": []}

        audios = batch["audio"]
        texts = batch["sentence"]
        # set_transform은 단일 샘플도 리스트로 감싸서 전달
        if not isinstance(audios, list):
            audios = [audios]
            texts = [texts]

        for audio, text in zip(audios, texts):
            sr = audio["sampling_rate"]
            wav = torch.tensor(audio["array"]).unsqueeze(0)  # (1, T)

            # 증강 적용 순서: Speed → Volume → Reverb → Noise
            # (config에서 비활성화한 기법은 prob=0.0으로 자동 스킵)
            wav_aug = wav
            if random.random() < params["p_speed"]:
                wav_aug = speed_perturb_simple(wav_aug, sr, params["min_speed"], params["max_speed"])
            if random.random() < params["p_volume"]:
                wav_aug = volume_perturb(wav_aug, params["min_gain_db"], params["max_gain_db"])
            if rir_bank and random.random() < params["p_reverb"]:
                wav_aug = add_reverb(wav_aug, rir_bank, sr)
            if random.random() < params["p_noise"]:
                wav_aug = add_noise_with_snr(wav_aug, loaded_noise_bank, sr, params["min_snr_db"], params["max_snr_db"])

            # log-mel spectrogram 추출
            feats = processor(
                wav_aug.squeeze(0).numpy(),
                sampling_rate=sr,
                return_tensors="np",
            ).input_features[0]

            # 텍스트 토크나이징
            labels = processor.tokenizer(text).input_ids

            result["input_features"].append(feats.astype(np.float32))
            result["labels"].append(labels)

        return result

    return transform


def make_eval_transform(processor):
    """
    평가/테스트용 transform 함수를 생성한다 (증강 없음).

    on-the-fly 모드에서 validation/test 데이터에 적용.
    증강 없이 log-mel 추출 + 토크나이징만 수행한다.

    Args:
        processor: WhisperProcessor

    Returns:
        set_transform()에 전달할 transform 함수
    """
    import numpy as np

    def transform(batch):
        result = {"input_features": [], "labels": []}

        audios = batch["audio"]
        texts = batch["sentence"]
        if not isinstance(audios, list):
            audios = [audios]
            texts = [texts]

        for audio, text in zip(audios, texts):
            sr = audio["sampling_rate"]
            feats = processor(
                audio["array"],
                sampling_rate=sr,
                return_tensors="np",
            ).input_features[0]
            labels = processor.tokenizer(text).input_ids

            result["input_features"].append(feats.astype(np.float32))
            result["labels"].append(labels)

        return result

    return transform


def apply_spec_augment(
    features: torch.Tensor,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """
    SpecAugment: log-mel 스펙트로그램에 주파수 마스킹과 시간 마스킹을 적용한다.

    원리:
      - Frequency Masking: mel 주파수 축(0~79)의 연속 f개 bin을 0으로 마스킹
      - Time Masking: 시간 축의 연속 t개 프레임을 0으로 마스킹
      마스킹 값(0)은 log-mel 정규화 후 무음에 해당하는 값이다.

    적용 위치:
      DataCollatorSpeechSeq2SeqWithPadding.__call__() 내부에서 호출되며,
      학습 시에만 적용되고 평가 시에는 비활성화된다.

    Args:
        features : (batch, freq, time) 형태의 log-mel 텐서
                   batch=배치 크기, freq=80(mel bins), time=최대 3000 프레임
        freq_mask_param : 주파수 마스킹 최대 폭 (mel bins, 최대 80)
                          권장값: 27 (전체 80bins의 약 1/3)
        time_mask_param : 시간 마스킹 최대 폭 (프레임 수)
                          권장값: 100 (30초 기준 3000프레임의 약 3%)
        num_freq_masks  : 주파수 마스킹 반복 횟수 (독립적으로 적용)
        num_time_masks  : 시간 마스킹 반복 횟수 (독립적으로 적용)

    Returns:
        마스킹이 적용된 features 텐서 (shape 동일)
    """
    features = features.clone()
    batch_size, freq_size, time_size = features.shape

    for i in range(batch_size):
        # --- 주파수 마스킹 (Frequency Masking) ---
        # mel 주파수 축에서 f개 연속 bin을 무음(0)으로 마스킹
        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_param)
            if f == 0 or freq_size <= f:
                continue
            f0 = random.randint(0, freq_size - f)
            features[i, f0:f0 + f, :] = 0.0

        # --- 시간 마스킹 (Time Masking) ---
        # 시간 축에서 t개 연속 프레임을 무음(0)으로 마스킹
        for _ in range(num_time_masks):
            t = random.randint(0, min(time_mask_param, time_size))
            if t == 0 or time_size <= t:
                continue
            t0 = random.randint(0, time_size - t)
            features[i, :, t0:t0 + t] = 0.0

    return features


def augment_example(example):
    """
    HuggingFace Dataset.map()에 전달되는 단일 샘플 증강 함수.

    Dataset의 각 example에 대해:
      1. audio 딕셔너리에서 numpy array → torch tensor 변환
      2. augment_waveform()으로 증강 적용
      3. 결과를 다시 numpy array로 변환하여 example에 저장

    주의: HF Audio feature의 내부 필드를 직접 수정하면 Arrow 캐시와 충돌할 수 있으므로,
    audio 딕셔너리 전체를 새 객체로 교체한다.

    Args:
        example: HuggingFace Dataset의 단일 샘플 딕셔너리
                 {"audio": {"array": np.ndarray, "sampling_rate": int}, "sentence": str, ...}

    Returns:
        증강이 적용된 example 딕셔너리
    """
    load_noise()

    import numpy as np

    # HF Audio feature에서 오디오 데이터 추출
    audio = example["audio"]
    sr = audio["sampling_rate"]
    wav = torch.tensor(audio["array"]).unsqueeze(0)  # (T,) → (1, T) 형태로 차원 추가

    # 오디오 증강 적용 (speed + volume + noise, 각각 50% 확률)
    wav_aug, sr_aug = augment_waveform(wav, sr, noise_bank)

    # 증강된 오디오를 numpy로 변환하여 example에 저장
    # audio 전체를 새 딕셔너리로 교체 (Arrow 캐시 호환성 보장)
    example["audio"] = {
        "array": wav_aug.squeeze(0).numpy(),   # (1, T) → (T,) 형태로 복원
        "sampling_rate": sr_aug,
    }
    return example
