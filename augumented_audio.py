import random
import torch
import torchaudio

# 예시: noise wav 파일들을 미리 로딩해서 리스트로 가지고 있기
import glob

noise_bank = []

def load_noise():
    NOISE_DIR = "/path/to/noise_wavs"   # 자기 노이즈 경로로 변경
    noise_paths = glob.glob(f"{NOISE_DIR}/*.wav")

    noise_bank = []
    for p in noise_paths:
        noise_waveform, noise_sr = torchaudio.load(p)  # (channels, T)
        # 필요하면 resample
        if noise_sr != 16000:
            noise_waveform = torchaudio.functional.resample(noise_waveform, noise_sr, 16000)
        noise_bank.append(noise_waveform)  # 모두 16kHz로 맞추기

'''
def speed_perturb(waveform: torch.Tensor,
                  sample_rate: int,
                  min_speed: float = 0.9,
                  max_speed: float = 1.1):
    
    """
    waveform: (1, T) 형태의 tensor
    sample_rate: int
    return: (waveform_aug, sample_rate)
    """
    speed = random.uniform(min_speed, max_speed)
    effects = [
        ["speed", f"{speed}"],   # 속도 변경
        ["rate", f"{sample_rate}"],  # 다시 원래 sample_rate로
    ]
    augmented, _ = sox_effects.apply_effects_tensor(
        waveform, sample_rate, effects
    )
    return augmented, sample_rate
'''

def speed_perturb_simple(waveform, sample_rate, min_speed=0.9, max_speed=1.1):
    speed = random.uniform(min_speed, max_speed)
    new_length = int(waveform.shape[-1] / speed)
    augmented = torchaudio.functional.resample(waveform, sample_rate, int(sample_rate * speed))
    augmented = torchaudio.functional.resample(augmented, int(sample_rate * speed), sample_rate)
    return augmented

import math
def volume_perturb(waveform: torch.Tensor,
                   min_gain_db: float = -6.0,
                   max_gain_db: float = 6.0):
    """
    waveform: (1, T)
    gain_db 범위에서 랜덤으로 골라 곱해줌
    """
    gain_db = random.uniform(min_gain_db, max_gain_db)
    gain = 10 ** (gain_db / 20)  # dB → linear
    return waveform * gain


def add_noise_with_snr(clean: torch.Tensor,
                       noise_bank,
                       sample_rate: int = 16000,
                       min_snr_db: float = 5.0,
                       max_snr_db: float = 20.0):
    """
    clean: (1, T)  waveform
    noise_bank: [ (1, T_noise), ... ] 리스트
    SNR(dB) 범위 내에서 랜덤 선택 후 noise를 섞는다.
    """
    if len(noise_bank) == 0:
        return clean

    # 1. 랜덤 noise 선택
    noise = random.choice(noise_bank)  # (1, Tn)
    # 길이 맞추기 (noise 길이가 짧으면 반복, 길면 랜덤 위치 crop)
    T = clean.shape[1]
    Tn = noise.shape[1]

    if Tn < T:
        # noise를 반복시켜서 길이 맞추기
        repeat_factor = (T // Tn) + 1
        noise = noise.repeat(1, repeat_factor)[:, :T]
    elif Tn > T:
        # noise에서 랜덤 crop
        start = random.randint(0, Tn - T)
        noise = noise[:, start:start+T]
    else:
        # 길이 같으면 그대로 사용
        pass

    # 2. SNR 기반 스케일링
    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()

    if noise_power == 0:
        return clean

    snr_db = random.uniform(min_snr_db, max_snr_db)
    snr = 10 ** (snr_db / 10)

    # clean_power / (scale^2 * noise_power) = snr
    # => scale = sqrt(clean_power / (noise_power * snr))
    scale = torch.sqrt(clean_power / (noise_power * snr))

    noisy = clean + scale * noise

    # clipping 방지용 (optional)
    noisy = torch.clamp(noisy, -1.0, 1.0)
    return noisy

def augment_waveform(waveform: torch.Tensor,
                     sample_rate: int,
                     noise_bank,
                     p_speed: float = 0.5,
                     p_volume: float = 0.5,
                     p_noise: float = 0.5):
    """
    waveform: (1, T)
    """
    # speed
    if random.random() < p_speed:
        waveform = speed_perturb_simple(waveform, sample_rate)

    # volume
    if random.random() < p_volume:
        waveform = volume_perturb(waveform)

    # noise
    if random.random() < p_noise:
        waveform = add_noise_with_snr(waveform, noise_bank, sample_rate)

    return waveform, sample_rate
from datasets import DatasetDict

def augment_example(example):
    load_noise()
    # HF Audio feature에서 waveform 불러오기
    import numpy as np
    audio = example["audio"]
    sr = audio["sampling_rate"]
    wav = torch.tensor(audio["array"]).unsqueeze(0)  # (1, T)

    # augmentation
    wav_aug, sr_aug = augment_waveform(wav, sr, noise_bank)

    # 다시 numpy로 되돌려서 example에 넣기
    #example["audio"]["array"] = wav_aug.squeeze(0).numpy()
    #example["audio"]["sampling_rate"] = sr_aug

# ✅ AudioDecoder 안쪽 필드를 건드리지 말고, audio 전체를 교체
    example["audio"] = {
        "array": wav_aug.squeeze(0).numpy(),
        "sampling_rate": sr_aug,
    }
    return example