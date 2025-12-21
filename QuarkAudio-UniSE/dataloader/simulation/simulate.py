import numpy as np
from copy import deepcopy
import librosa
import random

from .rir_utils import estimate_early_rir, add_reverberation
from .detect_non_silence import detect_non_silence


def mix_noise(speech, noise, snr=5.0):
    len_speech = speech.shape[-1]
    len_noise = noise.shape[-1]
    if len_noise < len_speech:
        offset = np.random.randint(0, len_speech - len_noise)
        # Repeat noise
        noise = np.pad(
            noise,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = np.random.randint(0, len_noise - len_speech)
        noise = noise[:, offset : offset + len_speech]
    
    rms_noise = noise[detect_non_silence(noise)].std()
    rms_speech = speech[detect_non_silence(speech)].std()

    scale_noise = 10 ** (-snr / 20) * rms_speech / (rms_noise + 1e-10)
    noisy = noise * scale_noise + speech
    return noisy


def bandwidth_limitation(speech_sample, fs: int, fs_new: list, res_type="soxr_hq"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        fs (int): sampling rate in Hz
        fs_new (int): effective sampling rate in Hz
        res_type (str): resampling method

    Returns:
        ret (np.ndarray): bandwidth-limited speech sample (1, Time)
    """
    opts = {"res_type": res_type}
    if fs == fs_new:
        return speech_sample
    assert fs > fs_new, (fs, fs_new)
    ret = librosa.resample(speech_sample, orig_sr=fs, target_sr=fs_new, **opts)
    # resample back to the original sampling rate
    ret = librosa.resample(ret, orig_sr=fs_new, target_sr=fs, **opts)
    return ret[:, : speech_sample.shape[1]]


def clipping(speech_sample, min_quantile = 0.1, max_quantile = 0.9):
    """Apply the clipping distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        min_quantile (float): lower bound on the quantile of samples to be clipped
        max_quantile (float): upper bound on the quantile of samples to be clipped

    Returns:
        ret (np.ndarray): clipped speech sample (1, Time)
    """
    q = np.array([min_quantile, max_quantile])
    min_, max_ = np.quantile(speech_sample, q, axis=-1, keepdims=False)
    # per-channel clipping
    ret = np.stack(
        [
            np.clip(speech_sample[i], min_[i], max_[i])
            for i in range(speech_sample.shape[0])
        ],
        axis=0,
    )
    return ret



def get_packet_loss_indices(
    speech_length, fs, packet_duration_ms, packet_loss_rate, max_continuous_packet_loss
):
    """Returns a list of indices (of packets) that are zeroed out."""

    # speech duration in ms and the number of packets
    speech_duration_ms = speech_length / fs * 1000
    num_packets = int(speech_duration_ms // packet_duration_ms)

    # calculate the packet loss duration
    packet_loss_duration_ms = packet_loss_rate * speech_duration_ms

    # calculate the number of packets to be zeroed out
    num_packet_loss = int(round(packet_loss_duration_ms / packet_duration_ms, 0))

    # list of length of each packet loss
    packet_loss_lengths = []
    for _ in range(num_packet_loss):
        num_continuous_packet_loss = np.random.randint(1, max_continuous_packet_loss)
        packet_loss_lengths.append(num_continuous_packet_loss)

        if num_packet_loss - sum(packet_loss_lengths) <= max_continuous_packet_loss:
            packet_loss_lengths.append(num_packet_loss - sum(packet_loss_lengths))
            break

    packet_loss_start_indices = np.random.choice(
        range(num_packets), len(packet_loss_lengths), replace=False
    )
    packet_loss_indices = []
    for idx, length in zip(packet_loss_start_indices, packet_loss_lengths):
        packet_loss_indices += list(range(idx, idx + length))

    return list(set(packet_loss_indices))


def packet_loss(
    speech_sample, fs: int, packet_loss_indices: list, packet_duration_ms: int = 20
):
    for idx in packet_loss_indices:
        start = idx * packet_duration_ms * fs // 1000
        end = (idx + 1) * packet_duration_ms * fs // 1000
        speech_sample[:, start:end] = 0

    return speech_sample


def simulate_data(mode, speech, interf, noise, rir, fs, config):
    # for interference
    if mode == 'tse' or mode == 'rtse':  # 启用TSE/rTSE模式
        sir = random.uniform(*config['tse_interference']['sir'])
    else:  # SE模式
        sir = random.uniform(*config['se_interference']['sir'])
    # for additive noise
    snr = random.uniform(*config['noise']['snr'])
    # for bandwidth limitation
    fs_new = random.choice(config['bandwidth_limitation']['fs_new'])
    res_type = config['bandwidth_limitation']['res_type']
    # for clipping
    min_quantile = random.uniform(*config['clipping']['min_quantile'])
    max_quantile = random.uniform(*config['clipping']['max_quantile'])
    # for packet loss
    packet_duration_ms = config['packet_loss']['packet_duration_ms']
    packet_loss_rate = random.uniform(*config['packet_loss']['packet_loss_rate'])
    max_continuous_packet_loss = config['packet_loss']['max_continuous_packet_loss']

    if interf is not None:
        noisy = mix_noise(speech, interf, snr=sir)
        interf = noisy - speech
    else:
        noisy = deepcopy(speech)

    if random.random() < config['reverberation']['prob'] and rir is not None:
        # print(np.max(rir))
        rir = rir / (np.max(np.abs(rir)) + 1e-5)
        noisy = add_reverberation(noisy, rir)
        early_rir = estimate_early_rir(rir, fs=fs)
        speech = add_reverberation(speech, early_rir)
        if interf is not None:
            interf = add_reverberation(interf, early_rir)
    
    if random.random() < config['noise']['prob']:
        noisy = mix_noise(noisy, noise, snr=snr)  # 以混响语音计算能量，不改变noisy-clean相对幅度

    order_list = [0, 1, 2]
    random.shuffle(order_list)

    for order in order_list:
        if order == 0 and random.random() < config['bandwidth_limitation']['prob']:
            noisy = bandwidth_limitation(noisy, fs, fs_new=fs_new, res_type=res_type)
        elif order == 1 and random.random() < config['clipping']['prob']:
            noisy = clipping(noisy, min_quantile=min_quantile, max_quantile=max_quantile)
        elif order == 2 and random.random() < config['packet_loss']['prob']:
            packet_loss_indices = get_packet_loss_indices(
                speech.shape[-1],
                fs,
                packet_duration_ms,
                packet_loss_rate,
                max_continuous_packet_loss,
            )
            noisy = packet_loss(noisy, fs, packet_loss_indices, packet_duration_ms)
    
    # 调整幅度防止削波
    max_val = max(np.max(np.abs(noisy)), np.max(np.abs(speech)))
    if interf is not None:
        max_val = max(max_val, np.max(np.abs(interf)))
    
    if max_val > 0.99:
        noisy = noisy / max_val * 0.99
        speech = speech / max_val * 0.99
        if interf is not None:
            interf = interf / max_val * 0.99
    
    return noisy, speech, interf


if __name__ == '__main__':
    import yaml
    import librosa
    import soundfile as sf
    
    def load_wav(path, fs=None):
        wav, fs_ = librosa.load(path, dtype=np.float32, sr=fs, mono=False)
        if wav.ndim == 1:
            wav = wav[None]  # (1, T)
        else:
            wav = wav[:1, :]  # 取第0通道
        return wav, fs_

    with open('/mnt/nas1/project/unified_llm_speech/bicodec_ar_sft_se/conf/simulation_train.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    speech, fs = load_wav('/mnt/nas1/002_0313018212@0.wav', fs=16000)
    noise, _ = load_wav('/mnt/nas1/dataset/fsd50k/FSD50K.dev_audio/0/237.wav', fs=16000)
    rir, _ = load_wav('/mnt/nas1/dataset/rir/RIRS_NOISES/simulated_rirs/largeroom/Room001/Room001-00009.wav', fs=16000)
    
    noisy, speech = simulate_data(speech, noise, rir, fs, config)
    sf.write('noisy.wav', noisy[0], fs)
    sf.write('speech.wav', speech[0], fs)
