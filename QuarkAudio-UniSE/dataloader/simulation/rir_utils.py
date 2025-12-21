import numpy as np
import scipy


def add_reverberation(speech_sample, rir_sample):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
    Returns:
        reverberant_sample (np.ndarray): output noisy sample (Channel, Time)
    """
    reverberant_sample = scipy.signal.convolve(speech_sample, rir_sample, mode="full")
    return reverberant_sample[:, : speech_sample.shape[1]]


# def estimate_early_rir(rir_sample, early_rir_sec: float = 0.08, fs: int = 48000):
#     """Estimate the part of RIR corresponding to the early reflections.

#     Args:
#         rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
#         early_rir_sec (float): the duration in seconds that we count as early RIR
#         fs (int): sampling frequency in Hz
#     Returns:
#         early_rir_sample (np.ndarray): estimated RIR (Channel, Time)
#     """
#     rir_start_sample = np.array([get_rir_start_sample(h) for h in rir_sample])
#     early_rir_samples = int(early_rir_sec * fs)
#     rir_stop_sample = rir_start_sample + early_rir_samples
#     rir_early = rir_sample.copy()
#     for i in range(rir_sample.shape[0]):
#         rir_early[i, rir_stop_sample[i]:] = 0
#     return rir_early


# ported from https://github.com/fgnt/sms_wsj/blob/master/sms_wsj/reverb/reverb_utils.py#L170
# def get_rir_start_sample(h, level_ratio=1e-1):
#     """Finds start sample in a room impulse response.

#     Selects that index as start sample where the first time
#     a value larger than `level_ratio * max_abs_value`
#     occurs.

#     If you intend to use this heuristic, test it on simulated and real RIR
#     first. This heuristic is developed on MIRD database RIRs and on some
#     simulated RIRs but may not be appropriate for your database.

#     If you want to use it to shorten impulse responses, keep the initial part
#     of the room impulse response intact and just set the tail to zero.

#     Params:
#         h: Room impulse response with Shape (num_samples,)
#         level_ratio: Ratio between start value and max value.

#     >>> get_rir_start_sample(np.array([0, 0, 1, 0.5, 0.1]))
#     2
#     """
#     assert level_ratio < 1, level_ratio
#     if h.ndim > 1:
#         assert h.shape[0] < 20, h.shape
#         h = np.reshape(h, (-1, h.shape[-1]))
#         return np.min(
#             [get_rir_start_sample(h_, level_ratio=level_ratio) for h_ in h]
#         )

#     abs_h = np.abs(h)
#     max_index = np.argmax(abs_h)
#     max_abs_value = abs_h[max_index]
#     # +1 because python excludes the last value
#     larger_than_threshold = abs_h[:max_index + 1] > level_ratio * max_abs_value

#     # Finds first occurrence of max
#     rir_start_sample = np.argmax(larger_than_threshold)
#     return rir_start_sample

'''
def estimate_early_rir(rir_sample, fs: int = 48000):
    """Estimate the part of RIR corresponding to the early reflections.

    Args:
        rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
        early_rir_sec (float): the duration in seconds that we count as early RIR
        fs (int): sampling frequency in Hz
    Returns:
        early_rir_sample (np.ndarray): estimated RIR (Channel, Time)
    """
    rir_start_sample = np.array([get_rir_start_sample(h) for h in rir_sample])
    rir_early = rir_sample.copy()
    for i in range(rir_sample.shape[0]):
        rir_early[i, :rir_start_sample[i]] = 0
        rir_early[i, rir_start_sample[i] + 1:] = 0
    return rir_early

def get_rir_start_sample(h):
    """Finds start sample in a room impulse response.

    Selects that index as start sample where the first time
    a value larger than `level_ratio * max_abs_value`
    occurs.

    If you intend to use this heuristic, test it on simulated and real RIR
    first. This heuristic is developed on MIRD database RIRs and on some
    simulated RIRs but may not be appropriate for your database.

    If you want to use it to shorten impulse responses, keep the initial part
    of the room impulse response intact and just set the tail to zero.

    Params:
        h: Room impulse response with Shape (num_samples,)
        level_ratio: Ratio between start value and max value.

    >>> get_rir_start_sample(np.array([0, 0, 1, 0.5, 0.1]))
    2
    """
    if h.ndim > 1:
        assert h.shape[0] < 20, h.shape
        h = np.reshape(h, (-1, h.shape[-1]))
        return np.min(
            [get_rir_start_sample(h_) for h_ in h]
        )

    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    return max_index
'''


def estimate_early_rir(rir_sample, fs: int = 48000):
    """Estimate the part of RIR corresponding to the early reflections.

    Args:
        rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
        early_rir_sec (float): the duration in seconds that we count as early RIR
        fs (int): sampling frequency in Hz
    Returns:
        early_rir_sample (np.ndarray): estimated RIR (Channel, Time)
    """
    rir_start_end_sample = np.array([get_rir_start_sample(h) for h in rir_sample])
    rir_early = np.zeros_like(rir_sample)
    for i in range(rir_sample.shape[0]):
        start = rir_start_end_sample[i][0]
        end = rir_start_end_sample[i][1]
        rir_early[i, start:end] = rir_sample[i, start:end]
    return rir_early


def get_rir_start_sample(h, level_ratio=1e-1):
    """Finds start sample in a room impulse response.

    Selects that index as start sample where the first time
    a value larger than `level_ratio * max_abs_value`
    occurs.

    If you intend to use this heuristic, test it on simulated and real RIR
    first. This heuristic is developed on MIRD database RIRs and on some
    simulated RIRs but may not be appropriate for your database.

    If you want to use it to shorten impulse responses, keep the initial part
    of the room impulse response intact and just set the tail to zero.

    Params:
        h: Room impulse response with Shape (num_samples,)
        level_ratio: Ratio between start value and max value.

    >>> get_rir_start_sample(np.array([0, 0, 1, 0.5, 0.1]))
    2
    """
    assert level_ratio < 1, level_ratio
    assert h.ndim == 1

    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    max_abs_value = abs_h[max_index]
    # +1 because python excludes the last value
    larger_than_threshold = abs_h[:max_index + 1] > level_ratio * max_abs_value
    smaller_than_threshold = abs_h[max_index + 1:] < level_ratio * max_abs_value

    # Finds first occurrence of max
    rir_start_sample = np.argmax(larger_than_threshold)
    rir_end_sample = np.argmax(smaller_than_threshold) + max_index + 1
    return rir_start_sample, rir_end_sample