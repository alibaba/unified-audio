import torch
import soundfile as sf
from typing import Union, List
from pathlib import Path
import numpy as np
import random
import pytorch_lightning as pl
import torch.utils
import torch.utils.data
from copy import deepcopy
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import librosa
import yaml
import time
import collections

from .simulation import simulate_data

import warnings
warnings.filterwarnings("ignore")


class WaveInfo:
    def __init__(self, line: str, type: str):
        split_list = line.strip().split(' ')
        assert type in ['speech', 'noise', 'rir'], type
        if type == 'rir':
            self.utt, self.path = split_list
            self.spk = 'unknown'
            self.fs = None
            self.offset = 0
            self.duration = None
        elif type == 'speech':
            self.utt, self.spk, self.path = split_list
            self.fs = None
            self.offset = 0
            self.duration = None
        elif type == 'noise':
            self.utt, self.fs, start, frames, self.path = split_list
            self.spk = 'unknown'
            self.fs = eval(self.fs)
            self.offset = eval(start) / self.fs
            self.duration = eval(frames) / self.fs


class TrainDataLoadIter:
    def __init__(
        self,
        simulation_config: Union[str, Path],
        speech_scp_path: Union[str, Path, List], 
        noise_scp_path: Union[str, Path, List], 
        rir_scp_path: Union[str, Path, List], 
        speech_scp_base_dir: Union[str, Path] = '',
        batch_size: int = 1, 
        cut_duration: Union[float, List[float]] = 3.0, 
        enroll_duration: float = 5.0,
        num_workers: int = 1, 
        prefetch: int = 0,
        samples_per_epoch: int = 10000,
    ):
        self.is_train = True
        self.batch_size = batch_size
        self.cut_duration = cut_duration
        self.enroll_duration = enroll_duration
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.samples_per_epoch = samples_per_epoch

        with open(simulation_config, "r") as f:
            self.simulation_config = yaml.safe_load(f)
        
        self.speech_scp_base_dir = Path(speech_scp_base_dir)
        self.speech_list = self.load_scp_to_list(speech_scp_path, 'speech')
        # 按说话人分类
        self.spk2speech = collections.defaultdict(list)
        for speech_info in self.speech_list:
            speech_info.path = self.speech_scp_base_dir / speech_info.path
            self.spk2speech[speech_info.spk].append(speech_info)
        self.spk_list = list(self.spk2speech.keys())
        for spk in self.spk_list:
            assert len(self.spk2speech[spk]) > 1

        self.noise_list = self.load_scp_to_list(noise_scp_path, 'noise')
        self.rir_list = self.load_scp_to_list(rir_scp_path, 'rir')
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
    
    def load_scp_to_list(self, scp_path, type):
        path_list = []
        if not isinstance(scp_path, List):
            scp_path = [scp_path]
        for p in scp_path:
            with open(p, 'r') as f:
                for line in f:
                    path_list.append(WaveInfo(line, type))
        return path_list
    
    def pad_or_cut_wav(self, wav, length, offset=None):
        # wav: [1, T]
        if wav.shape[-1] < length: # pad
            wav = np.pad(wav, [(0, 0), (0, length - wav.shape[-1])], mode='wrap')
            return wav, None
        else: # cut
            if offset is None:
                offset = random.randint(0, wav.shape[-1] - length)
            wav = wav[..., offset: offset + length]
            return wav, offset
    
    def normalize_src_tgt(self, src, tgt, low=0.1, high=0.99):
        max_tgt_value = np.max(np.abs(tgt)) + 1e-5
        max_src_value = np.max(np.abs(src)) + 1e-5
        max_value = max(max_tgt_value, max_src_value)
        threshold = high / max_value  # 防止削波

        target_value = random.uniform(low, high)
        factor = min(target_value / max_tgt_value, threshold)
        src = src * factor
        tgt = tgt * factor

        return src, tgt
    
    def normalize_mix_speech_inferf(self, mix, speech, interf, low=0.1, high=0.99):
        a, b, c = np.max(np.abs(mix)), np.max(np.abs(speech)), np.max(np.abs(interf))
        max_value = max(a, b, c) + 1e-5
        min_value = min(a, b, c)

        factor = high / max_value
        if min_value * factor <= low:
            return mix * factor, speech * factor, interf * factor
        else:
            factor = random.uniform(low / (min_value * factor), 1) * factor
            return mix * factor, speech * factor, interf * factor


    def load_wav(self, info: WaveInfo, fs=None):
        wav, fs_ = librosa.load(info.path, dtype=np.float32, sr=fs, mono=False, offset=info.offset, duration=info.duration)
        if wav.ndim == 1:
            wav = wav[None]  # (1, T)
        else:
            wav = wav[:1, :]  # 取第0通道
        return wav, fs_
    
    def load_wav_queue(self, info, fs, q):
        try:
            wav, fs = self.load_wav(info, fs)
            q.put((wav, fs))
        except Exception as e:
            q.put(e)
    
    def load_wav_with_timeout(self, info, fs=None, timeout=1.0):
        result_queue = queue.Queue()
        thread = threading.Thread(target=self.load_wav_queue, args=(info, fs, result_queue))
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError(f"读取音频文件超时：{info.path}")
        
        result = result_queue.get()
        if isinstance(result, Exception):
            raise Exception('load error')
        return result
    
    def process_one_sample(self, fs, cut_duration, mode):
        spk1, spk2 = random.sample(self.spk_list, 2)

        speech_info, enroll_info = random.sample(self.spk2speech[spk1], 2)
        interf_info = random.choice(self.spk2speech[spk2])
        if mode == 'tse' or mode == 'rtse':  # 启用TSE/rTSE模式
            try:
                speech, _ = self.load_wav_with_timeout(speech_info, fs, timeout=2.0)
                enroll, _ = self.load_wav_with_timeout(enroll_info, fs, timeout=2.0)
                interf, _ = self.load_wav_with_timeout(interf_info, fs, timeout=2.0)
            except Exception as e:
                print(e)
                return self.process_one_sample(fs, cut_duration, mode)
        elif mode == 'se' and random.random() < self.simulation_config['se_interference']['prob']:  # SE模式，启用干扰说话人
            try:
                speech, _ = self.load_wav_with_timeout(speech_info, fs, timeout=2.0)
                enroll = None
                interf, _ = self.load_wav_with_timeout(interf_info, fs, timeout=2.0)
            except Exception as e:
                print(e)
                return self.process_one_sample(fs, cut_duration, mode)
        else:  # SE模式，不启用干扰说话人
            try:
                speech, _ = self.load_wav_with_timeout(speech_info, fs, timeout=2.0)
                enroll = None
                interf = None
            except Exception as e:
                print(e)
                return self.process_one_sample(fs, cut_duration, mode)
        
        noise_info = random.choice(self.noise_list)
        noise, _ = self.load_wav(noise_info, fs)
        
        rir_info = random.choice(self.rir_list)
        rir, _ = self.load_wav(rir_info, fs)

        mix, speech, interf = simulate_data(
            mode=mode,
            speech=speech,
            interf=interf,
            noise=noise,
            rir=rir,
            fs=fs,
            config=self.simulation_config,
        )

        if cut_duration is not None:
            length = int(cut_duration * fs)
            mix, offset = self.pad_or_cut_wav(mix, length, offset=None)
            speech, _ = self.pad_or_cut_wav(speech, length, offset)
            if interf is not None:
                interf, _ = self.pad_or_cut_wav(interf, length, offset)
        else:
            length = speech.shape[-1]
        
        if interf is None:
            mix, speech = self.normalize_src_tgt(mix, speech)
        else:
            mix, speech, interf = self.normalize_mix_speech_inferf(mix, speech, interf)

        if enroll is not None:
            enroll, _ = self.pad_or_cut_wav(enroll, int(self.enroll_duration * fs), offset=None)
            enroll = enroll / (np.max(np.abs(enroll)) + 1e-5) * 0.99

        return enroll, mix, speech, interf, fs, length, speech_info.utt
    

    def data_iter_fn(self, q, event):
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for _ in range(len(self)): # for each batch
            fs = 16000 # sample a fs
            cut_duration = self.cut_duration if not isinstance(self.cut_duration, list) else random.uniform(*self.cut_duration)  # sample cut_duration
            mode = random.choice(['se', 'tse', 'rtse'])
            batch_enroll = []
            batch_mix = []
            batch_speech = []
            batch_interf = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, [fs] * self.batch_size, [cut_duration] * self.batch_size, [mode] * self.batch_size):
                enroll, mix, speech, interf, fs, length, name = result
                batch_enroll.append(enroll)
                batch_mix.append(mix)
                batch_speech.append(speech)
                batch_interf.append(interf)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            batch_enroll = torch.from_numpy(np.concatenate(batch_enroll, axis=0)).float() if mode != 'se' else None
            batch_mix = torch.from_numpy(np.concatenate(batch_mix, axis=0)).float()
            batch_speech = torch.from_numpy(np.concatenate(batch_speech, axis=0)).float()
            batch_interf = torch.from_numpy(np.concatenate(batch_interf, axis=0)).float() if mode != 'se' else None
            batch_fs = torch.LongTensor(batch_fs)
            lengths = torch.LongTensor(lengths)
            q.put((mode, batch_enroll, batch_mix, batch_speech, batch_interf, batch_fs, lengths, names))
        event.set()
    
    def __iter__(self):
        q = queue.Queue(maxsize=self.prefetch + 1)
        event = threading.Event()
        worker = threading.Thread(target=self.data_iter_fn, args=(q, event))
        worker.start()
        while not event.is_set() or not q.empty():
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                continue

    def __len__(self):
        """
        :return: number of batches in dataset
        """
        num_batches = int(self.samples_per_epoch // (self.world_size * self.batch_size))
        if self.is_train:
            return num_batches
        else:
            if self.rank < self.samples_per_epoch // self.batch_size - num_batches * self.world_size:
                return num_batches + 1
            else:
                return num_batches



class ValDataLoadIter:
    def __init__(
        self,
        data_enroll_dir: Union[str, Path],
        data_src_dir: Union[str, Path],
        data_tgt_dir: Union[str, Path],
        mode: str,
        enroll_duration: float = 5.0,
        batch_size: int = 1,
        num_workers: int = 1,
        prefetch: int = 0,
    ):
        self.is_train = False
        self.batch_size = batch_size
        self.mode = mode
        self.enroll_duration = enroll_duration

        if data_enroll_dir is not None:
            self.data_enroll_dir = Path(data_enroll_dir)
        else:
            self.data_enroll_dir = None
        self.data_src_dir = Path(data_src_dir)
        self.data_tgt_dir = Path(data_tgt_dir)

        self.wav_names = [p.name for p in self.data_src_dir.glob('*.flac')] + [p.name for p in self.data_src_dir.glob('*.wav')]
        self.num_workers = num_workers
        self.prefetch = prefetch
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    
    def load_wav(self, path, fs=None):
        wav, fs_ = sf.read(path, dtype='float32', always_2d=True)
        wav = wav[:, :1].T
        if fs is not None and fs != fs_:
            wav = librosa.resample(wav, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
            return wav, fs
        return wav, fs_
    
    def process_one_sample(self, name):
        assert self.batch_size == 1
        src, fs1 = self.load_wav(self.data_src_dir / name, fs=16000)
        tgt, fs2 = self.load_wav(self.data_tgt_dir / name, fs=16000)
        if self.data_enroll_dir is not None:
            enroll, fs3 = self.load_wav(self.data_enroll_dir / name, fs=16000)
        else:
            enroll, fs3 = None, None
        
        if enroll is not None:
            length = int(self.enroll_duration * 16000)
            if enroll.shape[-1] < length:
                enroll = np.pad(enroll, [(0, 0), (0, length - enroll.shape[-1])], mode='wrap')
            else:
                enroll = enroll[..., :length]
            enroll = enroll / (np.max(np.abs(enroll)) + 1e-5) * 0.99
        
        length = src.shape[-1]
        return enroll, src, tgt, 16000, length, Path(name).stem

    def data_iter_fn(self, q, event):
        wav_names = deepcopy(self.wav_names)
        assert self.batch_size == 1
        
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for sample_idx in range(self.rank * self.batch_size, len(wav_names), self.world_size * self.batch_size):
            batch_enroll = []
            batch_src = []
            batch_tgt = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, wav_names[sample_idx:sample_idx + self.batch_size]):
                enroll, src, tgt, fs, length, name = result
                batch_enroll.append(enroll)
                batch_src.append(src)
                batch_tgt.append(tgt)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            batch_enroll = torch.from_numpy(np.concatenate(batch_enroll, axis=0)).float() if self.data_enroll_dir else None
            batch_src = torch.from_numpy(np.concatenate(batch_src, axis=0)).float()
            batch_tgt = torch.from_numpy(np.concatenate(batch_tgt, axis=0)).float()
            batch_fs = torch.LongTensor(batch_fs)
            lengths = torch.LongTensor(lengths)
            q.put((self.mode, batch_enroll, batch_src, batch_tgt, batch_fs, lengths, names))
        event.set()

    def __iter__(self):
        q = queue.Queue(maxsize=self.prefetch + 1)
        event = threading.Event()
        worker = threading.Thread(target=self.data_iter_fn, args=(q, event))
        worker.start()
        while not event.is_set() or not q.empty():
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                continue

    def __len__(self):
        """
        :return: number of batches in dataset
        """
        num_batches = int(len(self.wav_names) // (self.world_size * self.batch_size))
        if self.is_train:
            return num_batches
        else:
            if self.rank < len(self.wav_names) // self.batch_size - num_batches * self.world_size:
                return num_batches + 1
            else:
                return num_batches


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_kwargs,
        val_kwargs,
        test_kwargs,
    ):
        super().__init__()
        self.train_kwargs = train_kwargs
        self.val_kwargs = val_kwargs
        self.test_kwargs = test_kwargs

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_iter = TrainDataLoadIter(**self.train_kwargs)
            self.val_iter = TrainDataLoadIter(**self.val_kwargs)
        if stage == 'test' or stage is None:
            self.test_iter = ValDataLoadIter(**self.test_kwargs)

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter

    def test_dataloader(self):
        return self.test_iter

