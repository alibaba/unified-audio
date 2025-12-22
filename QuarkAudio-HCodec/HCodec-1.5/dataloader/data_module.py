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

import warnings
warnings.filterwarnings("ignore")




class TrainDataLoadIter:
    def __init__(
        self,
        audio_scp_path: Union[str, Path, List],
        music_scp_path: Union[str, Path, List],
        speech_scp_path: Union[str, Path, List],
        speech_scp_base_dir: Union[str, Path] = '',
        domain_weights_dict: dict = {
            'speech': 1,
            'music': 1,
            'audio': 1,
        },
        batch_size: int = 1,
        cut_duration: Union[float, List[float]] = 5.0,
        num_workers: int = 1,
        prefetch: int = 0,
        samples_per_epoch: int = 10000,
    ):
        self.is_train = True
        self.batch_size = batch_size
        self.cut_duration = cut_duration
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.samples_per_epoch = samples_per_epoch
        
        self.speech_scp_base_dir = Path(speech_scp_base_dir)
        self.speech_list = self.load_scp_to_list(speech_scp_path)
        for i in range(len(self.speech_list)):
            self.speech_list[i] = str(self.speech_scp_base_dir / self.speech_list[i])
        
        self.audio_list = self.load_scp_to_list(audio_scp_path)
        self.music_list = self.load_scp_to_list(music_scp_path)
        self.domain_classes = [key for key in domain_weights_dict.keys()]
        self.domain_weights = [value for value in domain_weights_dict.values()]

        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
    
    def load_scp_to_list(self, scp_path):
        path_list = []
        if not isinstance(scp_path, List):
            scp_path = [scp_path]
        for p in scp_path:
            with open(p, 'r') as f:
                for line in f:
                    path = line.strip().split(' ')[-1]
                    path_list.append(path)
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

    def normalize_wav(self, wav, low=0.1, high=0.99):
        max_value = np.max(np.abs(wav)) + 1e-5
        target_value = random.uniform(low, high)
        wav = wav * target_value / max_value
        return wav


    def load_wav(self, info: str, fs=None, cut_duration=None):
        wav_duration = librosa.get_duration(path=info)  # 读mp3会舍入到1位小数
        if cut_duration is not None:
            offset = random.uniform(0, max(0, wav_duration - cut_duration))
        else:
            offset = 0.0

        if Path(info).suffix == '.mp3':
            wav, fs_ = librosa.load(info, dtype=np.float32, sr=fs, mono=False)
        else:
            wav, fs_ = librosa.load(info, dtype=np.float32, sr=fs, mono=False, offset=offset, duration=cut_duration)
        
        if wav.ndim == 1:
            wav = wav[None]  # (1, T)
        else:
            wav = wav[:1, :]  # 取第0通道
        
        if cut_duration is not None:
            wav, _ = self.pad_or_cut_wav(wav, length=int(cut_duration * fs_), offset=None)

        return wav, fs_

    
    def load_wav_queue(self, info, fs, cut_duration, q):
        try:
            wav, fs = self.load_wav(info, fs, cut_duration)
            q.put((wav, fs))
        except Exception as e:
            q.put(e)
    

    def load_wav_with_timeout(self, info, fs=None, cut_duration=None, timeout=1.0):
        result_queue = queue.Queue()
        thread = threading.Thread(target=self.load_wav_queue, args=(info, fs, cut_duration, result_queue))
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError(f"读取音频文件超时：{info}")
        
        result = result_queue.get()
        if isinstance(result, Exception):
            raise Exception('load error')
        return result
    

    def process_one_sample(self, fs, cut_duration):
        domain = random.choices(self.domain_classes, weights=self.domain_weights, k=1)[0]
        if domain == 'speech':
            wav_path = random.choice(self.speech_list)
        elif domain == 'music':
            wav_path = random.choice(self.music_list)
        else:
            wav_path = random.choice(self.audio_list)
        
        try:
            wav, _ = self.load_wav_with_timeout(wav_path, fs, cut_duration, timeout=10.0)
        except Exception as e:
            print(e)
            return self.process_one_sample(fs, cut_duration)

        length = wav.shape[-1]
        wav = self.normalize_wav(wav)

        return wav, fs, length, Path(wav_path).stem

    def data_iter_fn(self, q, event):
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for _ in range(len(self)): # for each batch
            fs = 16000 # sample a fs
            cut_duration = self.cut_duration if not isinstance(self.cut_duration, list) else random.uniform(*self.cut_duration)  # sample cut_duration
            batch_wav = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, [fs] * self.batch_size, [cut_duration] * self.batch_size):
                wav, fs, length, name = result
                batch_wav.append(wav)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            batch_wav = torch.from_numpy(np.concatenate(batch_wav, axis=0)).float()
            batch_fs = torch.LongTensor(batch_fs)
            lengths = torch.LongTensor(lengths)
            q.put((batch_wav, batch_fs, lengths, names))
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
        speech_scp_path: Union[str, Path], 
        music_scp_path: Union[str, Path], 
        audio_scp_path: Union[str, Path], 
        cut_duration: Union[float, List[float]],
        batch_size: int = 1, 
        num_workers: int = 1, 
        prefetch: int = 0,
        samples_per_epoch: int = 1000,
    ):
        self.is_train = False

        self.speech_list = self.load_scp_to_list(speech_scp_path)
        self.music_list = self.load_scp_to_list(music_scp_path)
        self.audio_list = self.load_scp_to_list(audio_scp_path)
        self.domain_range_list = ['speech', 'music', 'audio'] * (samples_per_epoch // 3 + 1)
        self.domain_range_list = self.domain_range_list[:samples_per_epoch]

        self.cut_duration = cut_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.samples_per_epoch = samples_per_epoch
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
    
    def load_scp_to_list(self, scp_path):
        path_list = []
        if not isinstance(scp_path, List):
            scp_path = [scp_path]
        for p in scp_path:
            with open(p, 'r') as f:
                for line in f:
                    path = line.strip().split(' ')[-1]
                    path_list.append(path)
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

    def load_wav(self, info: str, fs=None, cut_duration=None):
        wav_duration = librosa.get_duration(path=info)  # 读mp3会舍入到1位小数
        if cut_duration is not None:
            offset = random.uniform(0, max(0, wav_duration - cut_duration))
        else:
            offset = 0.0

        wav, fs_ = librosa.load(info, dtype=np.float32, sr=fs, mono=False, offset=offset, duration=cut_duration)
        if wav.ndim == 1:
            wav = wav[None]  # (1, T)
        else:
            wav = wav[:1, :]  # 取第0通道
        
        if cut_duration is not None:
            wav, _ = self.pad_or_cut_wav(wav, length=int(cut_duration * fs_), offset=None)

        return wav, fs_
    

    def process_one_sample(self, fs, cut_duration, domain):
        if domain == 'speech':
            wav_path = random.choice(self.speech_list)
        elif domain == 'music':
            wav_path = random.choice(self.music_list)
        else:
            wav_path = random.choice(self.audio_list)
        
        try:
            wav, _ = self.load_wav(wav_path, fs, cut_duration)
        except Exception as e:
            print(e)
            return self.process_one_sample(fs, cut_duration, domain)

        length = wav.shape[-1]
        return domain, wav, fs, length, Path(wav_path).stem
    

    def data_iter_fn(self, q, event):
        assert self.batch_size == 1
        
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for sample_idx in range(self.rank * self.batch_size, self.samples_per_epoch, self.world_size * self.batch_size):
            fs = 16000 # sample a fs
            cut_duration = self.cut_duration if not isinstance(self.cut_duration, list) else random.uniform(*self.cut_duration)  # sample cut_duration
            batch_domain = []
            batch_wav = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, [fs] * self.batch_size, [cut_duration] * self.batch_size, self.domain_range_list[sample_idx:sample_idx + self.batch_size]):
                domain, wav, fs, length, name = result
                batch_domain.append(domain)
                batch_wav.append(wav)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            batch_wav = torch.from_numpy(np.concatenate(batch_wav, axis=0)).float()
            batch_fs = torch.LongTensor(batch_fs)
            lengths = torch.LongTensor(lengths)
            q.put((batch_domain, batch_wav, batch_fs, lengths, names))
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


class TestDataLoadIter:
    def __init__(
        self, 
        wav_scp_path: Union[str, Path],
        domain: str,
        batch_size: int = 1, 
        num_workers: int = 1, 
        prefetch: int = 0,
    ):
        self.wav_list = self.load_scp_to_list(wav_scp_path)

        self.is_train = False
        self.domain = domain
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch = prefetch
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
    
    def load_scp_to_list(self, scp_path):
        path_list = []
        if not isinstance(scp_path, List):
            scp_path = [scp_path]
        for p in scp_path:
            with open(p, 'r') as f:
                for line in f:
                    path = line.strip().split(' ')[-1]
                    path_list.append(path)
        return path_list

    def load_wav(self, path, fs=None):
        wav, fs_ = librosa.load(path, dtype=np.float32, sr=fs, mono=False)
        if wav.ndim == 1:
            wav = wav[None]  # (1, T)
        else:
            wav = wav[:1, :]  # 取第0通道
        return wav, fs_
    
    def process_one_sample(self, path):
        assert self.batch_size == 1
        wav, fs = self.load_wav(path, fs=16000)
        assert fs == 16000

        # src, tgt, norm_factor = self.normalize_src_tgt(src, tgt)
        length = wav.shape[-1]
        return wav, fs, length, Path(path).name

    def data_iter_fn(self, q, event):
        wav_list = deepcopy(self.wav_list)
        assert self.batch_size == 1
        
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for sample_idx in range(self.rank * self.batch_size, len(wav_list), self.world_size * self.batch_size):
            batch_domain = []
            batch_wav = []
            batch_fs = []
            lengths = []
            names = []
            for result in executor.map(self.process_one_sample, wav_list[sample_idx:sample_idx + self.batch_size]):
                wav, fs, length, name = result
                batch_domain.append(self.domain)
                batch_wav.append(wav)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            batch_wav = torch.from_numpy(np.concatenate(batch_wav, axis=0)).float()
            batch_fs = torch.LongTensor(batch_fs)
            lengths = torch.LongTensor(lengths)
            q.put((batch_domain, batch_wav, batch_fs, lengths, names))
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
        num_batches = int(len(self.wav_list) // (self.world_size * self.batch_size))
        if self.is_train:
            return num_batches
        else:
            if self.rank < len(self.wav_list) // self.batch_size - num_batches * self.world_size:
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
            self.val_iter = ValDataLoadIter(**self.val_kwargs)
        if stage == 'test' or stage is None:
            self.test_iter = TestDataLoadIter(**self.test_kwargs)

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter

    def test_dataloader(self):
        return self.test_iter

