import torch
import soundfile as sf
from typing import Union, List
from pathlib import Path
import numpy as np
import os
import pytorch_lightning as pl
from copy import deepcopy
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import librosa


class TestDataLoadIter:
    """通用推理数据加载器，支持 mono/stereo wav"""
    def __init__(self, speech_dir, batch_size=1, num_workers=1, prefetch=0):
        self.batch_size = batch_size
        self.speech_dir = Path(speech_dir)
        self.wav_names = [p for p in self.speech_dir.rglob('*.flac')] + [p for p in self.speech_dir.rglob('*.wav')]
        self.num_workers = num_workers
        self.prefetch = prefetch
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def parse_spatial_info(self, wav_path):
        """从文件名解析空间信息 (azimuth, elevation, distance)"""
        rir_name = os.path.basename(str(wav_path))
        try:
            az  = max(0, min(359, float(rir_name[rir_name.find('_azi')+5:rir_name.find('_ele')])))
            el  = max(0, min(179, float(rir_name[rir_name.find('_ele')+5:rir_name.find('_dist')])+90))
            dis = max(0, min(49,  float(rir_name[rir_name.find('_dist')+6:rir_name.find('.wav')])*10))
            return np.array([[az, el, dis]]).astype(np.int64)
        except:
            return np.array([[0, 90, 10]]).astype(np.int64)

    def load_wav(self, path, fs=None):
        wav, fs_ = librosa.load(path, dtype=np.float32, sr=fs, mono=False)
        if wav.ndim == 1:
            wav = wav[None]
        return wav, fs_

    def process_one_sample(self, path):
        wav, fs = self.load_wav(path, fs=16000)
        spatial_info = self.parse_spatial_info(path)
        length = wav.shape[-1]
        wav = np.expand_dims(wav, axis=0)
        return wav, spatial_info, fs, length, path.name

    def data_iter_fn(self, q, event):
        wav_names = deepcopy(self.wav_names)
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for sample_idx in range(self.rank * self.batch_size, len(wav_names), self.world_size * self.batch_size):
            batch_wav, batch_info, batch_fs, lengths, names = [], [], [], [], []
            for result in executor.map(self.process_one_sample, wav_names[sample_idx:sample_idx + self.batch_size]):
                wav, info, fs, length, name = result
                batch_wav.append(wav)
                batch_info.append(info)
                batch_fs.append(fs)
                lengths.append(length)
                names.append(name)
            batch_wav = torch.from_numpy(np.concatenate(batch_wav, axis=0)).float()
            batch_info = torch.from_numpy(np.concatenate(batch_info, axis=0)).float()
            batch_fs = torch.LongTensor(batch_fs)
            lengths = torch.LongTensor(lengths)
            q.put((batch_wav, batch_info, batch_fs, lengths, names))
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
        num_batches = int(len(self.wav_names) // (self.world_size * self.batch_size))
        if self.rank < len(self.wav_names) // self.batch_size - num_batches * self.world_size:
            return num_batches + 1
        return num_batches


class DataModule(pl.LightningDataModule):
    def __init__(self, test_kwargs):
        super().__init__()
        self.test_kwargs = test_kwargs

    def setup(self, stage=None):
        self.test_iter = TestDataLoadIter(**self.test_kwargs)

    def test_dataloader(self):
        return self.test_iter
