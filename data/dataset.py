import math
import random
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from data.utils import load


def center_crop_audio(audio, output_frames: int):
    if len(audio.shape) == 1:
        target_len = audio.shape[0]
    elif len(audio.shape) == 2:
        target_len = audio.shape[1]
    else:
        raise SyntaxError(
            f"Wrong input audio shape {audio.shape}, expected 1 or 2 dimensions!"
        )

    assert (target_len - output_frames) % 2 == 0
    border_len = (target_len - output_frames) // 2

    if border_len == 0:
        return audio

    if len(audio.shape) == 1:
        return audio[border_len:-border_len]
    else:
        return audio[:, border_len:-border_len]


def extract_random_window(audio, chunk_length: int):
    if len(audio.shape) == 1:
        audio_len = audio.shape[0]
    elif len(audio.shape) == 2:
        audio_len = audio.shape[1]
    else:
        raise SyntaxError(
            f"Wrong input audio shape {audio.shape}, expected 1 or 2 dimensions!"
        )

    start = np.random.randint(0, audio_len - chunk_length + 1)

    if len(audio.shape) == 1:
        return audio[start : start + chunk_length], start
    else:
        return audio[:, start : start + chunk_length], start


class SeparationDataset(IterableDataset):
    def __init__(
        self,
        audio_paths: List[Dict[str, str]],
        instruments: List[str],
        sr: int,
        channels: int,
        input_frames: int,
        output_frames: int,
        chunks_per_audio: int,
        randomize: bool = False,
    ):
        self.audio_paths = audio_paths
        self.instruments = instruments
        self.randomize = randomize

        self.input_frames = input_frames
        self.output_frames = output_frames
        self.chunks_per_audio = chunks_per_audio
        self.sr = sr
        self.channels = channels

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or not self.randomize:
            # Single process data loading OR we are in random mode - then load all songs
            audio_paths = self.audio_paths
        else:
            # We're in a worker process and NOT in random mode - just load our partition of the whole set
            # split workload
            per_worker = int(
                math.ceil(len(self.audio_paths) / float(worker_info.num_workers))
            )
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.audio_paths))
            audio_paths = [self.audio_paths[i] for i in range(iter_start, iter_end)]

        # Randomly permute list of songs
        if self.randomize:
            new_audios = [dict() for _ in range(len(audio_paths))]
            # Random mixing of stems
            paths_per_inst = {
                inst: [
                    audio_path[inst]
                    for audio_path in audio_paths
                    if inst in audio_path.keys()
                ]
                for inst in self.instruments
            }
            for inst, paths in paths_per_inst.items():
                order = np.random.permutation(len(audio_paths))
                for i, path in enumerate(paths):
                    new_audios[order[i]][inst] = path
            audio_paths = new_audios

        random.shuffle(audio_paths)

        # Iterate over all examples
        for paths in audio_paths:
            # Load each stem
            source_audios = {}
            for inst, path in paths.items():
                # In this case, read in audio and convert to target sampling rate
                source_audio, _ = load(path, sr=self.sr, mono=(self.channels == 1))
                source_audios[inst] = source_audio

            # Repeatedly extract random chunk from each stem and generate examples
            for _ in range(self.chunks_per_audio):
                chunked_source_audios = {}
                if self.randomize:
                    # Random position for each source independently
                    for inst, audio in source_audios.items():
                        chunk, chunk_start_pos = extract_random_window(
                            audio, self.input_frames
                        )
                        chunked_source_audios[inst] = chunk
                else:
                    # Same position across all stems to keep them in sync
                    start = None
                    for inst, audio in source_audios.items():
                        if start is None:
                            chunk, start = extract_random_window(
                                audio, self.input_frames
                            )
                            chunked_source_audios[inst] = chunk
                        else:
                            chunked_source_audios[inst] = audio[
                                :, start : start + self.input_frames
                            ]

                # Data augmentation applied to stems
                # TODO if self.data_augmentation: use pedalboard to reimplement spot pipeline, + mix-augment (mix same stems)?

                # Create mix as linear mix of stems
                mix = np.sum(np.stack(list(chunked_source_audios.values())), 0)
                mix = np.clip(mix, -1, 1)
                # TODO detect clipping and prevent it in the first place?

                # Return example, with each stem as target
                # (mix: [channels, samples], targets: [channels * targets, samples]
                targets = np.concatenate(
                    [
                        center_crop_audio(
                            chunked_source_audios[inst], self.output_frames
                        )
                        for inst in self.instruments
                    ]
                )
                yield mix, targets
