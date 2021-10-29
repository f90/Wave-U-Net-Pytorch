import math
import random
from typing import List, Dict

import numpy as np
import torch
from pedalboard import Pedalboard
from pedalboard_native import Gain, Reverb
from torch.utils.data import Dataset, IterableDataset

from data.utils import load


def center_crop_audio(audio, output_crop: float):
    assert len(audio.shape) <= 3
    target_len = audio.shape[-1]

    assert 0.0 <= output_crop < 0.5
    border_len = int(round(output_crop * target_len))
    assert 2 * border_len < target_len

    if border_len == 0:
        return audio

    return audio[..., border_len:-border_len]


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


def maybe_add_effect(pedal_effects, effect, aug_apply_prob):
    if np.random.uniform(0, 1) <= aug_apply_prob:
        pedal_effects.append(effect)


def random_augment(audio, sr, aug_apply_prob: float = 0.25):
    # Make a Pedalboard object, containing multiple plugins:
    pedal_effects = []
    maybe_add_effect(
        pedal_effects, Gain(gain_db=np.random.uniform(-20, 0)), aug_apply_prob
    )
    maybe_add_effect(
        pedal_effects, Reverb(room_size=np.random.uniform(0.1, 0.9)), aug_apply_prob
    )
    board = Pedalboard(
        pedal_effects,
        sample_rate=sr,
    )

    # Run the audio through this pedalboard!
    effected = board(audio)

    # TODO pitch shifting makes us lose audio samples
    """
    if np.random.uniform(0, 1) <= aug_apply_prob:
        # create a transformer
        tfm = sox.Transformer()
        # shift the pitch up by 2 semitones
        tfm.pitch(np.random.uniform(-6, 6))
        # transform an in-memory array and return an array
        out = tfm.build_array(input_array=effected.T, sample_rate_in=sr).T
        if out.shape != effected.shape:
            if out.shape[1] > effected.shape[1]:
                assert out.shape[1] - effected.shape[1] < 5
                out = out[:, : effected.shape[1]]
            else:
                raise ValueError(
                    f"Shape of audio {effected.shape} was affected by pitch shifting and is now {out.shape}"
                )
        effected = out
    """

    return effected


class SeparationDataset(IterableDataset):
    def __init__(
        self,
        audio_paths: List[Dict[str, str]],
        instruments: List[str],
        sr: int,
        channels: int,
        input_frames: int,
        output_crop: float,
        chunks_per_audio: int,
        randomize: bool = False,
        augment: bool = False,
    ):
        self.audio_paths = audio_paths
        self.instruments = instruments
        self.randomize = randomize

        self.input_frames = input_frames
        self.output_crop = output_crop
        self.chunks_per_audio = chunks_per_audio
        self.sr = sr
        self.channels = channels
        self.augment = augment

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
                if self.augment:
                    for inst in chunked_source_audios.keys():
                        chunked_source_audios[inst] = random_augment(
                            chunked_source_audios[inst], self.sr
                        )

                # Create mix as linear mix of stems
                mix = np.sum(np.stack(list(chunked_source_audios.values())), 0)
                mix = np.clip(mix, -1, 1)
                # TODO detect clipping and prevent it in the first place?

                # Return example, with each stem as target
                # (mix: [channels, samples], targets: [channels * targets, samples]
                targets = np.concatenate(
                    [
                        center_crop_audio(chunked_source_audios[inst], self.output_crop)
                        for inst in self.instruments
                    ]
                )
                yield mix, targets
