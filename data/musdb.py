import musdb
import os
import numpy as np
import glob

from data.utils import load, write_wav


def get_musdbhq(database_path):
    '''
    Retrieve audio file paths for MUSDB HQ dataset
    :param database_path: MUSDB HQ root directory
    :return: dictionary with train and test keys, each containing list of samples, each sample containing all audio paths
    '''
    subsets = list()

    for subset in ["train", "test"]:
        print("Loading " + subset + " set...")
        tracks = glob.glob(os.path.join(database_path, subset, "*"))
        samples = list()

        # Go through tracks
        for track_folder in sorted(tracks):
            # Skip track if mixture is already written, assuming this track is done already
            example = dict()
            for stem in ["mix", "bass", "drums", "other", "vocals"]:
                filename = stem if stem != "mix" else "mixture"
                audio_path = os.path.join(track_folder, filename + ".wav")
                example[stem] = audio_path

            # Add other instruments to form accompaniment
            acc_path = os.path.join(track_folder, "accompaniment.wav")

            if not os.path.exists(acc_path):
                print("Writing accompaniment to " + track_folder)
                stem_audio = []
                for stem in ["bass", "drums", "other"]:
                    audio, sr = load(example[stem], sr=None, mono=False)
                    stem_audio.append(audio)
                acc_audio = np.clip(sum(stem_audio), -1.0, 1.0)
                write_wav(acc_path, acc_audio, sr)

            example["accompaniment"] = acc_path

            samples.append(example)

        subsets.append(samples)

    return subsets

def get_musdb(database_path):
    '''
    Retrieve audio file paths for MUSDB dataset
    :param database_path: MUSDB root directory
    :return: dictionary with train and test keys, each containing list of samples, each sample containing all audio paths
    '''
    mus = musdb.DB(root=database_path, is_wav=False)

    subsets = list()

    for subset in ["train", "test"]:
        tracks = mus.load_mus_tracks(subset)
        samples = list()

        # Go through tracks
        for track in sorted(tracks):
            # Skip track if mixture is already written, assuming this track is done already
            track_path = track.path[:-4]
            mix_path = track_path + "_mix.wav"
            acc_path = track_path + "_accompaniment.wav"
            if os.path.exists(mix_path):
                print("WARNING: Skipping track " + mix_path + " since it exists already")

                # Add paths and then skip
                paths = {"mix" : mix_path, "accompaniment" : acc_path}
                paths.update({key : track_path + "_" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})

                samples.append(paths)

                continue

            rate = track.rate

            # Go through each instrument
            paths = dict()
            stem_audio = dict()
            for stem in ["bass", "drums", "other", "vocals"]:
                path = track_path + "_" + stem + ".wav"
                audio = track.targets[stem].audio
                write_wav(path, audio, rate)
                stem_audio[stem] = audio
                paths[stem] = path

            # Add other instruments to form accompaniment
            acc_audio = np.clip(sum([stem_audio[key] for key in list(stem_audio.keys()) if key != "vocals"]), -1.0, 1.0)
            write_wav(acc_path, acc_audio, rate)
            paths["accompaniment"] = acc_path

            # Create mixture
            mix_audio = track.audio
            write_wav(mix_path, mix_audio, rate)
            paths["mix"] = mix_path

            diff_signal = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
            print("Maximum absolute deviation from source additivity constraint: " + str(np.max(diff_signal)))# Check if acc+vocals=mix
            print("Mean absolute deviation from source additivity constraint:    " + str(np.mean(diff_signal)))

            samples.append(paths)

        subsets.append(samples)

    print("DONE preparing dataset!")
    return subsets

def get_musdb_folds(root_path, version="HQ"):
    if version == "HQ":
        dataset = get_musdbhq(root_path)
    else:
        dataset = get_musdb(root_path)
    train_val_list = dataset[0]
    test_list = dataset[1]

    np.random.seed(1337) # Ensure that partitioning is always the same on each run
    train_list = np.random.choice(train_val_list, 75, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]
    # print("First training song: " + str(train_list[0])) # To debug whether partitioning is deterministic
    return {"train" : train_list, "val" : val_list, "test" : test_list}