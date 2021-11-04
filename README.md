# Wave-U-Net (Pytorch)
<a href="https://replicate.ai/f90/wave-u-net-pytorch"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=darkgreen" height=20></a>

Improved version of the [Wave-U-Net](https://arxiv.org/abs/1806.03185) for audio source separation, implemented in Pytorch.

Click [here](www.github.com/f90/Wave-U-Net) for the original Wave-U-Net implementation in Tensorflow.
You can find more information about the model and results there as well.

# Improvements

* Multi-instrument separation by default, using a separate standard Wave-U-Net for each source (can be set to one model as well)
* More scalable to larger data: A depth parameter D can be set that employs D convolutions for each single convolution in the original Wave-U-Net  
* More configurable: Layer type, resampling factor at each level etc. can be easily changed (different normalization, residual connections...)
* Fast training: Preprocesses the given dataset by saving the audio into HDF files, which can be read very quickly during training, thereby avoiding slowdown due to resampling and decoding
* Modular thanks to Pytorch: Easily replace components of the model with your own variants/layers/losses
* Better output handling: Separate output convolution for each source estimate with linear activation so amplitudes near 1 and -1 can be easily predicted, at test time thresholding to valid amplitude range [-1,1]
* Fixed or dynamic resampling: Either use fixed lowpass filter to avoid aliasing during resampling, or use a learnable convolution

# Installation

GPU strongly recommended to avoid very long training times.

### Option 1: Direct install (recommended)

System requirements:
* Linux-based OS
* Python 3.6

* [libsndfile](http://mega-nerd.com/libsndfile/) 

* [ffmpeg](https://www.ffmpeg.org/)
* CUDA 10.1 for GPU usage

Clone the repository:
```
git clone https://github.com/f90/Wave-U-Net-Pytorch.git
```

Recommended: Create a new virtual environment to install the required Python packages into, then activate the virtual environment:

```
virtualenv --python /usr/bin/python3.6 waveunet-env
source waveunet-env/bin/activate
```

Install all the required packages listed in the ``requirements.txt``:

```
pip3 install -r requirements.txt
```

### Option 2: Singularity

We also provide a Singularity container which allows you to avoid installing the correct Python, CUDA and other system libraries, however we don't provide specific advice on how to run the container and so only do this if you have to or know what you are doing (since you need to mount dataset paths to the container etc.)

To pull the container, run
```
singularity pull shub://f90/Wave-U-Net-Pytorch
```

Then run the container from the directory where you cloned this repository to, using the commands listed further below in this readme.

# Download datasets

To directly use the pre-trained models we provide for download to separate your own songs, now skip directly to the [last section](#test), since the datasets are not needed in that case.

To start training your own models, download the [full MUSDB18HQ dataset](https://sigsep.github.io/datasets/musdb.html) and extract it into a folder of your choice. It should have two subfolders: "test" and "train" as well as a README.md file.

You can of course use your own datasets for training, but for this you would need to modify the code manually, which will not be discussed here. However, we provide a loading function for the normal MUSDB18 dataset as well.

# Training the models

To train a Wave-U-Net, the basic command to use is

```
python3.6 train.py --dataset_dir /PATH/TO/MUSDB18HQ 
```
where the path to MUSDB18HQ dataset needs to be specified, which contains the ``train`` and ``test`` subfolders.

Add more command line parameters as needed:
* ``--cuda`` to activate GPU usage
* ``--hdf_dir PATH`` to save the preprocessed data (HDF files) to custom location PATH, instead of the default ``hdf`` subfolder in this repository
* ``--checkpoint_dir`` and ``--log_dir`` to specify where checkpoint files and logs are saved/loaded
* ``--load_model checkpoints/model_name/checkpoint_X`` to start training with weights given by a certain checkpoint

For more config options, see ``train.py``.

Training progress can be monitored by using Tensorboard on the respective ``log_dir``.
After training, the model is evaluated on the MUSDB18HQ test set, and SDR/SIR/SAR metrics are reported for all instruments and written into both the Tensorboard, and in more detail also into a ``results.pkl`` file in the ``checkpoint_dir``

# <a name="test"></a> Test trained models on songs!

We provide the default model in a pre-trained form as download so you can separate your own songs right away.

## Downloading our pretrained models

Download our pretrained model [here](https://www.dropbox.com/s/r374hce896g4xlj/models.7z?dl=1).
Extract the archive into the ``checkpoints`` subfolder in this repository, so that you have one subfolder for each model (e.g. ``REPO/checkpoints/waveunet``)

## Run pretrained model

To apply our pretrained model to any of your own songs, simply point to its audio file path using the ``input_path`` parameter:

```
python3.6 predict.py --load_model checkpoints/waveunet/model --input "audio_examples/Cristina Vane - So Easy/mix.mp3"
```

* Add ``--cuda `` when using a GPU, it should be much quicker
* Point ``--input`` to the music file you want to separate

By default, output is written where the input music file is located, using the original file name plus the instrument name as output file name. Use ``--output`` to customise the output directory.

To run your own model:
* Point ``--load_model`` to the checkpoint file of the model you are using. If you used non-default hyper-parameters to train your own model, you must specify them here again so the correct model is set up and can receive the weights!
