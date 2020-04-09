BootStrap: docker
From: nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

%post
    # Downloads the latest package lists (important).
    apt-get update -y
    # Runs apt-get while ensuring that there are no user prompts that would
    # cause the build process to hang.
    # python3-tk is required by matplotlib.
    # python3-dev is needed to require some packages.
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 \
        python3-tk \
        python3-pip \
        python3-dev \
        libsndfile1 \
        libsndfile1-dev \
        ffmpeg \
        git
    # Reduce the size of the image by deleting the package lists we downloaded,
    # which are useless now.
    rm -rf /var/lib/apt/lists/*

    # Install Pipenv.
    pip3 install pipenv

    # Install Python modules.
    pip3 install future numpy librosa musdb museval h5py tqdm sortedcontainers soundfile
    pip3 install torch==1.4.0 torchvision==0.5.0 tensorboard

%environment
    # Pipenv requires a certain terminal encoding.
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    # This configures Pipenv to store the packages in the current working
    # directory.
    export PIPENV_VENV_IN_PROJECT=1
