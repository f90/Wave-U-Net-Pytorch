BootStrap: docker
From: nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

%post
    # Create extra folders that JADE HPC can bind to
    # mkdir /tmp # This already exists
    mkdir /local_scratch
    mkdir /raid
    mkdir /raid/local_scratch

    # Downloads the latest package lists (important).
    apt-get update -y

    # Add Python 3.6 PPA (not available in 16.04 Ubuntu)
    apt-get install -y software-properties-common && \
        add-apt-repository ppa:jonathonf/python-3.6
    apt-get update -y

    # Runs apt-get while ensuring that there are no user prompts that would
    # cause the build process to hang.
    # python3-tk is required by matplotlib.
    # python3-dev is needed to require some packages.
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.6 \
        python3.6-tk \
        python3-pip \
        python3.6-dev \
        python3.6-venv \
        libsndfile1 \
        libsndfile1-dev \
        ffmpeg \
        git
    # Reduce the size of the image by deleting the package lists we downloaded,
    # which are useless now.
    rm -rf /var/lib/apt/lists/*

    # Install Python modules. Ensure it is installed for Python 3.6 not pre-shipped Python 3.5
    python3.6 -m pip install future numpy librosa musdb museval h5py tqdm sortedcontainers
    python3.6 -m pip install torch==1.1.0 torchvision==0.4.0 tensorboard

    # Set python alias to Python 3.6
    ln -s /usr/bin/python3.6 /usr/bin/python
    ln -s /usr/bin/pip3 /usr/bin/pip

    # update-alternatives --set python /usr/bin/python3.6

%environment
    # Pipenv requires a certain terminal encoding.
    #export LANG=C.UTF-8
    #export LC_ALL=C.UTF-8
    # This configures Pipenv to store the packages in the current working
    # directory.
    #export PIPENV_VENV_IN_PROJECT=1
