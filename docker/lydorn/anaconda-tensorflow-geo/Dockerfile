FROM nvidia/cuda:9.0-devel-ubuntu16.04

# Install useful tools

RUN apt-get update && apt-get install -y \
    wget \
    git \
    sudo \
    libcudnn7=7.0.5.15-1+cuda9.0 \
    libcudnn7-dev=7.0.5.15-1+cuda9.0 \
    fish

RUN rm -rf /var/lib/apt/lists/*

# Setup main volume

ENV WORKDIR=/workspace
VOLUME $WORKDIR
WORKDIR $WORKDIR

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
RUN apt-get update && apt-get install -y libgtk2.0

# Install Anaconda3
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH

# Install with conda
RUN conda update conda \
 && conda install python=3.6.8

# Install gdal
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-add-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update
RUN apt-get install -y libgdal-dev

# Cleanup
RUN apt-get clean && \
     apt-get autoremove

RUN conda install -c conda-forge tensorflow-gpu \
 && conda install -c menpo opencv3 \
 && conda install -c conda-forge scikit-image \
 && conda install -c conda-forge pyproj \
 && conda install -c conda-forge shapely \
 && conda install -c conda-forge cython \
 && conda install -c conda-forge pycocotools \
 && conda install -c conda-forge overpy \
 && conda install -c conda-forge jsmin \
 && conda install -c conda-forge joblib

# Additional libraries and version corrections
RUN conda install -c conda-forge gdal
RUN pip install msgpack
RUN pip install tensorflow==1.9
RUN pip install tornado==5.1

CMD fish
