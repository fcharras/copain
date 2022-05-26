# syntax=docker/dockerfile:1

ARG INCLUDE_FCEUX=yes
ARG BASE=ubuntu:20.04

FROM $BASE AS python_base_config
ENV PYENV_ROOT="/opt/pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
ENV VIRTUAL_ENV=/opt/venv

FROM python_base_config AS python_builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet; apt-get install -y --no-install-recommends \
        make \
        build-essential \
        libssl-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        # for pyenv installation
        git \
        ca-certificates \
     && rm -rf /var/lib/apt/lists/*
RUN curl -L \
    https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer \
    | bash
ARG PYTHON_VERSION=3.10.4
RUN CFLAGS="-O2 -pipe" \
    CONFIGURE_OPTS="--enable-shared --with-computed-gotos" \
    # --enable-optimizations # possible performance improvements if building for local use
    pyenv install -v $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    rm -rf /tmp/*
RUN python -m venv $VIRTUAL_ENV

FROM $BASE AS fceux_builder
ARG FCEUX_TAG=fceux-2.6.4
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet && \
    apt-get install -y --no-install-recommends sudo git gcc g++ make cmake ca-certificates && \
    update-ca-certificates && \
    git clone -b $FCEUX_TAG --depth 1 https://github.com/TASEmulators/fceux.git /tmp/src/ && \
    cd /tmp/src && \
    head -n -2 ./pipelines/linux_build.sh > ./pipelines/build.sh && \
    sed -i 's/sudo/sudo DEBIAN_FRONTEND=noninteractive/' ./pipelines/build.sh && \
    chmod +x ./pipelines/build.sh && \
    ./pipelines/build.sh

FROM $BASE AS fceux_yes
ARG DEBIAN_FRONTEND=noninteractive
COPY --from=fceux_builder /tmp/fceux-*.deb ./
RUN apt-get update --quiet && \
    apt-get install -y --no-install-recommends ./fceux-*.deb && \
    rm -rf /var/lib/apt/lists/* && \
    rm ./fceux-*.deb

FROM $BASE AS fceux_no

FROM fceux_${INCLUDE_FCEUX} AS base
COPY --from=python_builder $PYENV_ROOT $PYENV_ROOT
COPY --from=python_builder $VIRTUAL_ENV $VIRTUAL_ENV

FROM python_base_config AS copain
COPY --from=base / /
# apt packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet; apt-get install -y --no-install-recommends \
        vim \
        nano \
        less \
        lua5.1 \
        luarocks \
        gcc \
        xvfb \
    && rm -rf /var/lib/apt/lists/*
# python venv activate
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ARG PIP_NO_CACHE_DIR=false
ARG PIP_UPGRADE=true
ARG PIP_UPGRADE_STRATEGY=eager
# pip and lua packages
RUN python -m pip install pip \
    && pip install setuptools wheel \
    && rm -rf /tmp/*
RUN pip install \
        ipdb \
        ipython \
        numpy \
        cython \
        flake8 \
    && luarocks --lua-version 5.1 install luaposix \
    && git config --global url.https://github.com/.insteadOf git://github.com/ \
    && luarocks --lua-version 5.1 install lua-struct \
    && git config --global --unset url.https://github.com/.insteadOf \
    && rm -rf /tmp/*
RUN python -m pip install torch \
    && rm -rf /tmp/*
# enable CUDA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
CMD ["/bin/bash"]
