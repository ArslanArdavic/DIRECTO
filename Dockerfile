# ── Base image ────────────────────────────────────────────────────────────────
# We start from NVIDIA's official CUDA 12.1 image with cuDNN 8 and the full
# development toolkit (headers, compilers). The 'devel' variant is necessary
# because some conda packages compile CUDA extensions during installation.
# Ubuntu 22.04 is chosen for broad library compatibility with graph-tool.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent apt from asking interactive questions during package installation.
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ────────────────────────────────────────────────────────
# wget:        to download the Miniconda installer
# git:         conda may need it to fetch some packages
# build-essential: C/C++ compiler toolchain required by some conda packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda installation ─────────────────────────────────────────────────────
# We install Miniconda rather than using pip alone because graph-tool 2.97 is
# only reliably distributed as a pre-compiled conda package. Trying to build
# it from source is fragile and adds significant complexity.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH so all subsequent RUN commands can use it directly.
ENV PATH="/opt/conda/bin:$PATH"

# ── Copy repository into the image ────────────────────────────────────────────
# We copy the full repo first so conda can read environment.yaml and pip can
# read pyproject.toml. The WORKDIR is the project root throughout setup.
WORKDIR /workspace
COPY . .

# ── Problem 1 fix: clean environment.yaml ─────────────────────────────────────
# The '- e .' line in the pip section of environment.yaml is not valid syntax
# for conda's pip integration — it only works as a CLI flag, not a requirement
# specifier. We remove it here so conda env create runs without errors.
# The sed command is a no-op if the line was already removed in the repo.
RUN sed -i '/^\s*- e \./d' environment.yaml

# ── Problem 7 partial fix: pin NumPy in environment.yaml before env creation ──
# environment.yaml currently pins numpy==2.3.1, which is incompatible with
# PyTorch Lightning 2.0.4 (which uses the removed np.Inf alias). We rewrite
# the numpy pin to <2.0 directly in the YAML so the conda env is created
# with the correct version from the start, avoiding the need for a post-hoc
# downgrade step.
RUN sed -i 's/numpy==2\.3\.1/numpy<2.0/' environment.yaml

# ── Create the conda environment ──────────────────────────────────────────────
# This is the heaviest step — it downloads PyTorch, CUDA libraries, graph-tool,
# and all pip dependencies. It will take 20-40 minutes on the first build.
# Docker layer caching means this layer is only re-executed when environment.yaml
# changes, so day-to-day code changes skip straight past this step.
RUN conda env create -f environment.yaml && \
    conda clean -afy    # remove package tarballs to keep the image size down

# Make the directo conda environment the default shell for all subsequent steps.
SHELL ["conda", "run", "-n", "directo", "/bin/bash", "-c"]

# ── Problem 2+3 fix: editable install ─────────────────────────────────────────
# We upgrade pip first because older pip versions do not support pyproject.toml
# based editable installs and require a setup.py. The upgraded pip handles the
# pyproject.toml format correctly, registering src/ on the Python path so all
# internal imports resolve as expected at runtime.
RUN pip install --upgrade pip && pip install -e .

# ── Problem 4 fix: install DGL explicitly ─────────────────────────────────────
# DGL is silently omitted by conda's solver when resolving the dglteam channel
# during env creation. We install it in a separate layer after the environment
# exists, using the channel tag that matches our exact combination of
# PyTorch 2.4 and CUDA 12.1. The -y flag suppresses the confirmation prompt
# which would hang indefinitely in a non-interactive build context.
RUN conda install -y -c dglteam/label/th24_cu121 dgl && \
    conda clean -afy

# ── Verify the environment is healthy ─────────────────────────────────────────
# We run a quick import check at build time so any remaining environment issues
# surface during the image build rather than at job submission time on the cluster.
# If any of these imports fail, the docker build itself fails — giving you
# immediate feedback rather than a mysterious SLURM job failure hours later.
RUN python -c "import torch; print('torch', torch.__version__)"
RUN python -c "import dgl; print('dgl', dgl.__version__)"
RUN python -c "import graph_tool; print('graph_tool', graph_tool.__version__)"
RUN python -c "import pytorch_lightning; print('lightning', pytorch_lightning.__version__)"
RUN python -c "import numpy; print('numpy', numpy.__version__)"

# ── Runtime working directory ──────────────────────────────────────────────────
# main.py must be invoked from inside src/ because Hydra resolves config paths
# relative to the working directory at launch time. Setting WORKDIR here means
# that SLURM job scripts can call python main.py without needing to cd first.
WORKDIR /workspace/src

# ── Default command ────────────────────────────────────────────────────────────
# Drops into an interactive bash shell with the directo environment active.
# In practice SLURM overrides this with the actual job command, but it is
# useful for debugging by running the container interactively.
CMD ["conda", "run", "--no-capture-output", "-n", "directo", "bash"]
