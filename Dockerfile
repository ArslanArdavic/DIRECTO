FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Miniforge ships with mamba pre-installed and defaults to conda-forge,
# eliminating the circular problem of needing conda to install mamba.
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh
ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /workspace
COPY . .

# Problem 1 fix: remove the invalid '- e .' pip entry from environment.yaml
RUN sed -i '/^\s*- e \./d' environment.yaml

# Problem 7 fix: rewrite the NumPy pin to <2.0 before env creation so the
# environment is built correctly from the start
RUN sed -i 's/numpy==2\.3\.1/numpy<2.0/' environment.yaml

# Create the conda environment via mamba.
# --verbose ensures failures show exactly which package caused the problem.
RUN mamba env create -f environment.yaml --verbose && \
    mamba clean -afy && \
    conda clean -afy && \
    find /opt/conda -name "*.pyc" -delete && \
    find /opt/conda -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

SHELL ["conda", "run", "-n", "directo", "/bin/bash", "-c"]

# Force-install PyTorch CUDA 12.1 via pip using PyTorch's official wheel server.
# conda's solver probes for GPU hardware at install time and silently selects
# the CPU-only build when running on GPU-less hardware such as GitHub Actions
# runners. pip's wheel selection is based purely on the index URL — pointing it
# at the cu121 wheel server guarantees the CUDA build unconditionally regardless
# of what hardware is present at build time.
# We uninstall first to remove whatever build conda placed during env creation.
RUN pip uninstall -y torch torchvision torchaudio 2>/dev/null || true && \
    pip install \
        torch==2.4.0+cu121 \
        torchvision==0.19.0+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Assert that the CUDA build is actually installed.
# Fails the Docker build immediately if a CPU-only build ended up installed,
# preventing a broken image from ever being pushed to the registry.
RUN python -c "import torch; assert torch.version.cuda is not None, 'CPU-only PyTorch — check wheel URL'; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} — OK')"

# Problem 2+3 fix: editable install
RUN pip install --upgrade pip && pip install -e .

# Problem 4 fix: install DGL explicitly from the correct channel.
# No 'conda run -n directo' prefix needed — the SHELL directive above already
# executes every RUN command inside the directo environment. Adding it would
# cause double-nesting and a conda-inside-conda failure.
RUN conda install -y -c dglteam/label/th24_cu121 dgl && conda clean -afy

# Build-time verification — each line prints the installed version so the
# build log serves as a permanent record of what is inside this image.
RUN python -c "import torch; print('torch', torch.__version__)"
RUN python -c "import dgl; print('dgl', dgl.__version__)"
RUN python -c "import graph_tool; print('graph_tool', graph_tool.__version__)"
RUN python -c "import pytorch_lightning; print('lightning', pytorch_lightning.__version__)"
RUN python -c "import numpy; print('numpy', numpy.__version__)"

WORKDIR /workspace/src

CMD ["conda", "run", "--no-capture-output", "-n", "directo", "bash"]
