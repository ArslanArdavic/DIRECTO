FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Replace Miniconda with Miniforge. Miniforge ships with mamba pre-installed
# in the base environment and defaults to conda-forge, so there is no separate
# mamba installation step needed — it is simply ready to use immediately after
# the installer runs. This eliminates the circular problem of needing conda's
# slow solver to install conda's fast solver replacement.
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

# Now we can call mamba directly — no installation step needed because
# Miniforge already provides it. The --verbose flag ensures that if this
# step fails, the build log shows exactly which package caused the failure.
RUN mamba env create -f environment.yaml --verbose && \
    mamba clean -afy && \
    conda clean -afy && \
    find /opt/conda -name "*.pyc" -delete && \
    find /opt/conda -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

SHELL ["conda", "run", "-n", "directo", "/bin/bash", "-c"]

# Problem 2+3 fix: editable install
RUN pip install --upgrade pip && pip install -e .

# Problem 4 fix: install DGL explicitly from the correct channel
RUN conda install -y -c dglteam/label/th24_cu121 dgl && conda clean -afy

# Build-time verification
RUN python -c "import torch; print('torch', torch.__version__)"
RUN python -c "import dgl; print('dgl', dgl.__version__)"
RUN python -c "import graph_tool; print('graph_tool', graph_tool.__version__)"
RUN python -c "import pytorch_lightning; print('lightning', pytorch_lightning.__version__)"
RUN python -c "import numpy; print('numpy', numpy.__version__)"

WORKDIR /workspace/src

CMD ["conda", "run", "--no-capture-output", "-n", "directo", "bash"]