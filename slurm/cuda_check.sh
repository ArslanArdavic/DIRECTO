#!/bin/bash
#SBATCH --job-name=cuda_check
#SBATCH --output=/users/arda.arslan/allab/DIRECTO/slurm/log/cuda_check_%j.out
#SBATCH --error=/users/arda.arslan/allab/DIRECTO/slurm/log/cuda_check_%j.err
#SBATCH --time=00:05:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image ghcr.io\#arslanardavic/directo:latest
#SBATCH --container-mounts=/stratch:/stratch
#SBATCH --container-writable

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"

echo "--- Driver level ---"
nvidia-smi

echo "--- PyTorch level ---"
conda run --no-capture-output -n directo python -c "
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
print(f'CUDA version    : {torch.version.cuda}')
print(f'Device count    : {torch.cuda.device_count()}')
print(f'Device name     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

echo "Finished at: $(date)"