#!/bin/bash
#SBATCH --job-name=directo_search_er_dag
#SBATCH --output=/users/arda.arslan/allab/DIRECTO/slurm/log/phase2_search/search_%j.out
#SBATCH --error=/users/arda.arslan/allab/DIRECTO/slurm/log/phase2_search/search_%j.err
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image ghcr.io\#arslanardavic/directo:latest
#SBATCH --container-mounts=/stratch:/stratch
#SBATCH --container-writable

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Phase      : 2 — Sampling hyperparameter search"
echo "Start time : $(date)"

# Find the most recently modified checkpoint written by phase 1 to /stratch.
# The 'find | xargs ls -t | head -1' pattern lists all .ckpt files sorted by
# modification time and takes the newest one. This is robust to the
# timestamped directory name that Hydra generates, since we don't know in
# advance what timestamp phase 1 ran at.
CHECKPOINT=$(find /stratch/arda.arslan/directo/outputs -name "*.ckpt" \
    2>/dev/null | xargs ls -t 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found under /stratch/arda.arslan/directo/outputs/"
    echo "Did Phase 1 complete successfully and write to /stratch?"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

conda run --no-capture-output -n directo \
    python main.py \
        +experiment=synthetic \
        dataset=synthetic \
        dataset.graph_type=er \
        dataset.acyclic=True \
        model.extra_features=magnetic-eigenvalues \
        general.num_sample_fold=5 \
        general.test_only=$CHECKPOINT \
        general.wandb=online \
        sample.search=all \
        "hydra.run.dir=/users/arda.arslan/allab/DIRECTO/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-er-dag-search"

echo "Phase 2 finished at: $(date)"
echo "Read search_hyperparameters.csv in the output directory, then update BEST_* in phase3_eval.sh"