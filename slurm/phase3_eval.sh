#!/bin/bash
#SBATCH --job-name=directo_eval_er_dag
#SBATCH --output=/users/arda.arslan/allab/DIRECTO/slurm/log/phase3_eval/eval_%j.out
#SBATCH --error=/users/arda.arslan/allab/DIRECTO/slurm/log/phase3_eval/eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image ghcr.io\#arslanardavic/directo:latest
#SBATCH --container-mounts=/stratch:/stratch
#SBATCH --container-writable

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Phase      : 3 — Final evaluation"
echo "Start time : $(date)"

CHECKPOINT=$(find /stratch/arda.arslan/directo/outputs -name "*.ckpt" \
    2>/dev/null | xargs ls -t 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found under /stratch/arda.arslan/directo/outputs/"
    exit 1
fi

# ── SET THESE AFTER READING search_hyperparameters.csv FROM PHASE 2 ──────────
# Phase 3 cannot be fully automated because it requires a human decision:
# you need to read the CSV from phase 2 and choose the best combination.
# Update these three values before re-submitting this script, or submit it
# manually as a standalone job (sbatch slurm/phase3_eval.sh) after phase 2.
# The values below are a reasonable starting point based on the paper's
# Visual Genome results, but ER-DAG may prefer different settings.
BEST_ETA=10
BEST_OMEGA=0.1
BEST_DISTORTION=polydec

echo "Using checkpoint    : $CHECKPOINT"
echo "eta                 : $BEST_ETA"
echo "omega               : $BEST_OMEGA"
echo "time_distortion     : $BEST_DISTORTION"

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
        sample.eta=$BEST_ETA \
        sample.omega=$BEST_OMEGA \
        sample.time_distortion=$BEST_DISTORTION \
        "hydra.run.dir=/stratch/arda.arslan/directo/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-er-dag-eval"

echo "Phase 3 finished at: $(date)"
echo "Final metrics written to test_epoch*_res_*.txt — compare against Table 1."