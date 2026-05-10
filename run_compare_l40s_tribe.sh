#!/bin/bash
#SBATCH --job-name=compare_l40s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=compare_l40s_%j.log
#SBATCH --error=compare_l40s_%j.log

cd /home/users/shaunakm/Brain-LLM-Fine-Tuning

echo "=== Job started at $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== RAM: $(free -h | awk '/Mem/{print $2}') total ==="
echo ""

python compare_l40s_tribe_full.py \
    --output_dir ./brain-optimize-output-l40s \
    --save_dir   ./comparison-plots/l40s_full_tribe \
    --cache      /tmp/tribe-cache-${SLURM_JOB_ID} \
    --n_completions 2 \
    --max_new_tokens 160 \
    --temperature 0.7

echo ""
echo "=== Job completed at $(date) ==="
