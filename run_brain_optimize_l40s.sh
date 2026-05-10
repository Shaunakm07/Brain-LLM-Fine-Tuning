#!/bin/bash
#SBATCH --job-name=brain_opt_l40s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=brain_optimize_l40s_%j.log
#SBATCH --error=brain_optimize_l40s_%j.log

cd /home/users/shaunakm/Brain-LLM-Fine-Tuning

echo "=== Job started at $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== RAM: $(free -h | awk '/Mem/{print $2}') total ==="

python brain_optimize_l40s.py \
    --region broca \
    --model Qwen/Qwen2.5-3B-Instruct \
    --n_steps 200 \
    --n_completions 4 \
    --max_new_tokens 200 \
    --t_start 1.0 \
    --t_end 0.6 \
    --kl_coef 0.1 \
    --lr 5e-5 \
    --warmup_steps 5 \
    --output_dir ./brain-optimize-output-l40s \
    --cache /tmp/tribe-cache-${SLURM_JOB_ID}

echo ""
echo "=== Job completed at $(date) ==="
