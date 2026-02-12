#!/bin/bash

#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem 40G
#SBATCH -p public
#SBATCH -t 0-12:00:00   # time in d-hh:mm:ss
#SBATCH -A class_cse576fall2025
#SBATCH --job-name=gemma_all_text

# Activate virtual environment
source activate nlp_env

    # 
# Run benchmark with all three Gemma models on json and markdown (text-only)
# python benchmark_runner.py \
#     --models google/gemma-3-4b-it \
#     --representations json markdown image_arial image_times_new_roman image_red image_blue image_green \
#     --max-samples 10


python benchmark_runner.py \
    --models OpenGVLab/InternVL3_5-4B-hf \
    --representations json markdown image_arial image_times_new_roman image_red image_blue image_green \
    --max-samples 200