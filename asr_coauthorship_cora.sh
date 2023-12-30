#!/bin/bash
#SBATCH -A pi_zy
#SBATCH -p gpu2Q 
#SBATCH -q gpuq
#SBATCH --gres=gpu:1
python -u asr_coauthorship_cora.py  