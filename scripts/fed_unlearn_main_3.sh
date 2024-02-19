#!/bin/bash
#SBATCH --job-name=federaser
#SBATCH --nodes=1
#SBATCH --mem=50000
#SBATCH -o tensor_out_3.txt
#SBATCH -e tensor_error_3.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#module load gnu7/7.2.0
module load cuda/10.0.130
module load anaconda/3.6
#module load mvapich2
#module load pmix/1.2.3

source activate federaser

srun python ../Fed_Unlearn_main_3.py

