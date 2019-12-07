#!/bin/bash

#SBATCH --mem=100GB
#SBATCH --account=IscrC_TBD
#SBATCH --time=8:00:00
#SBATCH --error=gpu-gan.err
#SBATCH --output=gpu-gan.out
#SBATCH --partition=gll_usr_gpuprod
#SBATCH --gres=gpu:kepler:2

python3 -W ignore DeepGAN.py