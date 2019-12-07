#!/bin/bash

#SBATCH --job-name=cnn-training-test-on-gpu
#SBATCH --error=gpu-test.err
#SBATCH --output=gpu-test.out
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4

python3 -W ignore DeepGAN.py