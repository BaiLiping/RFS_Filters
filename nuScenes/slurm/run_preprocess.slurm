#! /bin/bash

#SBATCH -J preprocess
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/bailiping/Desktop/MOT/slurmoutfile/preprocess.out

source activate zhubinglabenv


python3 /home/bailiping/Desktop/MOT/utils/preprocessing.py