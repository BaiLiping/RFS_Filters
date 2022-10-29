#! /bin/bash

#SBATCH -J pmb
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/blp/Desktop/MOT/slurmoutfile/pmb_pointpillars.out

source activate zhubinglabenv


python3 /home/blp/Desktop/MOT/run_pmb_tracker.py --detection_file='/home/blp/Desktop/mmdetection3d/data/nuscenes/official_inference_result/pointpillars_val.json'
