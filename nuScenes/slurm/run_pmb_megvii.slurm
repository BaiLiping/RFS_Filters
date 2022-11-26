#! /bin/bash

#SBATCH -J pmb
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/bailiping/Desktop/MOT/slurmoutfile/pmb_megvii.out

source activate zhubinglabenv


python3 /home/bailiping/Desktop/MOT/run_pmb_tracker.py --detection_file='/media/bailiping/'My Passport'/mmdetection3d/data/nuscenes/official_inference_result/megvii_val.json'
