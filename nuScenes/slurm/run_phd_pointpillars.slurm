#! /bin/bash

#SBATCH -J phd
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/bailiping/Desktop/MOT/slurmoutfile/phd_pointpillars.out

source activate zhubinglabenv


python3 /home/bailiping/Desktop/MOT/run_phd_tracker.py --detection_file='/media/bailiping/'My Passport'/mmdetection3d/data/nuscenes/official_inference_result/pointpillars_val.json'
