#! /bin/bash

#SBATCH -J pmbmgnn
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/bailiping/Desktop/MOT/slurmoutfile/pmbmgnn_megvii.out

source activate zhubinglabenv


python3 /home/bailiping/Desktop/MOT/run_pmbmgnn_tracker.py --detection_file='/home/bailiping/Desktop/mmdetection3d/data/nuscenes/official_inference_result/megvii_val.json'
