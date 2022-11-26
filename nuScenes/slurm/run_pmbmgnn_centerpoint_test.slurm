#! /bin/bash

#SBATCH -J pmbmgnn
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/bailiping/Desktop/MOT/slurmoutfile/pmbmgnn_centerpoint_test.out

source activate zhubinglabenv


python3 /home/bailiping/Desktop/MOT/run_pmbmgnn_tracker.py --data_version='v1.0-test' --detection_file='/media/bailiping/'My Passport'/mmdetection3d/data/nuscenes/official_inference_result/centerpoint_test.json'
