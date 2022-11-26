import enum
import os
import sys
import shutil
import argparse
import numpy as np
import pickle
import json
from matplotlib import pyplot as plt
import time
from numpyencoder import NumpyEncoder

from numpy.random.mtrand import sample
from utils.nuscenes_dataset import NuScenes
from utils.utils import  create_scene_folder,gen_ordered_frames,generate_inference_visualization
from utils.box_util import box3d_iou
import copy
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

from trackers.PMBM import PMBM_Filter_Point_Target as pmbm_tracker
from trackers.PMBM import util as pmbm_ulti
import codecs
from datetime import datetime
from trackers.JPDA.jpda_tracker import run_jpda_tracker
'''
from trackers.GM_PHD import GM_PHD_Filter_Point_Target as gmphd_tracker
from trackers.GM_PHD import util as gmphd_util
from trackers.GM_CPHD import GM_CPHD_Filter_Point_Target as gmcphd_tracker
from trackers.GM_CPHD import util as gmcphd_util
from trackers.JPDA import util as jpda_util
from trackers.PDA import util as pda_util
'''
'''
lidar_3d_object_detection_inference_result_in_nuscenes_format_file='/media/bailiping/'My Passport'/mmdetection3d/data/nuscenes/official_inference_result/simpletrack_2hz_results_with_detection.json'
root_directory_for_dataset='/media/bailiping/My Passport/mmdetection3d/data/nuscenes'
dataset_version='v1.0-test'
'''
lidar_3d_object_detection_inference_result_in_nuscenes_format_file='/media/bailiping/'My Passport'/mmdetection3d/data/nuscenes/detection_result/BEV_fusion/val_results.json'
root_directory_for_dataset='/media/bailiping/My Passport/mmdetection3d/data/nuscenes'
dataset_version='v1.0-mini'

now=datetime.now()
formatedtime=now.strftime("%Y-%m-%d-%H-%M")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar_3d_object_detection_inference_result_in_nuscenes_format_file', default= lidar_3d_object_detection_inference_result_in_nuscenes_format_file, help='the path towards the inference_result_in_nuscenes_format_file_path')
    parser.add_argument('--type_of_tracker', default= 'pmbm_tracker', help='the tracker we want to run with estimated bboxes outputed from Lidar 3d object detector')
    parser.add_argument('--number_of_experiments', default= 1, help='the number of experiments')
    parser.add_argument('--dataset_version', default=dataset_version, help='v1.0-mini, v1.0-trainval or v1.0-test')
    parser.add_argument('--p_D', default=0.9, help='probability of detection')
    parser.add_argument('--p_S', default=0.9, help='probability of survival')
    parser.add_argument('--average_number_of_clutter_per_frame', default=5, help='the average number of clutter per frame estimation, this is used for the possion distribution assumed by the tracker')
    parser.add_argument('--bayesian_filter_config', default='Linear Kalman Filter')
    parser.add_argument('--motion_model_type', default='constant velocity')
    parser.add_argument('--root_directory_for_dataset', default=root_directory_for_dataset, help='root directory for the entire dataset, including mini, trainval and test')
    parser.add_argument('--bird_eye_view_with_map', default=True, help='generate bird eye view with cropped map, otherwise it would just be a blank canvas')
    parser.add_argument('--distance', default='Euclidean distance', help='the method for distance computation')
    parser.add_argument('--time', default=formatedtime,help='log the result into a folder named after current time')
    parser.add_argument('--detection_score_thr', default=0,help='the threshold for bbox detection score')
    parser.add_argument('--generate_visualization', default=True)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # read the nuscenes data
    nuscenes_data = NuScenes(version = args.dataset_version, dataroot=args.root_directory_for_dataset, verbose=False)
    with open(args.lidar_3d_object_detection_inference_result_in_nuscenes_format_file, 'rb') as f:
        estimated_bboxes_data_over_all_frames=json.load(f)
        #estimated_bboxes_data_over_all_frames_meta= json.load(f)
    #estimated_bboxes_data_over_all_frames=estimated_bboxes_data_over_all_frames_meta['results']
    
    # read out scenes of this dataset
    scenes=nuscenes_data.scene
    # read out frames of this dataset
    frames=nuscenes_data.sample
    #create a experiment folder based on the time
    #out_file_directory_for_this_experiment='/media/bailiping/'My Passport'/mmdetection3d/data/nuscenes/experiment_result/2022-01-15-15-07_birth0.0001_pd0.5_ps0.5'
    out_file_directory_for_this_experiment=root_directory_for_dataset
    # save the result for this classification
    with open('/media/bailiping/'My Passport'/mmdetection3d/data/nuscenes/detection_result/BEV_fusion/val_results.json', 'r') as f:
        result=json.load(f)

    for scene in scenes:
        out_file_directory_for_this_scene=create_scene_folder(scene, out_file_directory_for_this_experiment)
        ordered_frames=gen_ordered_frames(scene,frames)
        for frame_idx, frame in enumerate(ordered_frames): # Here we execute processing for each scan time(frame)          
            if frame['token'] in estimated_bboxes_data_over_all_frames['results'].keys():
                rectified_result=[]
                for bbox in result['results'][frame['token']]:
                    if bbox['detection_name']!='pedestrian':
                        rectified_result.append(bbox)
                    else:
                        if bbox['detection_score']>0.1:
                            rectified_result.append(bbox)

                nuscenes_data.render_tracker_result(args.detection_score_thr,rectified_result,rectified_result,frame['token'],nsweeps=1,out_path=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx),bird_eye_view_with_map=args.bird_eye_view_with_map,verbose=False)
                #generate_inference_visualization(nuscenes_data, estimated_bboxes_data_over_all_frames['results'], nsweeps = 1, root_directory_for_out_path=out_file_directory_for_this_scene)
                directory=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx)
                print("generating {}".format(directory))        
                plt.close()
    
if __name__ == '__main__':
    main()