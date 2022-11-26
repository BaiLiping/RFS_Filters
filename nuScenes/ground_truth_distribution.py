from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import BBox,predict_for_all,incorporate_track,visualize_duplicated_detection,iou3d,compute_duplicated_detection,fill_frame_result,gen_measurement_all, nms, nu_array2mot_bbox,create_scene_folder_name,create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
from datetime import datetime
from evaluate.util.utils import TrackingConfig, config_factory
import multiprocessing
import argparse
from tqdm import tqdm
import copy
import numpy as np
from utils.nuscenes_dataset import NuScenes
from matplotlib import pyplot as plt
from utils.plot_tracking_result import render_tracker_result

from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import Box, BBox, predict_for_all,incorporate_track,visualize_duplicated_detection,iou3d,compute_duplicated_detection,fill_frame_result,gen_measurement_all, nms, nu_array2mot_bbox,create_scene_folder_name,create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
from datetime import datetime
from evaluate.util.utils import TrackingConfig, config_factory
from evaluate.evaluate_tracking_result import TrackingEval
import multiprocessing
import argparse
from tqdm import tqdm
import copy
import numpy as np
from utils.nuscenes_dataset import NuScenes

import json
import math
import os
import os.path as osp
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from utils.utils import get_inference_colormap
from numpyencoder import NumpyEncoder


all='bicycle+motorcycle+trailer+truck+bus+pedestrian+car'
classifications_index = {'bicycle':0,'motorcycle':1,  'trailer':2, 'truck':3,'bus':4,'pedestrian':5,'car':6}
classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', default='v1.0-trainval', help='choose dataset version between [v1.0-trainval][v1.0-test][v1.0-mini]')
    parser.add_argument('--detection_file',default='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/detection_result/BEVfusion/val_results.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/bailiping/Desktop/MOT')
    parser.add_argument('--dataset_file', default='/media/bailiping/My Passport/mmdetection3d/data/nuscenes')
    parser.add_argument('--parallel_process', default=8)
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/bailiping/Desktop/experiment_result')
    parser.add_argument('--render_curves', default=False)
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default=True)
    parser.add_argument('--plot_result',default=True)
    parser.add_argument('--single_thread_debug',default=True)
    args = parser.parse_args()
    return args


def main(token, out_file_directory_for_this_experiment):
    args=parse_args()
    dataset_info_file='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/configs/dataset_info.json'
    
    if args.data_version =='v1.0-trainval':
        set_info='val'
    elif args.data_version == 'v1.0-mini':
        set_info='mini_val'
    elif args.data_version == 'v1.0-test':
        set_info='test'
    else:
        raise KeyError('wrong data version')
    # read ordered frame info
    with open(dataset_info_file, 'rb') as f:
        dataset_info=json.load(f)

    orderedframe=copy.deepcopy(dataset_info[set_info]['ordered_frame_info'])

    
    close_tracks_record={}
    close_tracks_record['results']={}
    close_terminal_tracks_record={}
    close_terminal_tracks_record['results']={}
    track_record={}
    initial_eucleadian_distance_record={}
    initial_x_distance_record={}
    initial_y_distance_record={}

    terminal_eucleadian_distance_record={}
    terminal_x_distance_record={}
    terminal_y_distance_record={}
    for classification in classifications:
        initial_eucleadian_distance_record[classification]=[]
        initial_x_distance_record[classification]=[]
        initial_y_distance_record[classification]=[]
        terminal_eucleadian_distance_record[classification]=[]
        terminal_x_distance_record[classification]=[]
        terminal_y_distance_record[classification]=[]
    for scene_idx in range(len(list(orderedframe.keys()))):
        if args.single_thread_debug==False:
            if scene_idx % args.parallel_process != token:
                continue
        scene_token = list(orderedframe.keys())[scene_idx]
        out_file_directory_for_this_scene=create_scene_folder_name(scene_token, out_file_directory_for_this_experiment)
        ordered_frames = orderedframe[scene_token]
        
        track_record[scene_token]={}
        id_list=[]
        for frame_idx, frame_token in enumerate(ordered_frames):
            close_tracks_record['results'][frame_token]=[]
            close_terminal_tracks_record['results'][frame_token]=[]
            ground_truth_bboxes_for_this_frame=dataset_info[set_info]['ground_truth_bboxes'][scene_token][str(frame_idx)]
            ground_truth_type_for_this_frame=dataset_info[set_info]['ground_truth_inst_types'][scene_token][str(frame_idx)]
            ground_truth_id_for_this_frame=dataset_info[set_info]['ground_truth_IDS'][scene_token][str(frame_idx)]
            ego_position_of_this_frame=dataset_info[set_info]['ego_position_info'][scene_token][str(frame_idx)]
            sensor_calibration_data_of_this_frame=dataset_info[set_info]['sensor_calibration_info'][scene_token][str(frame_idx)]
            
            for gt_idx, gt_id in enumerate(ground_truth_id_for_this_frame): 
                
                if gt_id not in id_list:
                    # initiate the record
                    instance_info={}
                    track_record[scene_token][gt_id]={}
                    id_list.append(gt_id)
                    translation=ground_truth_bboxes_for_this_frame[gt_idx][:3]
                    size=ground_truth_bboxes_for_this_frame[gt_idx][3:6]
                    rotation=ground_truth_bboxes_for_this_frame[gt_idx][6:]
                    tracking_name=ground_truth_type_for_this_frame[gt_idx]
                    track_record[scene_token][gt_id]['initial_frame']=frame_idx
                    track_record[scene_token][gt_id]['tracking_name']=tracking_name
                    track_record[scene_token][gt_id]['record']=[]
                    track_record[scene_token][gt_id]['frame_record']=[]
                    track_record[scene_token][gt_id]['frame_record'].append(frame_idx)
                    x_distance=ground_truth_bboxes_for_this_frame[gt_idx][0]-ego_position_of_this_frame[0]
                    y_distance=ground_truth_bboxes_for_this_frame[gt_idx][1]-ego_position_of_this_frame[1]
                    euclidean_distance=np.sqrt(np.power(ground_truth_bboxes_for_this_frame[gt_idx][0]-ego_position_of_this_frame[0],2)+np.power(ground_truth_bboxes_for_this_frame[gt_idx][1]-ego_position_of_this_frame[1],2))
                    instance_info['translation']=translation
                    instance_info['rotation']=rotation
                    instance_info['size']=size
                    instance_info['tracking_name']=tracking_name
                    instance_info['euclidean_distance']=euclidean_distance
                    instance_info['x_distance']=x_distance
                    instance_info['y_distance']=y_distance
                    instance_info['tracking_id']='0'
                    instance_info['tracking-score']=0
                    instance_info['velocity']=[0,0]
                    instance_info['sample_token']=frame_token
                    track_record[scene_token][gt_id]['record'].append(instance_info)
                else:

                    instance_info={}
                    track_record[scene_token][gt_id]['frame_record'].append(frame_idx)
                    translation=ground_truth_bboxes_for_this_frame[gt_idx][:3]
                    instance_info['translation']=translation
                    size=ground_truth_bboxes_for_this_frame[gt_idx][3:6]
                    rotation=ground_truth_bboxes_for_this_frame[gt_idx][6:]
                    instance_info['rotation']=rotation
                    instance_info['size']=size
                    instance_info['tracking_name']=ground_truth_type_for_this_frame[gt_idx]
                    x_distance=ground_truth_bboxes_for_this_frame[gt_idx][0]-ego_position_of_this_frame[0]
                    instance_info['x_distance']=x_distance
                    y_distance=ground_truth_bboxes_for_this_frame[gt_idx][1]-ego_position_of_this_frame[1]
                    instance_info['y_distance']=y_distance
                    euclidean_distance=np.power(ground_truth_bboxes_for_this_frame[gt_idx][0]-ego_position_of_this_frame[0],2)+np.power(ground_truth_bboxes_for_this_frame[gt_idx][1]-ego_position_of_this_frame[1],2)
                    instance_info['euclidean_distance']=euclidean_distance
                    instance_info['tracking_id']='0'
                    instance_info['tracking-score']=0
                    instance_info['velocity']=[0,0]
                    instance_info['sample_token']=frame_token
                    track_record[scene_token][gt_id]['record'].append(instance_info)
        
        for track_id in list( track_record[scene_token].keys()):
            track=track_record[scene_token][track_id]
            initial_frame=track['initial_frame']
            terminal_frame=track['frame_record'][-1]
            if initial_frame!=0:
                for classification in classifications:
                    if track['tracking_name']==classification:
                        if track['record'][0]['euclidean_distance']<60:
                            initial_eucleadian_distance_record[classification].append(track['record'][0]['euclidean_distance'])
                            initial_x_distance_record[classification].append(track['record'][0]['x_distance'])
                            initial_y_distance_record[classification].append(track['record'][0]['y_distance'])
                            if track['tracking_name'] in ['car', 'truck', 'trailer', 'bus']:
                                if track['record'][0]['euclidean_distance']<20:
                                    for instance in track['record']:
                                        close_tracks_record['results'][instance['sample_token']].append(instance)
            if terminal_frame!=len(ordered_frames):
                for classification in classifications:
                    if track['tracking_name']==classification:
                        if track['record'][-1]['euclidean_distance']<100:
                            terminal_eucleadian_distance_record[classification].append(track['record'][-1]['euclidean_distance'])
                            terminal_x_distance_record[classification].append(track['record'][-1]['x_distance'])
                            terminal_y_distance_record[classification].append(track['record'][-1]['y_distance'])
                            if track['record'][-1]['euclidean_distance']<20:
                                for instance in track['record']:
                                    close_terminal_tracks_record['results'][instance['sample_token']].append(instance)
        '''
        for frame_index, _ in enumerate(ordered_frames):
            for track_id in list(track_record[scene_token].keys()):
                track=track_record[scene_token][track_id]
                initial_frame=track['initial_frame']
                terminal_frame=track['frame_record'][-1]
                if initial_frame==frame_index:
                    target=track['record'][0]
                    target_box = Box(target['translation'],target['size'],Quaternion(target['rotation']))
                    target_box.translate(-np.array(ego_position_of_this_frame[:3]))
                    target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
                    center_bottom=target_box.center 
                    track_record[scene_token][track_id]['position_record']=[]
                    track_record[scene_token][track_id]['position_record'].append(center_bottom)
                else:
                    if frame_index>initial_frame and frame_index<=terminal_frame:
                        if frame_index in track['frame_record']:
                            position_idx=track['frame_record'].index(frame_index)
                            target=track_record[scene_token][track_id]['record'][position_idx]
                            #target_box = Box([target[0], target[1], target[2]],[target[3], target[4], target[5]],Quaternion([target[6], target[7], target[8], target[9]]))
                            target_box = Box(target['translation'],target['size'],Quaternion(target['rotation']))
                            target_box.translate(-np.array(ego_position_of_this_frame[:3]))
                            target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
                            center_bottom=target_box.center
                            track_record[scene_token][track_id]['position_record'].append(center_bottom) 
        '''
        
        print('done with scene {} process {}'.format(scene_idx, token))
    fig, axes=plt.subplots(7, figsize=(16, 25))
    fig.suptitle('Validation Set GT Track Initial Euclidean Distance From Ego', fontsize='25')
    for idx, classification in enumerate(classifications):
        ax=axes[idx]
        ax.hist(initial_eucleadian_distance_record[classification],  bins=30)
        ax.set_title(classification)
        ax.set(ylabel='Number of Tracks')
    axes[-1].set(xlabel='Euclidean Distance to Ego', ylabel='Number of Tracks')
    plt.savefig('/home/bailiping/Desktop/val_distribution.png')
    plt.close()

    fig, axes=plt.subplots(7, figsize=(16, 25))
    fig.suptitle('Validation Set GT Track Terminal Euclidean Distance From Ego', fontsize='25')
    for idx, classification in enumerate(classifications):
        ax=axes[idx]
        ax.hist(terminal_eucleadian_distance_record[classification],  bins=30)
        ax.set_title(classification)
        ax.set(ylabel='Number of Tracks')
    axes[-1].set(xlabel='Euclidean Distance to Ego', ylabel='Number of Tracks')
    plt.savefig('/home/bailiping/Desktop/val_terminal_distribution.png')
    plt.close()

    
    with open('/home/bailiping/Desktop/val_gt_track_record.json', 'w') as f:
        json.dump(track_record, f, cls=NumpyEncoder)
    with open('/home/bailiping/Desktop/close_track_record.json', 'w') as f:
        json.dump(close_tracks_record, f, cls=NumpyEncoder)
    with open('/home/bailiping/Desktop/close_terminal_track_record.json', 'w') as f:
        json.dump(close_terminal_tracks_record, f, cls=NumpyEncoder)

    
if __name__ == '__main__':
    # read out dataset version
    arguments = parse_args()
    dataset_info_file=arguments.programme_file+'/configs/dataset_info.json'
    config=arguments.programme_file+'/configs/pmbmgnn_parameters.json'
    if arguments.data_version =='v1.0-trainval':
        set_info='val'
    elif arguments.data_version == 'v1.0-mini':
        set_info='mini_val'
    elif arguments.data_version == 'v1.0-test':
        set_info='test'
    else:
        raise KeyError('wrong data version')
    # create a folder for this experiment
    now = datetime.now()
    formatedtime = now.strftime("%Y-%m-%d-%H-%M-%S")
    out_file_directory_for_this_experiment = create_experiment_folder(arguments.result_file, formatedtime, set_info)
    inputarguments=[]
    for token in range(arguments.parallel_process):
        inputarguments.append((token,out_file_directory_for_this_experiment))
    # start processing information
    if arguments.single_thread_debug:
        main(1,out_file_directory_for_this_experiment)
    else:
        pool = multiprocessing.Pool(processes=arguments.parallel_process)
        pool.starmap(main,inputarguments)
        pool.close()
    