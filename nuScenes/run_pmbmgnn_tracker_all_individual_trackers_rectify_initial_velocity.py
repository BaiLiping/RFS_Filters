from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import BBox, predict_for_all,incorporate_track,visualize_duplicated_detection,iou3d,compute_duplicated_detection,fill_frame_result,gen_measurement_all, nms, nu_array2mot_bbox,create_scene_folder_name,create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
from trackers.PMBMGNN import PMBMGNN_Filter_Point_Target_single_class as pmbmgnn_tracker
from trackers.PMBMGNN import util as pmbmgnn_ulti
from datetime import datetime
from evaluate.util.utils import TrackingConfig, config_factory
from evaluate.evaluate_tracking_result import TrackingEval
import multiprocessing
import argparse
from tqdm import tqdm
import copy
import numpy as np
from utils.nuscenes_dataset import NuScenes
from matplotlib import pyplot as plt


all='bicycle+motorcycle+trailer+truck+bus+pedestrian+car'
classifications_index = {'bicycle':0,'motorcycle':1,  'trailer':2, 'truck':3,'bus':4,'pedestrian':5,'car':6}
classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', default='v1.0-trainval', help='choose dataset version between [v1.0-trainval][v1.0-test][v1.0-mini]')
    parser.add_argument('--detection_file',default='/home/blp/Desktop/val_result_with_classification.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/blp/Desktop/MOT')
    parser.add_argument('--dataset_file', default='/home/blp/Desktop/mmdetection3d/data/nuscenes')
    parser.add_argument('--parallel_process', default=7)
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/blp/Desktop')
    parser.add_argument('--render_curves', default=False)
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default=True)
    parser.add_argument('--plot_result',default=False)
    parser.add_argument('--single_thread_debug',default=False)
    args = parser.parse_args()
    return args


def main():
    args=parse_args()
    dataset_info_file=args.programme_file+'/configs/dataset_info.json'
    config=args.programme_file+'/configs/pmbmgnn_parameters.json'
    
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

    orderedframe=dataset_info[set_info]['ordered_frame_info']
    # time stamp info
    timestamps=dataset_info[set_info]['time_stamp_info']
    # ego info
    egoposition=dataset_info[set_info]['ego_position_info']

    with open(args.detection_file, 'rb') as f:
        inference_meta= json.load(f)
    inference=inference_meta['results']
    inference_rectified_initial_velocity=copy.deepcopy(inference_meta)

    # read parameters
    with open(config, 'r') as f:
        parameters=json.load(f)
    
    filters=[]
    birth_rate_list=[]

    for classification in classifications:
        # readout parameters
        birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classification, parameters)
        detection_score_thr=0
        P_init=30
        # adjust detection_scrore_thr for pointpillars
        # this step is necessary because the pointpillar detections for cars and pedestrian would generate excessive number of bounding boxes and result in unnecessary compuations.
        # generate filter model based on the classification
        filter_model = pmbmgnn_ulti.gen_filter_model(clutter_rate,P_s,P_d, classification, extraction_thr, ber_thr, poi_thr, eB_thr,bernoulli_gating, use_ds_as_pd, P_init)
        pmbm_filter = pmbmgnn_tracker.PMBMGNN_Filter(filter_model)
        filters.append(pmbm_filter)
        birth_rate_list.append(birth_rate)
     
    
    for scene_idx in range(len(list(orderedframe.keys()))):
        scene_token = list(orderedframe.keys())[scene_idx]
        ordered_frames = orderedframe[scene_token]
        ego_info = egoposition[scene_token]
        filter_pruned_dict={}
        filter_predicted_dict={}
        results={}
        for frame_idx, frame_token in enumerate(ordered_frames):
            if frame_idx<2:
                if frame_idx == 0:
                    pre_timestamp = timestamps[scene_token][frame_idx]
                    previous_frame_token_exist=False
                cur_timestamp = timestamps[scene_token][frame_idx]
                time_lag = (cur_timestamp - pre_timestamp)/1e6
                giou_gating = -0.5
                # get measurements at global coordinate
                if frame_token in inference.keys():
                    estimated_bboxes_at_current_frame = inference[frame_token]
                else:
                    print('lacking inference file')
                    break
                
                Z_k_original=gen_measurement_all(estimated_bboxes_at_current_frame)
                results[frame_token]=[]
              
                
                for classification_index, classification in enumerate(classifications):
                    if True:
                        Z_k=[]
                        # There is no need for the initial frame to compute the duplicated detection
                        if len(Z_k_original)>0:
                            for z in Z_k_original:
                                if z['detection_name']==classification:
                                    Z_k.append(z)
                   
                    # prediction
                    if frame_idx == 0:
                        filter_predicted = filters[classification_index].predict_initial_step(Z_k, birth_rate_list[classification_index])
                        filter_predicted_dict[classification]=filter_predicted
                    else:
                        filter_predicted = filters[classification_index].predict(ego_info[str(frame_idx)],time_lag,filter_pruned_dict[classification], Z_k, birth_rate_list[classification_index])
                        filter_predicted_dict[classification]=filter_predicted
                    # update
                    filter_updated = filters[classification_index].update(Z_k, filter_predicted, confidence_score,giou_gating)
    
                    # state extraction
                    estimatedStates_for_this_classification = filters[classification_index].extractStates(filter_updated)
                    results[frame_token]=fill_frame_result(results[frame_token],estimatedStates_for_this_classification,frame_token)
        
                    # pruning
                    filter_pruned = filters[classification_index].prune(filter_updated)
                    filter_pruned_dict[classification]=filter_pruned
                
                # update time
                pre_timestamp = cur_timestamp
            
            if frame_idx==1:
                inference_rectified_initial_velocity['results'][ordered_frames[0]]=[]
                for frame0_bbox in results[ordered_frames[0]]:
                    instance_info = copy.deepcopy(frame0_bbox)
                    for frame1_bbox in results[ordered_frames[1]]:
                        if frame1_bbox['tracking_id']==frame0_bbox['tracking_id']:
                            print('original velocity {}'.format(instance_info['velocity']))
                            velocity_x=frame1_bbox['velocity'][0]-frame0_bbox['velocity'][0]
                            velocity_y=frame1_bbox['velocity'][1]-frame0_bbox['velocity'][1]
                            instance_info['velocity'][0]=velocity_x
                            instance_info['velocity'][1]=velocity_y
                            print('rectified velocity {}'.format(instance_info['velocity']))
                    instance_info['detection_name']=instance_info['tracking_name']
                    instance_info['detection_score']=instance_info['tracking_score']
                    inference_rectified_initial_velocity['results'][ordered_frames[0]].append(instance_info)
    
    with open('/home/blp/Desktop/val_results_initial_velocity.json', 'w') as f:
        json.dump(inference_rectified_initial_velocity, f, cls=NumpyEncoder)
    
if __name__ == '__main__':
    main()  