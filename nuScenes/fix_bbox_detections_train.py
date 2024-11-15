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
    parser.add_argument('--detection_file',default='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/detection_result/BEVfusion/val_results.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/bailiping/Desktop/MOT')
    parser.add_argument('--dataset_file', default='/media/bailiping/My Passport/mmdetection3d/data/nuscenes')
    parser.add_argument('--parallel_process', default=1)
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/bailiping/Desktop/experiment_result')
    parser.add_argument('--render_curves', default=False)
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default=True)
    parser.add_argument('--plot_result',default=False)
    parser.add_argument('--single_thread_debug',default=False)
    args = parser.parse_args()
    return args


def main(token, out_file_directory_for_this_experiment):
    args=parse_args()
    if args.plot_result:
        nuscenes_data = NuScenes(version = args.data_version, dataroot=args.dataset_file, verbose=False)
    dataset_info_file='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/configs/dataset_info.json'
    config='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/configs/pmbmgnn_parameters.json'
    
    if args.data_version =='v1.0-trainval':
        set_info='train'
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
    ground_truth_type=dataset_info[set_info]['ground_truth_inst_types']
    ground_truth_id=dataset_info[set_info]['ground_truth_IDS']
    ground_truth_bboxes=dataset_info[set_info]['ground_truth_bboxes']

    with open(args.detection_file, 'rb') as f:
        inference_meta= json.load(f)
    inference=inference_meta['results']

    # read parameters
    with open(config, 'r') as f:
        parameters=json.load(f)
    
    filters=[]
    birth_rate_list=[]
    z_correction_list=[]
    nms_list=[]

    for classification in classifications:
        # readout parameters
        Z_correction, birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classification, parameters)
        # adjust detection_scrore_thr for pointpillars
        # this step is necessary because the pointpillar detections for cars and pedestrian would generate excessive number of bounding boxes and result in unnecessary compuations.
        # generate filter model based on the classification
        filter_model = pmbmgnn_ulti.gen_filter_model(clutter_rate,P_s,P_d, classification, extraction_thr, ber_thr, poi_thr, eB_thr,bernoulli_gating, use_ds_as_pd, P_init)
        pmbm_filter = pmbmgnn_tracker.PMBMGNN_Filter(filter_model)
        filters.append(pmbm_filter)
        birth_rate_list.append(birth_rate)
        z_correction_list.append(Z_correction)
        nms_list.append(nms_score)
     
    
    for scene_idx in range(len(list(orderedframe.keys()))):
        if args.single_thread_debug==False:
            if scene_idx % args.parallel_process != token:
                continue
        scene_token = list(orderedframe.keys())[scene_idx]
        out_file_directory_for_this_scene=create_scene_folder_name(scene_token, out_file_directory_for_this_experiment)
        ordered_frames = orderedframe[scene_token]
        ego_info = egoposition[scene_token]
        filter_pruned_dict={}
        filter_predicted_dict={}
        for frame_idx, frame_token in enumerate(ordered_frames):
            ground_truth_bboxes_for_this_frame=ground_truth_bboxes[scene_token][str(frame_idx)]
            ground_truth_type_for_this_frame=ground_truth_type[scene_token][str(frame_idx)]
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
            frame_result=[]

                        
            for classification_index, classification in enumerate(classifications):
                # fix the z-position issue
                Z_k_classification=[]
                for z in Z_k_original:
                    if z['detection_name']==classification:
                        z['translation'][2]=z['translation'][2]-z_correction_list[classification_index]
                        Z_k_classification.append(z)
            
                # remove the overlapping bboxes according to nms score
                result_indexes = nms(Z_k_classification, threshold=nms_score)
                Z_k=[]
                for idx in result_indexes:
                    Z_k.append(Z_k_classification[idx])

                # combine the bounding boxes of all the classifications
                frame_result=np.hstack((frame_result,Z_k))            
             
            
            # update time
            pre_timestamp = cur_timestamp
            previous_frame_token_exist=True
            previous_frame_token=frame_token
            
            # save the result for this scene
            with open(out_file_directory_for_this_experiment+'/{}.json'.format(frame_token), 'w') as f:
                json.dump(frame_result, f, cls=NumpyEncoder)

        print('done with scene {} process {}'.format(scene_idx, token))

if __name__ == '__main__':
    # read out dataset version
    arguments = parse_args()
    dataset_info_file='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/configs/dataset_info.json'
    config='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/configs/pmbmgnn_parameters.json'
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
        
    
    with open(dataset_info_file, 'r') as f:
        dataset_info=json.load(f)

    orderedframe=dataset_info[set_info]['ordered_frame_info']
    # aggregate results for all classifications
    submission = {}
    submission['results']={}
    submission['meta']={}
    for scene_idx in range(len(list(orderedframe.keys()))):
        scene_token = list(orderedframe.keys())[scene_idx]
        ordered_frames = orderedframe[scene_token]

        for frame_token in ordered_frames:
            submission['results'][frame_token]=[]
            with open(out_file_directory_for_this_experiment+'/{}.json'.format(frame_token), 'r') as f:
                frame_submission=json.load(f)
            for bbox_info in frame_submission:
                submission['results'][frame_token].append(bbox_info)
    with open('/home/bailiping/Desktop/training_restuls_fixed.json', 'w') as f:
        json.dump(submission, f, cls=NumpyEncoder)