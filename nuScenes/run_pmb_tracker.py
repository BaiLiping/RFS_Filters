import enum
import os
import sys
import shutil
import argparse
import numpy as np
import json
from matplotlib import pyplot as plt
import time
from numpyencoder import NumpyEncoder

from numpy.random.mtrand import sample
from nuscenes import NuScenes
from utils.utils import nms,readout_parameters, create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file
import copy
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

from trackers.PMB import PMB_filter as pmb_tracker
from trackers.PMB import util as pmb_ulti
from datetime import datetime

from evaluate.evaluate_tracking_result import TrackingEval
from evaluate.util.utils import TrackingConfig, config_factory
import multiprocessing
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', default='v1.0-trainval', help='choose dataset version between [v1.0-trainval][v1.0-test][v1.0-mini]')
    parser.add_argument('--detection_file',default='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/detection_result/BEVfusion/val_results.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/bailiping/Desktop/MOT')
    parser.add_argument('--dataset_file', default='/media/bailiping/My Passport/mmdetection3d/data/nuscenes')
    parser.add_argument('--parallel_process', default=8)
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/bailiping/Desktop/experiment_result')
    parser.add_argument('--render_curves', default='False')
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default='True')

    args = parser.parse_args()
    return args

def main(classification,out_file_directory_for_this_experiment):
    args=parse_args()
    dataset_info_file='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/configs/dataset_info.json'
    config=args.programme_file+'/configs/pmb_parameters.json'
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

    # read parameters
    with open(config, 'r') as f:
        parameters=json.load(f)
    # initiate submission file
    classification_submission = initiate_classification_submission_file(classification)
    # readout parameters
    z_correction, birth_rate, P_s,P_d, use_ds_as_pd, clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classification, parameters)
    
    # adjust detection_scrore_thr for pointpillars
    # this step is necessary because the pointpillar detections for cars and pedestrian would generate excessive number of bounding boxes and result in unnecessary compuations.

    
    r_threshold =0
    #extraction_thr=0.5
    # generate filter model based on the classification
    filter_model = pmb_ulti.gen_filter_model(classification, clutter_rate, P_d, P_s,poi_thr)
 
    # generate filter model based on the classification
    for scene_idx in range(len(list(orderedframe.keys()))):
        scene_token = list(orderedframe.keys())[scene_idx]
        ordered_frames = orderedframe[scene_token]
        ego_info = egoposition[scene_token]
        # generate filter based on the filter model
        pmb_filter = pmb_tracker.PMB_Filter(filter_model, 'constant velocity', 'Linear Kalman Filter', classification)
        Z_list = []
        id_max=0
        initialize=False

        # initiate filter_pruned
        filter_pruned = {}
        filter_pruned['meanPois'] = []
        for frame_idx, frame_token in enumerate(ordered_frames):
            if frame_idx == 0:
                pre_timestamp = timestamps[scene_token][frame_idx]
            cur_timestamp = timestamps[scene_token][frame_idx]
            time_lag = (cur_timestamp - pre_timestamp)/1e6
            giou_gating = -0.5
            # get measurements at global coordinate
            if frame_token in inference.keys():
                estimated_bboxes_at_current_frame = inference[frame_token]
            else:
                print('lacking inference file')
                break
            classification_submission['results'][frame_token] = []
            Z_k_all = gen_measurement_of_this_class(detection_score_thr, estimated_bboxes_at_current_frame, classification)
            # preprocessing of input. Remove bboxes that overlaps
            result_indexes = nms(Z_k_all, threshold=nms_score)
            Z_k=[]
            for idx in result_indexes:
                Z_k.append(Z_k_all[idx])

            # prediction
            if initialize==False:  # For the fisrt frame, there are only new birth targets rather than surviving targets thus we call seperate function.
                filter_predicted = pmb_filter.predict_for_initial_step(Z_k, birth_rate)
                initialize=True
                filter_updated, id_max = pmb_filter.update(Z_k, filter_predicted, Z_list,id_max,confidence_score,bernoulli_gating) #Eq. 20 of [2]
                # loopy belief propogation
                pupd,pnew = pmb_filter.loopy_belief_propogation(Z_k, filter_updated)
                r,filter_updated, Z_list= pmb_filter.tomb(pupd,pnew,filter_updated,r_threshold)
                estimatedStates_for_this_classification=[]
                for i in range(len(r)):
                    if r[i]>extraction_thr:
                        estimatedStates_for_this_classification.append(copy.deepcopy(Z_list[i]))                                      
                # fill the submission file for this frame
                if len(estimatedStates_for_this_classification) != 0:
                    for idx in range(len(estimatedStates_for_this_classification)):
                        estimate=copy.deepcopy(estimatedStates_for_this_classification[idx])
                        instance_info = {}
                        instance_info['sample_token'] = copy.deepcopy(frame_token)
                        instance_info['translation'] = copy.deepcopy(estimate['translation'])
                        instance_info['size'] = copy.deepcopy(estimate['size'])
                        instance_info['rotation'] = copy.deepcopy(estimate['rotation'])
                        instance_info['velocity'] = copy.deepcopy(estimate['velocity'])
                        instance_info['tracking_id'] = copy.deepcopy(classification+'_'+str(estimate['id']))
                        instance_info['tracking_name'] = copy.deepcopy(classification)
                        instance_info['tracking_score']=copy.deepcopy(estimate['detection_score'])
                        #instance_info['tracking_score'] = estimatedStates_for_this_classification['w'][idx]
                        classification_submission['results'][frame_token].append(instance_info)
            else:
                filter_predicted, Z_list= pmb_filter.predict(Z_k,filter_updated, Z_list, birth_rate,time_lag)  
                filter_updated, id_max = pmb_filter.update(Z_k, filter_predicted, Z_list,id_max,confidence_score,bernoulli_gating)
                # loopy belief propogation
                pupd,pnew = pmb_filter.loopy_belief_propogation(Z_k, filter_updated)
                r,filter_updated, Z_list= pmb_filter.tomb(pupd,pnew,filter_updated,r_threshold)
                estimatedStates_for_this_classification=[]
                for i in range(len(r)):
                    if r[i]>extraction_thr:
                        estimatedStates_for_this_classification.append(copy.deepcopy(Z_list[i]))                                       
                # fill the submission file for this frame
                if len(estimatedStates_for_this_classification) != 0:
                    for idx in range(len(estimatedStates_for_this_classification)):
                        estimate=copy.deepcopy(estimatedStates_for_this_classification[idx])
                        instance_info = {}
                        instance_info['sample_token'] = copy.deepcopy(frame_token)
                        instance_info['translation'] = copy.deepcopy(estimate['translation'])
                        instance_info['size'] = copy.deepcopy(estimate['size'])
                        instance_info['rotation'] = copy.deepcopy(estimate['rotation'])
                        instance_info['velocity'] = copy.deepcopy(estimate['velocity'])
                        instance_info['tracking_id'] = copy.deepcopy(classification+'_'+str(estimate['id']))
                        instance_info['tracking_name'] = copy.deepcopy(classification)
                        instance_info['tracking_score']=copy.deepcopy(estimate['detection_score'])
                        classification_submission['results'][frame_token].append(instance_info)
                #else:
                #    print('no estimation')
            #print(len(classification_submission['results'][frame_token]))
            pre_timestamp = cur_timestamp
        print('done with {} scene {}'.format(classification,scene_idx))
    # save the result for this classification
    with open(out_file_directory_for_this_experiment+'/{}_submission.json'.format(classification), 'w') as f:
        json.dump(classification_submission, f, indent=4, cls=NumpyEncoder)
    print('{} is done'.format(classification))
        
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
    classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']
    #classifications = ['motorcycle']
    # create a folder for this experiment
    now = datetime.now()
    formatedtime = now.strftime("%Y-%m-%d-%H-%M-%S")
    out_file_directory_for_this_experiment = create_experiment_folder(arguments.result_file, formatedtime, set_info)
    inputarguments=[]
    for classification in classifications:
        inputarguments.append((classification,out_file_directory_for_this_experiment))
    # start processing information
    pool = multiprocessing.Pool(processes=arguments.parallel_process)
    pool.starmap(main,inputarguments)
    pool.close()
    
    with open(dataset_info_file, 'r') as f:
        dataset_info=json.load(f)

    orderedframe=dataset_info[set_info]['ordered_frame_info']
    # aggregate results for all classifications
    submission = initiate_submission_file(orderedframe)
    for classification in classifications:
        with open(out_file_directory_for_this_experiment+'/{}_submission.json'.format(classification), 'r') as f:
            submission_for_this_class = json.load(f)
            result_of_this_class = submission_for_this_class['results']
            for frame_token in result_of_this_class:
                for bbox_info in result_of_this_class[frame_token]:
                    submission['results'][frame_token].append(bbox_info)
    with open(out_file_directory_for_this_experiment+'/val_submission.json', 'w') as f:
        json.dump(submission, f, cls=NumpyEncoder)

    if arguments.data_version == 'v1.0-trainval' or arguments.data_version == 'v1.0-mini':
        # evaluate result if it is validation set
        result_path_ = os.path.expanduser(out_file_directory_for_this_experiment+'/val_submission.json')
        output_dir_ = os.path.expanduser(out_file_directory_for_this_experiment+'/nuscenes-metrics')
        eval_set_ = set_info
        dataroot_ = arguments.dataset_file
        version_ = arguments.data_version
        config_path = arguments.config_path
        render_curves_ = arguments.render_curves
        verbose_ = arguments.verbose
        render_classes_ = arguments.render_classes
    
        if config_path == '':
            cfg_ = config_factory(arguments.programme_file+'/configs/tracking_config.json')
        else:
            with open(config_path, 'r') as _f:
                cfg_ = TrackingConfig.deserialize(json.load(_f))
        nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                                 nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                                 render_classes=render_classes_)
        nusc_eval.visualization_and_evaluation_of_tracking_results(render_curves=render_curves_)
