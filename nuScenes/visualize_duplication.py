from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import BBox, visualize_duplicated_detection,iou3d,compute_duplicated_detection,fill_frame_result,gen_measurement_all, nms, nu_array2mot_bbox,create_scene_folder_name,create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', default='v1.0-trainval', help='choose dataset version between [v1.0-trainval][v1.0-test][v1.0-mini]')
    parser.add_argument('--detection_file',default='/home/bailiping/Desktop/val_result_with_classification.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/bailiping/Desktop/MOT')
    parser.add_argument('--dataset_file', default='/media/bailiping/My Passport/mmdetection3d/data/nuscenes')
    parser.add_argument('--parallel_process', default=2)
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/bailiping/Desktop/experiment_result')
    parser.add_argument('--render_curves', default='False')
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default='True')
    args = parser.parse_args()
    return args  

def main(token,out_file_directory_for_this_experiment):
    args=parse_args()
    #out_file_directory_for_this_experiment='/home/bailiping/Desktop/experiment_sep30'
    nuscenes_data = NuScenes(version = args.data_version, dataroot=args.dataset_file, verbose=False)
    dataset_info_file='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/configs/dataset_info.json'
    config='/media/bailiping/My Passport/mmdetection3d/data/nuscenes/configs/pmbmgnn_parameters.json'
    
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
    ground_truth_type=dataset_info[set_info]['ground_truth_inst_types']
    ground_truth_id=dataset_info[set_info]['ground_truth_IDS']
    ground_truth_bboxes=dataset_info[set_info]['ground_truth_bboxes']

    with open(args.detection_file, 'rb') as f:
        inference_meta= json.load(f)
    inference=inference_meta['results']

    # read parameters
    with open(config, 'r') as f:
        parameters=json.load(f)

    # readout parameters
    birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters('car', parameters)
    # adjust detection_scrore_thr for pointpillars
    # this step is necessary because the pointpillar detections for cars and pedestrian would generate excessive number of bounding boxes and result in unnecessary compuations.
    # generate filter model based on the classification
    filter_model = pmbmgnn_ulti.gen_filter_model(clutter_rate,P_s,P_d, 'car', extraction_thr, ber_thr, poi_thr, eB_thr,bernoulli_gating, use_ds_as_pd, P_init)
    for scene_idx in range(len(list(orderedframe.keys()))):
        if scene_idx % args.parallel_process != token:
            continue
        scene_token = list(orderedframe.keys())[scene_idx]
        out_file_directory_for_this_scene=create_scene_folder_name(scene_token, out_file_directory_for_this_experiment)
        ordered_frames = orderedframe[scene_token]
        ego_info = egoposition[scene_token]
        # generate filter based on the filter model
        pmbm_filter = pmbmgnn_tracker.PMBMGNN_Filter(filter_model)
        for frame_idx, frame_token in enumerate(ordered_frames):
            ground_truth_type_for_this_frame=ground_truth_type[scene_token][str(frame_idx)]
            ground_truth_id_for_this_frame=ground_truth_id[scene_token][str(frame_idx)]
            ground_truth_bboxes_for_this_frame=ground_truth_bboxes[scene_token][str(frame_idx)]
            frame_result=[]
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

            Z_k_original=gen_measurement_all(estimated_bboxes_at_current_frame)
            Z_k=compute_duplicated_detection(Z_k_original)
            duplicated_Z_k=visualize_duplicated_detection(Z_k)

            #plot result
            nuscenes_data.render_tracker_result(0,ground_truth_bboxes_for_this_frame,ground_truth_type_for_this_frame,duplicated_Z_k,frame_token,nsweeps=1,out_path=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx),bird_eye_view_with_map=False,verbose=False)
            directory=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx)
            print("generating {}".format(directory))        
            plt.close()
            
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
    #classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']
    classifications=['car']
    # create a folder for this experiment
    now = datetime.now()
    formatedtime = now.strftime("%Y-%m-%d-%H-%M-%S")
    out_file_directory_for_this_experiment = create_experiment_folder(arguments.result_file, formatedtime, set_info)
    inputarguments=[]
    for token in range(arguments.parallel_process):
        inputarguments.append((token,out_file_directory_for_this_experiment))   
    #with multiprocessing.Pool(arguments.parallel_process) as p:
    #    p.map(main, inputarguments)

    ## start processing information
    pool = multiprocessing.Pool(processes=arguments.parallel_process)
    pool.starmap(main,inputarguments)
    
    pool.close()
    
    with open(dataset_info_file, 'r') as f:
        dataset_info=json.load(f)

    orderedframe=dataset_info[set_info]['ordered_frame_info']
    # aggregate results for all classifications
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
            cfg_ = config_factory('/media/bailiping/My Passport/mmdetection3d/data/nuscenes/configs/tracking_config.json')
        else:
            with open(config_path, 'r') as _f:
                cfg_ = TrackingConfig.deserialize(json.load(_f))
        nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                                 nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                                 render_classes=render_classes_)
        nusc_eval.visualization_and_evaluation_of_tracking_results(render_curves=render_curves_)