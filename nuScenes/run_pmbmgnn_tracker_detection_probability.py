from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import create_scene_folder_name,nms, incorporate_track, predict_for_all, create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file, predict_for_all, readout_parameters
from trackers.PMBMGNN import PMBMGNN_Filter_Point_Target as pmbmgnn_tracker
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
    parser.add_argument('--data_version', default='v1.0-mini', help='choose dataset version between [v1.0-trainval][v1.0-test][v1.0-mini]')
    parser.add_argument('--detection_file',default='/home/blp/Desktop/val_result_with_classification.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/blp/Desktop/MOT')
    parser.add_argument('--dataset_file', default='/home/blp/Desktop/mmdetection3d/data/nuscenes')
    parser.add_argument('--parallel_process', default=5)
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/blp/Desktop')
    parser.add_argument('--render_curves', default='False')
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default='True')
    args = parser.parse_args()
    return args

def main(out_file_directory_for_this_experiment):
    args=parse_args()
    nuscenes_data = NuScenes(version = args.data_version, dataroot=args.dataset_file, verbose=False)
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
    ground_truth_type=dataset_info[set_info]['ground_truth_inst_types']
    ground_truth_id=dataset_info[set_info]['ground_truth_IDS']
    ground_truth_bboxes=dataset_info[set_info]['ground_truth_bboxes']

    with open(args.detection_file, 'rb') as f:
        inference_meta= json.load(f)
    inference=inference_meta['results']

    # read parameters
    with open(config, 'r') as f:
        parameters=json.load(f)
    
    for scene_idx in range(len(list(orderedframe.keys()))):
        scene_token = list(orderedframe.keys())[scene_idx]
        ordered_frames = orderedframe[scene_token]
        ego_info = egoposition[scene_token]
        out_file_directory_for_this_scene=create_scene_folder_name(scene_token, out_file_directory_for_this_experiment)
        # generate filter based on the filter model

        filter_models=[]
        filters=[]
        detection_score_thrs=[]
        filter_predicted_list=[]
        filter_updated_list=[]
        filter_pruned_list=[]
        classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']
        for classification in classifications:
            birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classification, parameters)
            filter_model = pmbmgnn_ulti.gen_filter_model(clutter_rate,P_s,P_d, classification, extraction_thr, ber_thr, poi_thr, eB_thr,bernoulli_gating, use_ds_as_pd, P_init)
            filter_models.append(filter_model)
            pmbm_filter = pmbmgnn_tracker.PMBMGNN_Filter(filter_model)
            filters.append(pmbm_filter)
            detection_score_thrs.append(detection_score_thr)

        for frame_idx, frame_token in enumerate(ordered_frames):
            ground_truth_type_for_this_frame=ground_truth_type[scene_token][str(frame_idx)]
            ground_truth_id_for_this_frame=ground_truth_id[scene_token][str(frame_idx)]
            ground_truth_bboxes_for_this_frame=ground_truth_bboxes[scene_token][str(frame_idx)]
            classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']
            frame_result=[]
            for classification_index, classification in enumerate(classifications):
                Z_k = gen_measurement_of_this_class(detection_score_thrs[classification_index], inference[frame_token], classification)
                if frame_idx == 0:
                    pre_timestamp = timestamps[scene_token][frame_idx]
                    cur_timestamp=pre_timestamp
                    
                    previous_frame_token=''
                    
                    Z_k_with_track=[]
                    for z in Z_k:
                        classification_probability=z['classification_probability']
                        if classification==classifications[np.argmax(classification_probability)]:
                            Z_k_with_track.append(z)
                    filter_predicted = filters[classification_index].predict_initial_step(Z_k_with_track, birth_rate)
                    filter_predicted_list.append(filter_predicted)

                else:
    
                    with open(out_file_directory_for_this_experiment+'/{}.json'.format(previous_frame_token), 'rb') as f:
                        previous_tracking_result=json.load(f)
                    cur_timestamp = timestamps[scene_token][frame_idx]
                    time_lag = (cur_timestamp - pre_timestamp)/1e6
                    Z_k_predict=predict_for_all(previous_tracking_result,time_lag)
                    Z_k_with_track=incorporate_track(Z_k_predict, Z_k)
                    filter_predicted = filters[classification_index].predict(ego_info[str(frame_idx)],time_lag,filter_pruned_list[classification_index], Z_k_with_track, birth_rate)
                    filter_predicted_list.append(filter_predicted)
                # update
                filter_updated = filters[classification_index].update(Z_k_with_track, filter_predicted, confidence_score,-0.5)
                filter_updated_list.append(filter_updated)
                # state extraction
                estimatedStates_for_this_classification = filters[classification_index].extractStates(filter_updated)
                # sort out the data format
                for idx in range(len(estimatedStates_for_this_classification['mean'])):
                    instance_info = {}
                    instance_info['sample_token'] = frame_token
                    translation_of_this_target = [estimatedStates_for_this_classification['mean'][idx][0][0],
                                                  estimatedStates_for_this_classification['mean'][idx][1][0], estimatedStates_for_this_classification['elevation'][idx]]
                    instance_info['translation'] = translation_of_this_target
                    instance_info['size'] = estimatedStates_for_this_classification['size'][idx]
                    instance_info['rotation'] = estimatedStates_for_this_classification['rotation'][idx]
                    instance_info['velocity'] = [estimatedStates_for_this_classification['mean']
                                                 [idx][2][0], estimatedStates_for_this_classification['mean'][idx][3][0]]
                    instance_info['tracking_id'] = estimatedStates_for_this_classification['classification'][idx]+'_'+str(
                        estimatedStates_for_this_classification['id'][idx])
                    instance_info['tracking_name'] = estimatedStates_for_this_classification['classification'][idx]
                    instance_info['tracking_score']=estimatedStates_for_this_classification['detection_score'][idx]
                    frame_result.append(instance_info)
    
                # pruning
                filter_pruned = filters[classification_index].prune(filter_updated)
                filter_pruned_list.append(filter_pruned)
    
            pre_timestamp = cur_timestamp
            previous_frame_token=frame_token
            tracking_result={}
            tracking_result[frame_token]=frame_result

            nuscenes_data.render_tracker_result(0,ground_truth_bboxes_for_this_frame,frame_result,frame_token,nsweeps=1,out_path=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx),bird_eye_view_with_map=False,verbose=False)
            directory=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx)
            print("generating {}".format(directory))        
            plt.close()
            # save the result for this frame
            with open(out_file_directory_for_this_experiment+'/{}.json'.format(frame_token), 'w') as f:
                json.dump(frame_result, f, cls=NumpyEncoder)

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
    classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']
    #classifications=['car']
    # create a folder for this experiment
    now = datetime.now()
    formatedtime = now.strftime("%Y-%m-%d-%H-%M-%S")
    out_file_directory_for_this_experiment = create_experiment_folder(arguments.result_file, formatedtime, set_info)
    main(out_file_directory_for_this_experiment)

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
