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
    parser.add_argument('--detection_file',default='/home/blp/Desktop/val_results_fixed.json', help='directory for the inference file')
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


def main(token, out_file_directory_for_this_experiment):
    args=parse_args()
    if args.plot_result:
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
    
    filters=[]
    birth_rate_list=[]

    for classification in classifications:
        # readout parameters
        z_adjustment, birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classification, parameters)
        # adjust detection_scrore_thr for pointpillars
        # this step is necessary because the pointpillar detections for cars and pedestrian would generate excessive number of bounding boxes and result in unnecessary compuations.
        # generate filter model based on the classification
        filter_model = pmbmgnn_ulti.gen_filter_model(clutter_rate,P_s,P_d, classification, extraction_thr, ber_thr, poi_thr, eB_thr,bernoulli_gating, use_ds_as_pd, P_init)
        pmbm_filter = pmbmgnn_tracker.PMBMGNN_Filter(filter_model)
        filters.append(pmbm_filter)
        birth_rate_list.append(birth_rate)
     
    
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
                Z_k=[]
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
                frame_result_withduplication=fill_frame_result(frame_result,estimatedStates_for_this_classification,frame_token)

                # get rid of duplicated tracks
                #frame_result_without_duplication_temp=compute_duplicated_detection(frame_result_withduplication)
                #frame_result_without_duplication=[]
                '''
                for track_idx, track in enumerate(frame_result_without_duplication_temp):
                    if track['tracking_name']=='motorcycle' or track['tracking_name']=='bus':
                        if len(track['duplicated_detection'])>0:
                            probabilities=[]
                            classes=[]
                            ids=[]
                            probabilities.append(track['tracking_score'])
                            classes.append(track['tracking_name'])
                            ids.append(track['tracking_id'])
                            for z_1 in track['duplicated_detection']:
                                probabilities.append(z_1['tracking_score'])
                                classes.append(z_1['tracking_name'])
                                ids.append(z_1['tracking_id'])
         
                            # adjust detection name
                            normalized_tracking_scores=probabilities/np.sum(probabilities)
                            highest_class_index=np.argmax(probabilities)
                            # if there is a clear preference
                            if normalized_tracking_scores[highest_class_index]>0.8:
                                highest_class=classes[highest_class_index]
                                track['tracking_name']=highest_class
                                track['tracking_score']=probabilities[highest_class_index]
                                track['tracking_id']=ids[highest_class_index]
                                frame_result_without_duplication.append(track)
                            else:
                                frame_result_without_duplication.append(track)
                                for z_1 in track['duplicated_detection']:
                                    frame_result_without_duplication.append(z_1)
                        else:
                            frame_result_without_duplication.append(track)
                '''
                # pruning
                filter_pruned = filters[classification_index].prune(filter_updated)
                filter_pruned_dict[classification]=filter_pruned
            
            # update time
            pre_timestamp = cur_timestamp
            previous_frame_token_exist=True
            previous_frame_token=frame_token
            
            # save the result for this scene
            with open(out_file_directory_for_this_experiment+'/{}.json'.format(frame_token), 'w') as f:
                json.dump(frame_result_withduplication, f, cls=NumpyEncoder)
            if args.plot_result:
                if frame_idx<2:
                    nuscenes_data.render_tracker_result(0,ground_truth_bboxes_for_this_frame,ground_truth_type_for_this_frame,frame_result_withduplication,frame_token,nsweeps=1,out_path=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx),bird_eye_view_with_map=False,verbose=False)
                    directory=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx)
                    print("generating {}".format(directory))        
                    plt.close()

        print('done with scene {} process {}'.format(scene_idx, token))

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
