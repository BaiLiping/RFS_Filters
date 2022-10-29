from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import BBox, nms, nu_array2mot_bbox,create_scene_folder_name,create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
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
from shapely.geometry import Polygon
import itertools


all='bicycle+motorcycle+trailer+truck+bus+pedestrian+car'
classifications_index = {'bicycle':0,'motorcycle':1,  'trailer':2, 'truck':3,'bus':4,'pedestrian':5,'car':6}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', default='v1.0-mini', help='choose dataset version between [v1.0-trainval][v1.0-test][v1.0-mini]')
    parser.add_argument('--detection_file',default='/home/blp/Desktop/val_result_with_classification.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/blp/Desktop/MOT')
    parser.add_argument('--dataset_file', default='/home/blp/Desktop/mmdetection3d/data/nuscenes')
    parser.add_argument('--parallel_process', default=1)
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/blp/Desktop')
    parser.add_argument('--render_curves', default='False')
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default='True')
    args = parser.parse_args()
    return args


def iou3d(box_a, box_b):
    boxa_corners = np.array(BBox.box2corners2d(box_a))
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap_area = reca.intersection(recb).area
    iou_2d = overlap_area / (reca.area + recb.area - overlap_area)

    ha, hb = box_a.h, box_b.h
    za, zb = box_a.z, box_b.z
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    overlap_volume = overlap_area * overlap_height
    union_volume = box_a.w * box_a.l * ha + box_b.w * box_b.l * hb - overlap_volume
    iou_3d = overlap_volume / (union_volume + 1e-5)

    return iou_2d,iou_3d

def predict_for_all(Z_k, lag_time):
    F = np.eye(4, dtype=np.float64)
    I = lag_time*np.eye(2, dtype=np.float64)
    F[0:2, 2:4] = I
    '''
    instance_info['translation'] = translation_of_this_target
    instance_info['size'] = estimatedStates_for_this_classification['size'][idx]
    instance_info['rotation'] = estimatedStates_for_this_classification['rotation'][idx]
    instance_info['velocity'] = [estimatedStates_for_this_classification['mean']
                                 [idx][2][0], estimatedStates_for_this_classification['mean'][idx][3][0]]
    instance_info['tracking_id'] = estimatedStates_for_this_classification['classification'][idx]+'_'+str(
        estimatedStates_for_this_classification['id'][idx])
    instance_info['tracking_name'] = estimatedStates_for_this_classification['classification'][idx]
    instance_info['tracking_score']=estimatedStates_for_this_classification['detection_score'][idx]
    '''
    Z_k_new=[]
    for z in Z_k:
        position=z['translation'][:2]
        for i in range(2):
            position.append(z['velocity'][i])
        new_position=F.dot(position)
        for j in range(2):
            z['translation'][j]=new_position[j]
        Z_k_new.append(z)
    return Z_k_new

def incorporate_track(Z_k_predict, Z_k):
    #Z_k_delete=[]
    Z_k_new=[]
    for index_2, z_2 in enumerate(Z_k): 
        if z_2['detection_name']=='pedestrian' or z_2['detection_name']=='motorcycle':
            Z_k_new.append(z_2)
        else:
            if len(Z_k_predict)>0:
                for index_1, z_1 in enumerate(Z_k_predict): 
                    iou_result=iou3d(nu_array2mot_bbox(z_1), nu_array2mot_bbox(z_2))[0]
                    if iou_result>0.7:
                        if z_1['tracking_name']!='truck':
                                z_2['detection_score']*=0.1
                                #print('{} track {} detection'.format(z_1['tracking_name'],z_2['detection_name']))
                            #else:
                        #    z_2['detection_score']=1            
                Z_k_new.append(z_2)
            else:
                Z_k_new.append(z_2)
                        #print('change from {} to {} at frame'.format(Z_k[index_2]['detection_name'], z_1['tracking_name'], frame_index))
                        #Z_k[index_2]['detection_name']=z_1['tracking_name']
    else:
        Z_k_new=Z_k
                    
    #for z_delete in Z_k_delete:
    #    print('deleted one overlapping measurement for {} at frame {}'.format(z_delete['detection_name'], frame_index))
    #    Z_k.remove(z_delete)
    return Z_k_new

def compute_duplicated_detection(Z_k):
    skip_z_index=[]
    Z_k_new=[]
    for index_1, z_1 in enumerate(Z_k):
        if index_1 not in skip_z_index: 
            duplicated_detection=[]
            for index_2, z_2 in enumerate(Z_k): 
                twodiou_result, threediou_result=iou3d(nu_array2mot_bbox(z_1), nu_array2mot_bbox(z_2))
                if twodiou_result>0.7 and index_2!=index_1:
                    skip_z_index.append(index_2)
                    duplicated_detection.append(copy.deepcopy(z_2))
            
            z_1['duplicated_detection']=duplicated_detection
            Z_k_new.append(z_1)
    return Z_k_new   

def gen_measurement_all(estimated_bboxes_at_current_frame):
        
    # read parameters
    with open('/home/blp/Desktop/MOT/configs/pmbmgnn_parameters.json', 'r') as f:
        parameters=json.load(f)
    Z_k=[]
    for classification in ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']:
        birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classification, parameters)
         
        for box_index, box in enumerate(estimated_bboxes_at_current_frame):
            '''
            if classification=='pedestrian':
                if box['detection_name']==classification:
                    if box['translation'][2]<1.5:
                        Z_k.append(box)
                        
            else:
                if box['detection_name']==classification and box['detection_score']>detection_score_thr:
                    if box['translation'][2]<3:
                        Z_k.append(box)
            '''
            if box['detection_name']==classification and box['detection_score']>detection_score_thr:
                Z_k.append(box)
            
    return Z_k

def fill_frame_result(frame_result,estimation, frame_token):
    for idx in range(len(estimation['mean'])):
        instance_info = {}
        instance_info['sample_token'] = frame_token
        translation_of_this_target = [estimation['mean'][idx][0][0],
                                      estimation['mean'][idx][1][0], estimation['elevation'][idx]]
        instance_info['translation'] = translation_of_this_target
        instance_info['size'] = estimation['size'][idx]
        instance_info['rotation'] = estimation['rotation'][idx]
        instance_info['velocity'] = [estimation['mean']
                                     [idx][2][0], estimation['mean'][idx][3][0]]
        #instance_info['tracking_id'] = estimation['classification'][idx]+'_'+str(
        #    estimation['id'][idx])
        instance_info['tracking_id'] = str(classifications_index[estimation['classification'][idx]])+'_'+str(estimation['id'][idx])
        instance_info['tracking_name'] = estimation['classification'][idx]
        instance_info['tracking_score']=estimation['detection_score'][idx]
        #results_all['results'][frame_token].append(instance_info)
        frame_result.append(instance_info)
    return frame_result   

def main(token,out_file_directory_for_this_experiment):
    args=parse_args()
    #out_file_directory_for_this_experiment='/home/blp/Desktop/experiment_sep30'
    #nuscenes_data = NuScenes(version = args.data_version, dataroot=args.dataset_file, verbose=False)
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
            # prediction
            if frame_idx == 0:
                filter_predicted = pmbm_filter.predict_initial_step(Z_k, birth_rate)
                last_frame_result=[]

            else:
                filter_predicted = pmbm_filter.predict(ego_info[str(frame_idx)],time_lag,filter_pruned, Z_k, birth_rate)
                #estimatedStates_temp= pmbm_filter.extractStates(filter_predicted)
                #last_frame_result=fill_frame_result(last_frame_result,estimatedStates_temp, frame_token)

            #incorporate_track(last_frame_result,Z_k)
            # update
            filter_updated = pmbm_filter.update(Z_k, filter_predicted, confidence_score,giou_gating)
            # state extraction
            estimatedStates_for_this_classification = pmbm_filter.extractStates(filter_updated)
            # sort out the data format
            frame_result_temp=[]
            frame_result_temp=fill_frame_result(frame_result_temp,estimatedStates_for_this_classification, frame_token)
            duplicated_tracks=compute_duplicated_detection(frame_result_temp)
            for track in duplicated_tracks:
                if len(track['duplicated_detection'])>0:
                    tracking_scores=[]
                    tracking_names=[]
                    #print('tracking name before: {}'.format(track['tracking_name']))
                    tracking_scores.append(track['tracking_score'])
                    tracking_names.append(track['tracking_name'])
                    for duplicated_track in track['duplicated_detection']:
                        tracking_scores.append(duplicated_track['tracking_score'])
                        tracking_names.append(duplicated_track['tracking_name'])
                    normalized_tracking_scores=tracking_scores/np.sum(tracking_scores)
                    best_tracking_name_index=np.argmax(normalized_tracking_scores)
                    #best_tracking_score=tracking_scores[best_tracking_name_index]
                    best_tracking_name=tracking_names[best_tracking_name_index]
                    track['tracking_name']=best_tracking_name
                    #print('tracking name after: {}'.format(track['tracking_name']))
                    #track['tracking_score']=best_tracking_score
                    frame_result.append(track)
                else:
                    frame_result.append(track)

            # pruning
            filter_pruned = pmbm_filter.prune(filter_updated)
            pre_timestamp = cur_timestamp

            #plot result
            #nuscenes_data.render_tracker_result(0,ground_truth_bboxes_for_this_frame,ground_truth_type_for_this_frame,frame_result,frame_token,nsweeps=1,out_path=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx),bird_eye_view_with_map=False,verbose=False)
            #directory=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx)
            #print("generating {}".format(directory))        
            #plt.close()
            

            # save the result for this scene
            with open(out_file_directory_for_this_experiment+'/{}.json'.format(frame_token), 'w') as f:
                json.dump(frame_result, f, cls=NumpyEncoder)

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
            cfg_ = config_factory(arguments.programme_file+'/configs/tracking_config.json')
        else:
            with open(config_path, 'r') as _f:
                cfg_ = TrackingConfig.deserialize(json.load(_f))
        nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                                 nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                                 render_classes=render_classes_)
        nusc_eval.visualization_and_evaluation_of_tracking_results(render_curves=render_curves_)