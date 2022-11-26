from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import BBox, create_scene_folder_name,nu_array2mot_bbox,iou3d, nms, create_experiment_folder, initiate_submission_file, gen_measurement_all,gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
from trackers.PMBMGNN import PMBMGNN_Filter_Point_Target4 as pmbmgnn_tracker
from trackers.PMBMGNN import util as pmbmgnn_ulti
from datetime import datetime
from evaluate.util.utils import TrackingConfig, config_factory
from evaluate.evaluate_tracking_result import TrackingEval
import multiprocessing
import argparse
from tqdm import tqdm
import numpy as np
from utils.nuscenes_dataset import NuScenes
from matplotlib import pyplot as plt
import copy


all='bicycle+motorcycle+trailer+truck+bus+pedestrian+car'
classifications_index = {'bicycle':0,'motorcycle':1,  'trailer':2, 'truck':3,'bus':4,'pedestrian':5,'car':6}
classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', default='v1.0-trainval', help='choose dataset version between [v1.0-trainval][v1.0-test][v1.0-mini]')
    parser.add_argument('--detection_file',default='/home/bailiping/Desktop/preprocessed_result.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/bailiping/Desktop/MOT')
    parser.add_argument('--dataset_file', default='/media/bailiping/My Passport/mmdetection3d/data/nuscenes')
    parser.add_argument('--parallel_process', default=5)
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/bailiping/Desktop/experiment_result')
    parser.add_argument('--render_curves', default='False')
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default='True')
    args = parser.parse_args()
    return args

def main():
    args=parse_args()
    nuscenes_data = NuScenes(version = args.data_version, dataroot=args.dataset_file, verbose=False)
    dataset_info_file='/media/bailiping/My\ Passport/mmdetection3d/data/nuscenes/configs/dataset_info.json'
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

    inference_with_classification_score={}
    inference_with_classification_score['results']={}

    # readout parameters
    # generate filter model based on the classification
    for scene_idx in range(len(list(orderedframe.keys()))):
        
        scene_token = list(orderedframe.keys())[scene_idx]
        out_file_directory_for_this_scene=create_scene_folder_name(scene_token, '/media/bailiping/'My Passport'/mmdetection3d/data/nuscenes/visualization')
        ordered_frames = orderedframe[scene_token]
        ego_info = egoposition[scene_token]
        # generate filter based on the filter model
        for frame_idx, frame_token in enumerate(ordered_frames):
            if frame_idx == 0:
                pre_timestamp = timestamps[scene_token][frame_idx]
            cur_timestamp = timestamps[scene_token][frame_idx]
            time_lag = (cur_timestamp - pre_timestamp)/1e6
            # get measurements at global coordinate
            if frame_token in inference.keys():
                estimated_bboxes_at_current_frame = inference[frame_token]
            else:
                print('lacking inference file')
                break
            # initiate submission file
            inference_with_classification_score['results'][frame_token] = []
            Z_k = gen_measurement_all(0, estimated_bboxes_at_current_frame)
            ground_truth_type_for_this_frame=ground_truth_type[scene_token][str(frame_idx)]
            ground_truth_id_for_this_frame=ground_truth_id[scene_token][str(frame_idx)]
            ground_truth_bboxes_for_this_frame=ground_truth_bboxes[scene_token][str(frame_idx)]
            # preprocessing of input. Remove bboxes that overlaps
            bbox_num=len(Z_k)
            ious=np.zeros((bbox_num, bbox_num))
            for index_1, z_1 in enumerate(Z_k):
                for index_2, z_2 in enumerate(Z_k):  
                    ious[index_1,index_2]=iou3d(nu_array2mot_bbox(z_1), nu_array2mot_bbox(z_2))[0]

            delete_measurement=[]
            for measurement_index in range(bbox_num):
                ious_for_this_measurement=ious[measurement_index]
                ious_for_this_measurement_over_threshold=np.where(ious_for_this_measurement>0.1)[0]
                number_of_overlapping_detections=len(ious_for_this_measurement_over_threshold)
                                
                classification_probability = np.zeros(7)
                if number_of_overlapping_detections>1:
                    for idx in ious_for_this_measurement_over_threshold:
                        classification_probability[classifications_index[Z_k[idx]['detection_name']]]=Z_k[idx]['detection_score']
                    detection_classification=classifications[np.argmax(classification_probability)]
                    if detection_classification!=Z_k[measurement_index]['detection_name']:
                        delete_measurement.append(Z_k[measurement_index])
                else:
                    classification_probability[classifications_index[Z_k[measurement_index]['detection_name']]]=1
                
            for delete_meas in delete_measurement:
                Z_k.remove(delete_meas)
            for measurement_index in range(len(Z_k)):
                Z_k_new=copy.deepcopy(Z_k[measurement_index])
                Z_k_new['classification_probability']=classification_probability
                inference_with_classification_score['results'][frame_token].append(Z_k_new)
            #nuscenes_data.render_tracker_result(0,ground_truth_bboxes_for_this_frame,Z_k,frame_token,nsweeps=1,out_path=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx),bird_eye_view_with_map=False,verbose=False)
            #directory=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx)
            #print("generating {}".format(directory))        
            #plt.close()
    with open('/home/bailiping/Desktop/val_result_remove_overlapping.json', 'w') as f:
        json.dump(inference_with_classification_score,f,cls=NumpyEncoder)

if __name__ == '__main__':
    main()