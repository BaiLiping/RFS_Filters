from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import nms, create_experiment_folder, initiate_submission_file, gen_measurement_all,gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
from trackers.PMBMGNN import PMBMGNN_Filter_Point_Target3 as pmbmgnn_tracker
from trackers.PMBMGNN import util as pmbmgnn_ulti
from datetime import datetime
from evaluate.util.utils import TrackingConfig, config_factory
from evaluate.evaluate_tracking_result import TrackingEval
import multiprocessing
import argparse
from tqdm import tqdm

all='bicycle+motorcycle+trailer+truck+bus+pedestrian+car'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', default='v1.0-trainval', help='choose dataset version between [v1.0-trainval][v1.0-test][v1.0-mini]')
    parser.add_argument('--detection_file',default='/media/bailiping/'My Passport'/mmdetection3d/data/nuscenes/detection_result/BEV_fusion/val_results.json', help='directory for the inference file')
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
    preprocessed_detection = {}
    preprocessed_detection['results']={}
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

    with open(args.detection_file, 'rb') as f:
        inference_meta= json.load(f)
    inference=inference_meta['results']

    # read parameters
    with open(config, 'r') as f:
        parameters=json.load(f)

    # readout parameters
    nms_score=0.1
    # adjust detection_scrore_thr for pointpillars
    # this step is necessary because the pointpillar detections for cars and pedestrian would generate excessive number of bounding boxes and result in unnecessary compuations.
    # generate filter model based on the classification
    for scene_idx in range(len(list(orderedframe.keys()))):
        scene_token = list(orderedframe.keys())[scene_idx]
        ordered_frames = orderedframe[scene_token]
        # generate filter based on the filter model
        for frame_idx, frame_token in enumerate(ordered_frames):
            if frame_idx == 0:
                pre_timestamp = timestamps[scene_token][frame_idx]
            cur_timestamp = timestamps[scene_token][frame_idx]
            # get measurements at global coordinate
            if frame_token in inference.keys():
                estimated_bboxes_at_current_frame = inference[frame_token]
            else:
                print('lacking inference file')
                break
            preprocessed_detection['results'][frame_token]=[]
            classifications=['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']
            for classification in classifications:
                Z_k_all = gen_measurement_of_this_class(0, estimated_bboxes_at_current_frame, classification)
                #Z_k_all=gen_measurement_all(detection_score_thr, estimated_bboxes_at_current_frame)
                # preprocessing of input. Remove bboxes that overlaps
                result_indexes = nms(Z_k_all, threshold=nms_score)
                Z_k=[]
                for idx in result_indexes:
                    Z_k.append(Z_k_all[idx])
                
                for bbox in Z_k:
                    temp_result={}
                    temp_result['translation']=bbox['translation']
                    temp_result['size']=bbox['size']
                    temp_result['velocity']=bbox['velocity']
                    temp_result['rotation']=bbox['rotation']
                    temp_result['detection_score']=bbox['detection_score']
                    temp_result['detection_name']=bbox['detection_name']
                    temp_result['sample_token']=bbox['sample_token']
                    preprocessed_detection['results'][frame_token].append(temp_result)
    
    # save the result for this scene
    with open('/home/bailiping/Desktop/preprocessed_result.json', 'w') as f:
        json.dump(preprocessed_detection, f, cls=NumpyEncoder)

if __name__ == '__main__':
    main()
