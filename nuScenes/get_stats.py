from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import BBox, compute_overlapping_score_to_gt,predict_for_all,incorporate_track,visualize_duplicated_detection,iou3d,compute_duplicated_detection,fill_frame_result,gen_measurement_all, nms, nu_array2mot_bbox,create_scene_folder_name,create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
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
    parser.add_argument('--detection_file',default='/home/zhubinglab/Desktop/val_results_fixed.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/zhubinglab/Desktop/nuScenes_Tracker')
    parser.add_argument('--dataset_file', default='/home/zhubinglab/Desktop/mmdetection3d/data/nuscenes')
    args = parser.parse_args()
    return args


def main():
    args=parse_args()
    dataset_info_file=args.programme_file+'/configs/dataset_info_with_train.json'
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
    ground_truth_type=dataset_info[set_info]['ground_truth_inst_types']
    ground_truth_bboxes=dataset_info[set_info]['ground_truth_bboxes']

    with open(args.detection_file, 'rb') as f:
        inference_meta= json.load(f)
    inference=inference_meta['results']
    inference_without_duplication=copy.deepcopy(inference_meta)
    correctedly_detected_ground_truth={}
    '''
    for classification in classifications:
        correctedly_detected_ground_truth[classification]={}
        correctedly_detected_ground_truth[classification]['single_count']=0
        correctedly_detected_ground_truth[classification]['single_count_correct_classification']=0
        correctedly_detected_ground_truth[classification]['single_count_wrong_classification']=0
        correctedly_detected_ground_truth[classification]['single_count_wrong_classification_record']=[np.zeros(len(classifications)) for i in range(len(classifications))]
    '''
    
    for scene_idx in range(len(list(orderedframe.keys()))):
        scene_token = list(orderedframe.keys())[scene_idx]
        ordered_frames = orderedframe[scene_token]

        for frame_idx, frame_token in enumerate(ordered_frames):
            #inference_without_duplication['results'][frame_token]=[]
            ground_truth_bboxes_for_this_frame_original=ground_truth_bboxes[scene_token][str(frame_idx)]
            ground_truth_type_for_this_frame=ground_truth_type[scene_token][str(frame_idx)]
            ground_truth_bboxes_for_this_frame=[]
            for gt in ground_truth_bboxes_for_this_frame_original:
                instance={}
                instance['translation']=[gt[0], gt[1], gt[2]]
                instance['size']=[gt[3], gt[4], gt[5]]
                instance['rotation']=[gt[6], gt[7], gt[8], gt[9]]
                ground_truth_bboxes_for_this_frame.append(instance)

            # get measurements at global coordinate
            if frame_token in inference.keys():
                estimated_bboxes_at_current_frame = inference[frame_token]
            else:
                print('lacking inference file')
                break
            
            Z_k_original=gen_measurement_all(estimated_bboxes_at_current_frame)

            Z_k_all_temp=compute_duplicated_detection(Z_k_original)
            Z_k_all=[]
            for z in Z_k_all_temp:
                if len(z['duplicated_detection'])>0:
                    classification_list=[]
                    score_list=[]
                    classification_list.append(z['detection_name'])
                    score_list.append(z['detection_score'])
                    for z_1 in z['duplicated_detection']:
                        classification_list.append(z['detection_name'])
                        score_list.append(z['detection_score'])
                    max_index=np.argmax(score_list)
                    max_classification=classification_list[max_index]
                    max_score=np.max(score_list)
                    z['detection_name']=max_classification
                    z['detection_score']=max_score
                Z_k_all.append(z)

            inference_without_duplication['results'][frame_token]=Z_k_all
            '''
            overlapping_score_twod,overlapping_score_threed=compute_overlapping_score_to_gt(Z_k_original, ground_truth_bboxes_for_this_frame)
            for z_index, z in enumerate(Z_k_original):
                overlapping_score=overlapping_score_twod[z_index]
                if len(overlapping_score)>0:
                    max_score_index=np.argmax(overlapping_score)
                    if np.max(overlapping_score)>0.3:
                        classification_of_gt=ground_truth_type_for_this_frame[max_score_index]
                        correctedly_detected_ground_truth[classification_of_gt]['single_count']+=1    
                        if z['detection_name']==classification_of_gt:
                            correctedly_detected_ground_truth[classification_of_gt]['single_count_correct_classification']+=1
                        else:
                            correctedly_detected_ground_truth[classification_of_gt]['single_count_wrong_classification']+=1
                            correctedly_detected_ground_truth[classification_of_gt]['single_count_wrong_classification_record'][classifications_index[classification_of_gt]][classifications_index[z['detection_name']]]+=1
            '''
    # save the results without duplication
    with open('/home/zhubinglab/Desktop/val_results_without_duplication.json', 'w') as f:
        json.dump(inference_without_duplication,f,cls=NumpyEncoder)
    
    ## save the stats
    #with open('/home/zhubinglab/Desktop/training_statistics_hard_classifier.json', 'w') as f:
    #    json.dump(correctedly_detected_ground_truth, f, cls=NumpyEncoder)  

    '''
    # print out the stats
    for idx1, classification in enumerate(classifications):
        print('correctly detected {} for {} times with single detection'.format(classification, correctedly_detected_ground_truth[classification]['single_count']))
        print('correctly detected {} for {} times with single detection which is the correct classificication'.format(classification, correctedly_detected_ground_truth[classification]['single_count_correct_classification']))
        print('correctly detected {} for {} times with single detection which is the wrong classificication'.format(classification, correctedly_detected_ground_truth[classification]['single_count_wrong_classification']))
        #print('the list of wrong detetion is: {}'.format(correctedly_detected_ground_truth[classification]['single_count_wrong_classification_record']))

        for idx2, classification2 in enumerate(classifications):   
            if correctedly_detected_ground_truth[classification_of_gt]['single_count_wrong_classification_record'][classification_of_gt[classification]][idx2]>0:
                print('mis-detect {} to be {} for {} times'.format(classification, classification2, correctedly_detected_ground_truth[classification]['single_count_wrong_classification_record'][idx1][idx2]))
    '''
                    
if __name__ == '__main__':
    main()