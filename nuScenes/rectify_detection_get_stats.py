from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import BBox, create_scene_folder_name, instance_info2bbox_array,nu_array2mot_bbox,iou3d, nms, create_experiment_folder, initiate_submission_file, gen_measurement_all,gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
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
from evaluate.util.utils import points_in_box,Box,TrackingEvaluation,recall_metric_curve, summary_plot,AVG_METRIC_MAP, MOT_METRIC_MAP, LEGACY_METRICS, TrackingMetrics, TrackingMetricDataList, TrackingConfig,DetectionBox, TrackingBox, \
    TrackingMetricData, create_tracks,print_final_metrics, render_for_the_best_threshold,config_factory,load_prediction, load_gt, add_center_dist
import sklearn
from pyquaternion import Quaternion
from motmetrics.lap import linear_sum_assignment


all='bicycle+motorcycle+trailer+truck+bus+pedestrian+car'
classifications_index = {'bicycle':0,'motorcycle':1,  'trailer':2, 'truck':3,'bus':4,'pedestrian':5,'car':6}
classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']
difference_in_z={'bicycle':0.58,'motorcycle':0.57,'trailer':2.02,'truck':1.36,'bus':1.59,'pedestrian':1.03,'car':0.82}

class_range={
    "car": 50,
    "truck": 50,
    "bus": 50,
    "trailer": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40
  }

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

def get_box_class_field(eval_boxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    box = eval_boxes[0]
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)
    return class_field

def filter_eval_boxes(nusc: NuScenes,
                      eval_boxes,
                      class_range,
                      frame_token,
                      verbose: bool = False):
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = get_box_class_field(eval_boxes)
    eval_boxes_new=[]
    for ind, box in enumerate(eval_boxes):
        if box.__getattribute__(class_field) in classifications:
            if box.ego_dist < class_range[box.__getattribute__(class_field)]:
                eval_boxes_new.append(box)

    # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
    eval_boxes_without_zero_points = [box for box in eval_boxes_new if not box.num_pts == 0]

    # Perform bike-rack filtering.
    sample_anns = nusc.get('sample', frame_token)['anns']
    bikerack_recs = [nusc.get('sample_annotation', ann) for ann in sample_anns if
                     nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack']
    bikerack_boxes = [Box(rec['translation'], rec['size'], Quaternion(rec['rotation'])) for rec in bikerack_recs]
    filtered_boxes = []
    for box in eval_boxes_without_zero_points:
        if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
            in_a_bikerack = False
            for bikerack_box in bikerack_boxes:
                if np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
                    in_a_bikerack = True
            if not in_a_bikerack:
                filtered_boxes.append(box)
        else:
            filtered_boxes.append(box)
    return filtered_boxes

def formated_to_dict(instance_formated):
    instance={}
    instance['sample_token']=instance_formated.sample_token
    instance['ego_dist']=instance_formated.ego_dist
    instance['ego_translation']=instance_formated.ego_translation
    instance['rotation']=instance_formated.rotation
    instance['size']=instance_formated.size
    instance['tracking_id']=instance_formated.tracking_id
    instance['tracking_name']=instance_formated.tracking_name
    instance['tracking_score']=instance_formated.tracking_score
    instance['translation']=instance_formated.translation
    instance['velocity']=instance_formated.velocity
    return instance

def formated_to_dict_det(instance_formated):
    instance={}
    instance['sample_token']=instance_formated.sample_token
    instance['ego_dist']=instance_formated.ego_dist
    instance['ego_translation']=instance_formated.ego_translation
    instance['rotation']=instance_formated.rotation
    instance['size']=instance_formated.size
    instance['detection_name']=instance_formated.detection_name
    instance['detection_score']=instance_formated.detection_score
    instance['translation']=instance_formated.translation
    instance['velocity']=instance_formated.velocity
    return instance

def main():
    args=parse_args()
    evaluation_result={}
    gt_result={}
    number_of_classifications=len(classifications)

    register_for_gt_classifications=[np.zeros(number_of_classifications) for i in range(number_of_classifications)]
    count_for_gt_classifications=np.zeros(number_of_classifications)
    z_for_gt_classifications=[[] for i in range(7)]
    z_for_detection_classifications=[[] for i in range(7)]
    hit_for_gt_classifications=np.zeros(number_of_classifications)
    double_hit_for_gt_classifications=np.zeros(number_of_classifications)
    single_hit_correct_classification=np.zeros(number_of_classifications)
    single_hit_wrong_classification=[np.zeros(number_of_classifications) for i in range(number_of_classifications)]
    double_hit_correct_classification=np.zeros(number_of_classifications)
    double_hit_wrong_classification=[np.zeros(number_of_classifications) for i in range(number_of_classifications)]
    multiple_correct_detections_for_gt=np.zeros(number_of_classifications)

    nuscenes_data = NuScenes(version = args.data_version, dataroot=args.dataset_file, verbose=False)
    dataset_info_file='/media/bailiping/My\ Passport/mmdetection3d/data/nuscenes/configs/dataset_info.json'
    
    set_info='val'
    # read ordered frame info
    with open(dataset_info_file, 'rb') as f:
        dataset_info=json.load(f)

    orderedframe=dataset_info[set_info]['ordered_frame_info']

    pred_boxes, _ = load_prediction(args.detection_file, 500, DetectionBox,verbose=True)
    gt_boxes = load_gt(nuscenes_data, set_info, TrackingBox, verbose=True)
    pred_boxes = add_center_dist(nuscenes_data, pred_boxes)
    gt_boxes = add_center_dist(nuscenes_data, gt_boxes)
    prediction_rectified={}
    prediction_rectified['meta']=[]
    prediction_rectified['results']={}

    # readout parameters
    # generate filter model based on the classification
    for scene_idx in range(len(list(orderedframe.keys()))):
        
        scene_token = list(orderedframe.keys())[scene_idx]
        ordered_frames = orderedframe[scene_token]
        # generate filter based on the filter model
        for frame_idx, frame_token in enumerate(ordered_frames):
            evaluation_result[frame_token]=[]          
            gt_result[frame_token]=[]
            prediction_rectified['results'][frame_token]=[]

            gt_bboxes_of_this_frame_before_filter=gt_boxes.boxes[frame_token]
            prediction_bboxes_of_this_frame_before_filter=pred_boxes.boxes[frame_token]
            # gen rectified detection, rectify the z situation
            '''
            for pred_box in prediction_bboxes_of_this_frame_before_filter:
                pred_instance=formated_to_dict_det(pred_box)
                copy_translation=pred_instance['translation']
                pred_instance['translation']=[]
                
                for i in range(2):
                    pred_instance['translation'].append(copy_translation[i])
                if pred_instance['detection_name'] in classifications:
                    pred_instance['translation'].append(copy_translation[2]-difference_in_z[pred_instance['detection_name']])
                    prediction_rectified['results'][frame_token].append(pred_instance)
            '''

            if len(gt_bboxes_of_this_frame_before_filter)>0 and len(prediction_bboxes_of_this_frame_before_filter)>0:
                prediction_bboxes_of_this_frame_after_filter = filter_eval_boxes(nuscenes_data, prediction_bboxes_of_this_frame_before_filter, class_range, frame_token, verbose=True)
                gt_bboxes_of_this_frame_after_filter = filter_eval_boxes(nuscenes_data, gt_bboxes_of_this_frame_before_filter, class_range, frame_token, verbose=True)
            else:
                prediction_bboxes_of_this_frame_after_filter=prediction_bboxes_of_this_frame_before_filter
                gt_bboxes_of_this_frame_after_filter=gt_bboxes_of_this_frame_before_filter

            
            if len(gt_bboxes_of_this_frame_after_filter)>0 and len(prediction_bboxes_of_this_frame_after_filter)>0:
                gt_elevation=np.array([b.translation[2] for b in gt_bboxes_of_this_frame_after_filter])
                pred_elevation=np.array([b.translation[2] for b in prediction_bboxes_of_this_frame_after_filter])
                gt_position=np.array([b.translation[:2] for b in gt_bboxes_of_this_frame_after_filter])
                pred_position=np.array([b.translation[:2] for b in prediction_bboxes_of_this_frame_after_filter])
    
                position_distance_gt_pred=sklearn.metrics.pairwise.euclidean_distances(gt_position,pred_position)
                position_distance_gt_pred[position_distance_gt_pred >= 2] = -100
    
                position_distance_pred_gt=sklearn.metrics.pairwise.euclidean_distances(pred_position, gt_position)
                position_distance_pred_gt[position_distance_pred_gt >= 2] = -100
                number_of_detections=len(prediction_bboxes_of_this_frame_after_filter)
                number_of_gts=len(gt_bboxes_of_this_frame_after_filter)
                false_positive_detection=[i for i in range(number_of_detections)]            
                #ordered_index, matched_detection_index = linear_sum_assignment(position_distance)
                for gt_index in range(number_of_gts):
                    gt_instance_formated=gt_bboxes_of_this_frame_after_filter[gt_index]
                    gt_instance=formated_to_dict(gt_instance_formated)
                    gt_result[frame_token].append(gt_instance)
                    distance_for_this_gt=position_distance_gt_pred[gt_index]
                    gt_name=gt_bboxes_of_this_frame_after_filter[gt_index].tracking_name
                    count_for_gt_classifications[classifications_index[gt_name]]+=1
                    z_for_gt_classifications[classifications_index[gt_name]].append(gt_instance_formated.translation[2])
                    hit_detection_index=[]
                    gt_associated_with_detections=[]
    
    
                    for detection_index in range(number_of_detections):
                        if distance_for_this_gt[detection_index]!=-100 and detection_index in false_positive_detection:
                            false_positive_detection.remove(detection_index)
                            hit_detection_index.append(detection_index)
                        gt_associated_with_this_detection=[]
                        distance_for_this_detection = position_distance_pred_gt[detection_index]
                        for gt_index2 in range(number_of_gts):
                            if distance_for_this_detection[gt_index2]!=-100:
                                gt_associated_with_this_detection.append(gt_index2)
                        gt_associated_with_detections.append(gt_associated_with_this_detection)
    
                    instance_formated=gt_bboxes_of_this_frame_after_filter[gt_index]
                    instance=formated_to_dict(instance_formated)
                    if len(hit_detection_index)==0:
                        instance['matched']=False
                    else:
                        hit_for_gt_classifications[classifications_index[gt_name]]+=1
                        
                        
                        instance['matched']=True
                        instance['matched_detections']=[]
                        instance['matched_detection_types']=[]
                        instance['matched_detection_scores']=[]
                        instance['number_of_matched_detection']=len(hit_detection_index)
                        
                        for matched_detection_index in hit_detection_index:
                            prediction_bbox=prediction_bboxes_of_this_frame_after_filter[matched_detection_index]
                            
                            prediction_bbox_dict=formated_to_dict_det(prediction_bbox)
                            
                            associated_gt=gt_associated_with_detections[matched_detection_index]
                            prediction_bbox_dict['ground_truth_associated']=[]
                            for ass_gt_index in associated_gt:
                                prediction_bbox_dict['ground_truth_associated'].append(formated_to_dict(gt_bboxes_of_this_frame_after_filter[ass_gt_index]))
                            instance['matched_detections'].append(prediction_bbox_dict)
                            prediction_name=prediction_bbox.detection_name
                            prediction_score=prediction_bbox.detection_score
                            instance['matched_detection_types'].append(prediction_name)
                            instance['matched_detection_scores'].append(prediction_score)
                            register_for_gt_classifications[classifications_index[gt_name]][classifications_index[prediction_name]]+=1
    
                        if len(hit_detection_index)>1:
                            instance['multiple_hits']=True
                            correct_classification=[]
                            double_hit_for_gt_classifications[classifications_index[gt_name]]+=1
                            for matched_detection_index in hit_detection_index:
                                prediction_name=prediction_bboxes_of_this_frame_after_filter[matched_detection_index].detection_name
                                prediction_score=prediction_bboxes_of_this_frame_after_filter[matched_detection_index].detection_score
                                
                                if prediction_name==gt_name:
                                    correct_classification.append(matched_detection_index)
                                    z_for_detection_classifications[classifications_index[prediction_name]].append(prediction_bboxes_of_this_frame_after_filter[matched_detection_index].translation[2])
                            if len(correct_classification)==1:
                                double_hit_correct_classification[classifications_index[gt_name]]+=1
                            elif len(correct_classification)>1:
                                multiple_correct_detections_for_gt[classifications_index[gt_name]]+=1
                            else:
                                for matched_detection_index in hit_detection_index:
                                    prediction_name=prediction_bboxes_of_this_frame_after_filter[matched_detection_index].detection_name
                                    if prediction_name!=gt_name:
                                        double_hit_wrong_classification[classifications_index[gt_name]][classifications_index[prediction_name]]+=1
                        else:
                            instance['multiple_hits']=False
                            prediction_name=prediction_bboxes_of_this_frame_after_filter[hit_detection_index[0]].detection_name
                            if prediction_name==gt_name:
                                single_hit_correct_classification[classifications_index[gt_name]]+=1
                                z_for_detection_classifications[classifications_index[prediction_name]].append(prediction_bboxes_of_this_frame_after_filter[hit_detection_index[0]].translation[2])
                            else:
                                single_hit_wrong_classification[classifications_index[gt_name]][classifications_index[prediction_name]]+=1
                    evaluation_result[frame_token].append(instance)
                number_of_false_positives=len(false_positive_detection)         
                if number_of_false_positives>0:
                    false_positive_pred_position=np.array([prediction_bboxes_of_this_frame_after_filter[index].translation[:2] for index in false_positive_detection])
                    position_distance_pred_pred=sklearn.metrics.pairwise.euclidean_distances(false_positive_pred_position,false_positive_pred_position)
                    position_distance_pred_pred[position_distance_pred_pred >= 1] = -100
                    for position_idex, flase_positive_index in enumerate(false_positive_detection):
                        instance_formated=prediction_bboxes_of_this_frame_after_filter[flase_positive_index]
                        instance=formated_to_dict_det(instance_formated)
                        instance['flase_positive']=True
                        distances_from_this_prediction=position_distance_pred_pred[position_idex]
                        
                        cluttered_false_positive=[]
                        for false_positive_detection_index in range(number_of_false_positives):
                            if distances_from_this_prediction[false_positive_detection_index]!=-100 and false_positive_detection_index!=position_idex:
                                cluttered_false_positive.append(prediction_bboxes_of_this_frame_after_filter[false_positive_detection[false_positive_detection_index]])
    
                        instance['cluttered_false_positive']=cluttered_false_positive
                        evaluation_result[frame_token].append(instance)

    '''
    register_for_gt_classifications=[np.zeros(number_of_classifications) for i in range(number_of_classifications)]
    count_for_gt_classifications=np.zeros(number_of_classifications)
    hit_for_gt_classifications=np.zeros(number_of_classifications)
    double_hit_for_gt_classifications=np.zeros(number_of_classifications)
    single_hit_correct_classification=np.zeros(number_of_classifications)
    single_hit_wrong_classification=[np.zeros(number_of_classifications) for i in range(number_of_classifications)]
    double_hit_correct_classification=np.zeros(number_of_classifications)
    double_hit_wrong_classification=[np.zeros(number_of_classifications) for i in range(number_of_classifications)]
    multiple_correct_detections_for_gt=np.zeros(number_of_classifications)
    '''
    print('-----------------------------------------------------------------------------------------------------------------------')
    for idx, classification in enumerate(classifications):
        print('-----------------------------------------------------------------------------------------------------------------------')
        print('there are {} gt for {}'.format(count_for_gt_classifications[idx],classification))
        print('there are {} correct detections for {}'.format(hit_for_gt_classifications[idx],classification))
        print('there are {} correct single detections for {}'.format(single_hit_correct_classification[idx],classification))
        print('there are {} correct multiple detections for {}'.format(double_hit_for_gt_classifications[idx],classification))
        sum_z=np.sum(z_for_gt_classifications[idx])
        len_z=len(z_for_gt_classifications[idx])
        average_z=sum_z/len_z
        print('the average gt z for this class is {}'.format(np.round(average_z,2)))
    
        sum_z_det=np.sum(z_for_detection_classifications[idx])
        len_z_det=len(z_for_detection_classifications[idx])
        average_z_det=sum_z_det/len_z_det
        print('the average z detection for this class is {}'.format(np.round(average_z_det,2)))
        for idx2, classification2 in enumerate(classifications):
            if classification2==classification:
                pass
                #print('there are {} hit detections with correct classification'.format(register_for_gt_classifications[idx][idx]))
            else:
                percentage=np.round(register_for_gt_classifications[idx][idx2]/hit_for_gt_classifications[idx],2)
                print('there are {} of hit detections misclassified from {} to {} '.format(percentage,classification,classification2))
        print('-----------------------------------------------------------------------------------------------------------------------')

    with open('/home/bailiping/Desktop/gt_results.json', 'w') as f:
        json.dump(gt_result,f,cls=NumpyEncoder)
    with open('/home/bailiping/Desktop/evaluation_results.json', 'w') as f:
        json.dump(gt_result,f,cls=NumpyEncoder)
    with open('/home/bailiping/Desktop/rectified_val_results.json', 'w') as f:
        json.dump(prediction_rectified,f,cls=NumpyEncoder)
   
if __name__ == '__main__':
    main()