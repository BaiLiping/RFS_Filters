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
from utils.utils import nms,readout_parameters, create_experiment_folder, gen_ordered_frames, gen_measurement_of_this_class, initiate_classification_submission_file
import copy
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

from trackers.GM_PHD import GM_PHD_Filter_Point_Target as phd_tracker
from trackers.GM_PHD import util as phd_ulti
from datetime import datetime
from evaluate.util.utils import TrackingConfig, config_factory
from evaluate.evaluate_tracking_result import TrackingEval


detection_file = '/media/bailiping/My Passport/mmdetection3d/data/nuscenes/official_inference_result/centerpoint_val.json'
dataset_file = '/media/bailiping/My Passport/mmdetection3d/data/nuscenes'
data_version = 'v1.0-trainval'


now = datetime.now()
formatedtime = now.strftime("%Y-%m-%d-%H-%M-%S")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_file',
                        default=detection_file, help='the path towards the inference_result_in_nuscenes_format_file_path')
    parser.add_argument('--type_of_tracker', default='pmbm_tracker',
                        help='the tracker we want to run with estimated bboxes outputed from Lidar 3d object detector')
    parser.add_argument('--number_of_experiments', default=1,
                        help='the number of experiments')
    parser.add_argument('--data_version', default=data_version,
                        help='v1.0-mini, v1.0-trainval or v1.0-test')
    parser.add_argument('--programme_file', default='/home/bailiping/Desktop/MOT')
    parser.add_argument('--bayesian_filter_config',
                        default='Linear Kalman Filter')
    parser.add_argument('--motion_model_type', default='constant velocity')
    parser.add_argument('--dataset_file', default=dataset_file,
                        help='root directory for the entire dataset, including mini, trainval and test')
    parser.add_argument('--bird_eye_view_with_map', default=True,
                        help='generate bird eye view with cropped map, otherwise it would just be a blank canvas')
    parser.add_argument('--distance', default='Euclidean distance',
                        help='the method for distance computation')
    parser.add_argument('--time', default=formatedtime,
                        help='log the result into a folder named after current time')
    parser.add_argument('--comment', default='phd')
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/bailiping/Desktop')
    parser.add_argument('--render_curves', default='False')
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default='True')
    args = parser.parse_args()
    return args
    
    
def main():
    args = parse_args()
    dataset_info=args.programme_file+'/configs/dataset_info.json'
    config=args.programme_file+'/configs/phd_parameters.json'
    # read the nuscenes data
    nuscenes_data = NuScenes(version=args.data_version,
                             dataroot=args.dataset_file, verbose=False)
    # read out scenes of this dataset
    scenes = nuscenes_data.scene
    # read out frames of this dataset
    frames = nuscenes_data.sample
    # read out inference result
    with open(args.detection_file, 'rb') as f:
        estimated_bboxes_data_over_all_frames_meta = json.load(f)
    estimated_bboxes_data_over_all_frames=estimated_bboxes_data_over_all_frames_meta['results']
    # add commment to this experiment
    comment = args.comment
    # create a experiment folder based on the time
    out_file_directory_for_this_experiment = create_experiment_folder(
        args.result_file, args.time, comment)
    # partition the measurements by its classification
    classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']
    # read parameters
    with open(config, 'r') as f:
        parameters=json.load(f)

    for classification_idx, classification in enumerate(classifications):
        # initiate submission file
        classification_submission = initiate_classification_submission_file(classification)
        
        # readout parameters
        birth_rate, P_s, P_d, _,average_number_of_clutter_per_frame_class, _, extraction_threshold, _, _, _, detection_score_thr, nms_score, _, _ = readout_parameters(classification, parameters)

        # adjust detection_scrore_thr for pointpillars
        # this step is necessary because the pointpillar detections for cars and pedestrian would generate excessive number of bounding boxes and result in unnecessary compuations.
        if args.detection_file[21:]=='pointpillars_val.json':
            if classification=='car' or classification == 'pedestrian':
                detection_score_thr = 0.2
            else:
                detection_score_thr = 0
        else:
            detection_score_thr=0
        # generate filter model based on the classification
        filter_model = phd_ulti.gen_filter_model(classification, average_number_of_clutter_per_frame_class, P_d, P_s,extraction_threshold)
        for scene in scenes:
            ordered_frames = gen_ordered_frames(scene, frames)
            start = time.process_time()
            # generate filter based on the filter model
            phd_filter = phd_tracker.GM_PHD_Filter(filter_model, 'constant velocity', 'Linear Kalman Filter')
            filter_pruned_and_merged = {}
            filter_pruned_and_merged['w'] = []
            for frame_idx, frame in enumerate(ordered_frames):
                # initiate the result for this frame

                # read out information pertaining to this frame
                lidar_record = nuscenes_data.get(
                    'sample_data', frame['data']['LIDAR_TOP'])
                pose_record = nuscenes_data.get(
                    'ego_pose', lidar_record['ego_pose_token'])
                cs_record = nuscenes_data.get(
                    'calibrated_sensor', lidar_record['calibrated_sensor_token'])
                filter_pruned = {}
                filter_pruned['meanPois'] = []
                if frame['token'] in estimated_bboxes_data_over_all_frames.keys():
                    # get measurements at global coordinate
                    estimated_bboxes_at_current_frame = estimated_bboxes_data_over_all_frames[
                        frame['token']]
                    classification_submission['results'][frame['token']] = []
                    if len(estimated_bboxes_at_current_frame) == 0:
                        print('the {} frame has zero detections'.format(frame_idx))
                    else:
                        Z_k_all = gen_measurement_of_this_class(detection_score_thr, estimated_bboxes_at_current_frame, classification)
                        # preprocessing of input. Remove bboxes that overlaps
                        result_indexes = nms(Z_k_all, threshold=nms_score)
                        Z_k=[]
                        for idx in result_indexes:
                            Z_k.append(Z_k_all[idx])
                        if len(Z_k) > 0:
                            if len(filter_pruned_and_merged['w']) == 0:  # For the fisrt frame, there are only new birth targets rather than surviving targets thus we call seperate function.
                                # the initial step the labmda for weight update is w_birthinit instead of w_birthsum
                                filter_predicted = phd_filter.predict_for_initial_step(Z_k, birth_rate)
                            else:
                                filter_predicted = phd_filter.predict(Z_k,filter_pruned_and_merged, birth_rate)
            
                            filter_updated = phd_filter.update(Z_k, filter_predicted) #Eq. 20 of [2]
                            filter_pruned_and_merged = phd_filter.pruneAndMerge(filter_updated)
    
                            estimatedStates_for_this_classification = phd_filter.extractStates(filter_pruned_and_merged)
                            # fill the submission file for this frame
                            if len(estimatedStates_for_this_classification['mean']) != 0:
                                for idx in range(len(estimatedStates_for_this_classification['mean'])):
                                    instance_info = {}
                                    instance_info['sample_token'] = frame['token']
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
                                    #instance_info['tracking_score'] = estimatedStates_for_this_classification['w'][idx]
                                    classification_submission['results'][frame['token']].append(
                                        instance_info)

            end = time.process_time()
            print("This is %s %s PHD processing takes %f seconds" %
                  (scene['name'], classification, (end - start)))
        # save the result for this classification
        with open(out_file_directory_for_this_experiment+'/{}_submission.json'.format(classification), 'w') as f:
            json.dump(classification_submission, f, indent=4, cls=NumpyEncoder)
    # aggregate results for all classifications
    submission={}
    submission['meta']={}
    submission['meta']["use_camera"]=False
    submission['meta']["use_lidar"]=True
    submission['meta']["use_radar"]=False
    submission['meta']["use_map"]=False
    submission['meta']["use_external"]=False
    submission['results']={}

    for classification in classifications:
        with open(out_file_directory_for_this_experiment+'/{}_submission.json'.format(classification), 'r') as f:
            submission_for_this_class = json.load(f)
            result_of_this_class = submission_for_this_class['results']
            for frame_token in result_of_this_class:
                if frame_token in submission['results'].keys():
                    for bbox_info in result_of_this_class[frame_token]:
                        submission['results'][frame_token].append(bbox_info)
                else:
                    submission['results'][frame_token]=[]
                    for bbox_info in result_of_this_class[frame_token]:
                        submission['results'][frame_token].append(bbox_info)
    # save the aggregate submission result
    with open(out_file_directory_for_this_experiment+'/val_submission.json', 'w') as f:
        json.dump(submission, f, indent=4, cls=NumpyEncoder)

    result_path_ = os.path.expanduser(out_file_directory_for_this_experiment+'/val_submission.json')
    output_dir_ = os.path.expanduser(out_file_directory_for_this_experiment+'/nuscenes-metrics')
    eval_set_ = 'val'
    dataroot_ = args.dataset_file
    version_ = args.data_version
    config_path = args.config_path
    render_curves_ = args.render_curves
    verbose_ = args.verbose
    render_classes_ = args.render_classes

    if config_path == '':
        cfg_ = config_factory(args.programme_file+'/configs/tracking_config.json')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialize(json.load(_f))
    nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                             nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                             render_classes=render_classes_)
    nusc_eval.visualization_and_evaluation_of_tracking_results(render_curves=render_curves_)

if __name__ == '__main__':
    main()
