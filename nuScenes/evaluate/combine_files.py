import json
from numpyencoder import NumpyEncoder

from nuscenes import NuScenes
from utils.utils import compute_trajectory, boxes_iou_bev, initiate_submission_file_mini, create_experiment_folder, create_classification_folder, initiate_submission_file, create_scene_folder, gen_ordered_frames, gen_measurement_of_this_class, initiate_classification_submission_file

lidar_3d_object_detection_inference_result_in_nuscenes_format_file = '/home/blp/Desktop/mmdetection3d/data/nuscenes/official_inference_result/centerpoint_val_detection.json'
root_directory_for_dataset = '/home/blp/Desktop/mmdetection3d/data/nuscenes'
dataset_version = 'v1.0-trainval'

def main():
    # read the nuscenes data
    nuscenes_data = NuScenes(version=dataset_version,
                             dataroot=root_directory_for_dataset, verbose=False)
    # read out frames of this dataset
    frames = nuscenes_data.sample
    # read out inference result
    with open(lidar_3d_object_detection_inference_result_in_nuscenes_format_file, 'rb') as f:
        estimated_bboxes_data_over_all_frames_meta = json.load(f)
    estimated_bboxes_data_over_all_frames = estimated_bboxes_data_over_all_frames_meta['results']
    # add commment to this experiment
    # create a experiment folder based on the time
    out_file_directory_for_this_experiment = '/home/blp/Desktop/Radar_Perception_Project/Project_5/evaluate'

    # partition the measurements by its classification
    classifications = ['bus', 'bicycle','motorcycle',  'trailer', 'truck']
    omitted=['bus','pedestrian','car', 'bicycle','motorcycle',  'trailer', 'truck']

    submission = initiate_submission_file_mini(frames, estimated_bboxes_data_over_all_frames)
    for classification in classifications:
        with open(out_file_directory_for_this_experiment+'/{}_submission.json'.format(classification), 'r') as f:
            submission_for_this_class = json.load(f)
            result_of_this_class = submission_for_this_class['results']
            for frame_token in result_of_this_class:
                # if frame['token'] in estimated_bboxes_data_over_all_frames.keys():
                for bbox_info in result_of_this_class[frame_token]:
                    submission['results'][frame_token].append(bbox_info)
    # save the aggregate submission result
    with open('/home/blp/Desktop/Radar_Perception_Project/Project_5/evaluate/val_submission_car.json', 'w') as f:
        json.dump(submission, f, indent=4, cls=NumpyEncoder)

if __name__ == '__main__':
    main()
