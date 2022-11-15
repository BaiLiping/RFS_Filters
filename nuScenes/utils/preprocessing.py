'''
preprocessing dataset to generate aggregated information about each frame. dataset_info.json would be generated at configs.
the dataset_info.json file contains information such as ordered frame tokens and it would be processed by the filters.
'''

import os, numpy as np, nuscenes, argparse, json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from numpyencoder import NumpyEncoder

from copy import deepcopy
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask


tracking_mapping = {
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'}




detection_mapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}

def instance_info2bbox_array(info):
    translation = info.center.tolist()
    size = info.wlh.tolist()
    rotation = info.orientation.q.tolist()
    return translation + size + rotation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--programme_file', default='/home/bailiping/Desktop/MOT')
    parser.add_argument('--dataset_folder',default='/media/bailiping/My Passport/mmdetection3d/data/nuscenes')
    args = parser.parse_args()
    return args

def main(args):
    radar_positions=['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']
    set_infos=['val','train','test','mini_val']
    aggregated_info={}
    for set_info in set_infos:
        if set_info=='val' or set_info=='train':
            data_version ='v1.0-trainval'
        elif set_info=='mini_val':
            data_version = 'v1.0-mini'
        elif set_info=='test':
            data_version='v1.0-test'
        nusc = NuScenes(version=data_version, dataroot=args.dataset_folder, verbose=True)
        scenes=nusc.scene
        scene_names = splits.create_splits_scenes()[set_info]
        pbar = tqdm(total=len(scene_names))

        
        aggregated_info[set_info]={}
        # ordered frame info
        aggregated_info[set_info]['ordered_frame_info']={}   
        # time stamp info
        aggregated_info[set_info]['time_stamp_info']={}
        # ego info
        aggregated_info[set_info]['ego_position_info']={}
        aggregated_info[set_info]['cs_info']={}
        aggregated_info[set_info]['radar_points']={}
        # sensor calibration info
        aggregated_info[set_info]['sensor_calibration_info']={}
        # ground truth info
        aggregated_info[set_info]['ground_truth_IDS']={}
        aggregated_info[set_info]['ground_truth_inst_types']={}
        aggregated_info[set_info]['ground_truth_bboxes']={}
        #sensor token information
        aggregated_info[set_info]['lidar_token']={}
        aggregated_info[set_info]['radar_token']={}
        for radar_position in radar_positions:
            aggregated_info[set_info]['radar_token'][radar_position]={}
        
        #sensor calibration information
        aggregated_info[set_info]['lidar_calibration_token']={}
        aggregated_info[set_info]['radar_calibration_token']={}
        for radar_position in radar_positions:
            aggregated_info[set_info]['radar_calibration_token'][radar_position]={}
        #sensor path information
        aggregated_info[set_info]['lidar_path']={}
        aggregated_info[set_info]['radar_path']={}
        for radar_position in radar_positions:
            aggregated_info[set_info]['radar_path'][radar_position]={}

        for scene_index, scene_info in enumerate(scenes):
            scene_name = scene_info['name']
            scene_token = scene_info['token']
            if scene_name not in scene_names:
                continue
            # ordered frame info
            aggregated_info[set_info]['ordered_frame_info'][scene_token]=[]
            
            # time stamp info
            aggregated_info[set_info]['time_stamp_info'][scene_token]=[]

            # time stamp info
            aggregated_info[set_info]['radar_points'][scene_token]={}
    
            # ego info
            aggregated_info[set_info]['ego_position_info'][scene_token]={}
            # sensor calibration info
            aggregated_info[set_info]['sensor_calibration_info'][scene_token]={}
            # ground truth info
            aggregated_info[set_info]['ground_truth_IDS'][scene_token]={}
            aggregated_info[set_info]['ground_truth_inst_types'][scene_token]={}
            aggregated_info[set_info]['ground_truth_bboxes'][scene_token]={}

            #sensor token information
            aggregated_info[set_info]['lidar_token'][scene_token]=[]
            aggregated_info[set_info]['radar_token'][scene_token]={}
            for radar_position in radar_positions:
                aggregated_info[set_info]['radar_token'][radar_position][scene_token]=[]
            
            #sensor calibration information
            aggregated_info[set_info]['lidar_calibration_token'][scene_token]=[]
            aggregated_info[set_info]['radar_calibration_token'][scene_token]={}
            for radar_position in radar_positions:
                aggregated_info[set_info]['radar_calibration_token'][radar_position][scene_token]=[]
            
            
            #sensor path information
            aggregated_info[set_info]['lidar_path'][scene_token]=[]
            aggregated_info[set_info]['radar_path'][scene_token]={}
            for radar_position in radar_positions:
                aggregated_info[set_info]['radar_path'][radar_position][scene_token]=[]
    
    
            first_sample_token = scene_info['first_sample_token']
            last_sample_token = scene_info['last_sample_token']
            frame_data = nusc.get('sample', first_sample_token)
            cur_sample_token = deepcopy(first_sample_token)
            
            frame_index = 0

            while True:
                aggregated_info[set_info]['ground_truth_IDS'][scene_token][str(frame_index)]=[]
                aggregated_info[set_info]['ground_truth_inst_types'][scene_token][str(frame_index)]=[]
                aggregated_info[set_info]['ground_truth_bboxes'][scene_token][str(frame_index)]=[]

                # find the path to lidar data
                frame_data = nusc.get('sample', cur_sample_token)
                aggregated_info[set_info]['ordered_frame_info'][scene_token].append(cur_sample_token)
                aggregated_info[set_info]['time_stamp_info'][scene_token].append(frame_data['timestamp'])
                
                lidar_token = frame_data['data']['LIDAR_TOP']
                lidar_data = nusc.get('sample_data', lidar_token)
                calib_token = lidar_data['calibrated_sensor_token']
                calib_pose = nusc.get('calibrated_sensor', calib_token)
                lidar_path = lidar_data['filename']
                
                ego_token = lidar_data['ego_pose_token']
                ego_pose = nusc.get('ego_pose', ego_token)

                aggregated_info[set_info]['lidar_token'][scene_token].append(lidar_token)
                aggregated_info[set_info]['lidar_path'][scene_token].append(lidar_path)
                aggregated_info[set_info]['lidar_calibration_token'][scene_token].append(calib_token)
                aggregated_info[set_info]['sensor_calibration_info'][scene_token][str(frame_index)] = calib_pose['translation'] + calib_pose['rotation']
                aggregated_info[set_info]['ego_position_info'][scene_token][str(frame_index)] = ego_pose['translation'] + ego_pose['rotation']

                if data_version == 'v1.0-trainval' or data_version=='v1.0-mini':
                    #instances = nusc.get_boxes(lidar_token)
                    sample_annotation_tokens = frame_data['anns']  
                    for sample_annotation_token in sample_annotation_tokens:
                        sample_annotation = nusc.get('sample_annotation', sample_annotation_token)

                        
                        if sample_annotation['category_name'] in tracking_mapping.keys():
                            aggregated_info[set_info]['ground_truth_inst_types'][scene_token][str(frame_index)].append(tracking_mapping[sample_annotation['category_name']])
                            aggregated_info[set_info]['ground_truth_IDS'][scene_token][str(frame_index)].append(sample_annotation['instance_token'])
                            aggregated_info[set_info]['ground_truth_bboxes'][scene_token][str(frame_index)].append(sample_annotation['translation'] + sample_annotation['size'] + sample_annotation['rotation'])

                # clean up and prepare for the next
                cur_sample_token = frame_data['next']

                if cur_sample_token == '':
                    break
                frame_index += 1
            #with open(args.programme_file+'/configs/dataset_info_with_radar_{}.json'.format(scene_token), 'w') as f:
            #    json.dump(aggregated_info, f, cls=NumpyEncoder)
            #f.close()
            pbar.update(1)
        pbar.close()
    
    with open(args.programme_file+'/configs/dataset_info.json', 'w') as f:
        json.dump(aggregated_info, f, cls=NumpyEncoder)
    f.close()

if __name__ == '__main__':
    args=parse_args()

    main(args)
