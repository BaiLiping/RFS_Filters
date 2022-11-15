from logging import raiseExceptions
import os
import json
from numpyencoder import NumpyEncoder
from utils.utils import Box, predict_for_all,incorporate_track,visualize_duplicated_detection,iou3d,compute_duplicated_detection,fill_frame_result,gen_measurement_all, nms, nu_array2mot_bbox,create_scene_folder_name,create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
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
from pyquaternion import Quaternion


import json
import math
import os
import os.path as osp
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from utils.utils import get_inference_colormap
from numpyencoder import NumpyEncoder
import copy



all='bicycle+motorcycle+trailer+truck+bus+pedestrian+car'
classifications_index = {'bicycle':0,'motorcycle':1,  'trailer':2, 'truck':3,'bus':4,'pedestrian':5,'car':6}
classifications = ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_version', default='v1.0-trainval', help='choose dataset version between [v1.0-trainval][v1.0-test][v1.0-mini]')
    parser.add_argument('--detection_file',default='/home/bailiping/Desktop/MHT_train_submission.json', help='directory for the inference file')
    parser.add_argument('--programme_file', default='/home/bailiping/Desktop/MOT')
    parser.add_argument('--dataset_file', default='/media/bailiping/My Passport/mmdetection3d/data/nuscenes')
    parser.add_argument('--parallel_process', default=4)
    parser.add_argument('--render_classes', default='')
    parser.add_argument('--result_file', default='/home/bailiping/Desktop')
    parser.add_argument('--render_curves', default=False)
    parser.add_argument('--config_path',default='')
    parser.add_argument('--verbose',default=True)
    parser.add_argument('--plot_result',default=True)
    parser.add_argument('--single_thread_debug',default=False)
    args = parser.parse_args()
    return args

def gen_track_record(inference_file, dataset_version):
    dataset_info_file='/home/bailiping/mmdetection3d/data/nuscenes/configs/dataset_info.json'
    
    if dataset_version =='v1.0-trainval':
        set_info='train'
    elif dataset_version == 'v1.0-mini':
        set_info='mini_val'
    elif dataset_version == 'v1.0-test':
        set_info='test'
    else:
        raise KeyError('wrong data version')
    # read ordered frame info
    with open(dataset_info_file, 'rb') as f:
        dataset_info=json.load(f)

    orderedframe=copy.deepcopy(dataset_info[set_info]['ordered_frame_info'])

    track_record={}

    for scene_idx in range(len(list(orderedframe.keys()))):
        scene_token = list(orderedframe.keys())[scene_idx]
        ordered_frames = orderedframe[scene_token]
        track_record[scene_token]={}
        id_list=[]
        for frame_idx, frame_token in enumerate(ordered_frames):
            inference_of_this_frame=inference_file[frame_token]
            for bbox_idx, bbox in enumerate(inference_of_this_frame): 
                if bbox['tracking_id'] not in id_list:
                    # initiate the record
                    track_record[scene_token][bbox['tracking_id']]={}
                    id_list.append(bbox['tracking_id'])
                    track_record[scene_token][bbox['tracking_id']]['initial_frame']=frame_idx
                    track_record[scene_token][bbox['tracking_id']]['tracking_name']=bbox['tracking_name']
                    track_record[scene_token][bbox['tracking_id']]['record']=[]
                    track_record[scene_token][bbox['tracking_id']]['frame_record']=[]
                    track_record[scene_token][bbox['tracking_id']]['frame_record'].append(frame_idx)
                    track_record[scene_token][bbox['tracking_id']]['record'].append(bbox)
                else:
                    track_record[scene_token][bbox['tracking_id']]['frame_record'].append(frame_idx)
                    track_record[scene_token][bbox['tracking_id']]['record'].append(bbox)
        
    return track_record
class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2, linestyle: str='-') -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth, linestyle=linestyle)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth,linestyle=linestyle)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth, linestyle=linestyle)

    def copy(self) -> 'Box':
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)


def main(token, out_file_directory_for_this_experiment):
    args=parse_args()
    dataset_info_file=args.programme_file+'/configs/dataset_info.json'
    
    if args.data_version =='v1.0-trainval':
        set_info='train'
    elif args.data_version == 'v1.0-mini':
        set_info='mini_val'
    elif args.data_version == 'v1.0-test':
        set_info='test'
    else:
        raise KeyError('wrong data version')
    # read ordered frame info
    with open(dataset_info_file, 'rb') as f:
        dataset_info=json.load(f)

    with open('/home/bailiping/Desktop/train_submission.json', 'rb') as f:
        result_meta=json.load(f)
    tracking_result_all=result_meta['results']
    inference_track_record=gen_track_record(tracking_result_all, args.data_version)
    orderedframe=dataset_info[set_info]['ordered_frame_info']
    egoposition=dataset_info[set_info]['ego_position_info']
    sensor_calibration_data=dataset_info[set_info]['sensor_calibration_info']



    # get ground track record
    with open('/home/bailiping/Desktop/train_gt_track_record.json', 'rb') as f:
        gt_track_record=json.load(f)



    for scene_idx in range(len(list(orderedframe.keys()))):
        if args.single_thread_debug==False:
            if scene_idx % args.parallel_process != token:
                continue
        scene_token = list(orderedframe.keys())[scene_idx]
        plot=False
        ordered_frames = orderedframe[scene_token]
        gt_track_record_of_the_scene=gt_track_record[scene_token]
        for frame_idx, frame_token in enumerate(ordered_frames):
            ego_position_of_this_frame=egoposition[scene_token][str(frame_idx)]
            sensor_calibration_data_of_this_frame=sensor_calibration_data[scene_token][str(frame_idx)]

            # go through the record for the first time
            for track_id in list(inference_track_record[scene_token].keys()):
                initial_frame=inference_track_record[scene_token][track_id]['initial_frame']
                terminal_frame=inference_track_record[scene_token][track_id]['frame_record'][-1]
                if initial_frame==frame_idx:
                    target=inference_track_record[scene_token][track_id]['record'][0]
                    target_box = Box(target['translation'],target['size'],Quaternion(target['rotation']))
                    target_box.translate(-np.array(ego_position_of_this_frame[:3]))
                    target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
                    center_bottom=target_box.center 
                    inference_track_record[scene_token][track_id]['position_record']=[]
                    inference_track_record[scene_token][track_id]['position_record'].append(center_bottom)
                else:
                    if frame_idx>initial_frame and frame_idx<=terminal_frame:
                        if frame_idx in inference_track_record[scene_token][track_id]['frame_record']:
                            position_idx=inference_track_record[scene_token][track_id]['frame_record'].index(frame_idx)
                            target=inference_track_record[scene_token][track_id]['record'][position_idx]
                            target_box = Box(target['translation'],target['size'],Quaternion(target['rotation']))
                            target_box.translate(-np.array(ego_position_of_this_frame[:3]))
                            target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
                            center_bottom=target_box.center
                            inference_track_record[scene_token][track_id]['position_record'].append(center_bottom) 

        # go through the record second time to intepolate between frames
        for frame_idx, frame_token in enumerate(ordered_frames):
            ego_position_of_this_frame=egoposition[scene_token][str(frame_idx)]
            sensor_calibration_data_of_this_frame=sensor_calibration_data[scene_token][str(frame_idx)]
            for track_id in list(inference_track_record[scene_token].keys()):
                track=inference_track_record[scene_token][track_id]
                initial_frame=track['initial_frame']
                terminal_frame=track['frame_record'][-1]
                all_frames=[]
                all_positions=[]
                all_frames_detection_flag=[]
                all_record=[]
        
                for i_index, i in enumerate(track['frame_record']):
                    last_frame_index=i
                    all_positions.append(track['position_record'][i_index])
                    all_frames_detection_flag.append(True)
                    all_frames.append(i)
                    all_record.append(track['record'][i_index])
                    if i_index+1<len(track['frame_record']):
                        next_frame_index=track['frame_record'][i_index+1] 
                        position_difference=next_frame_index-last_frame_index
                        if position_difference>1:
                            next_position=track['position_record'][i_index+1]
                            last_position=track['position_record'][i_index]
                            delta_x=(next_position[0]-last_position[0])/position_difference
                            delta_y=(next_position[1]-last_position[1])/position_difference
                            
                            for p in range(position_difference-1):
                                all_positions.append([last_position[0]+delta_x*(p+1), last_position[1]+delta_y*(p+1),0])
                                all_frames_detection_flag.append(False)
                                all_frames.append(i+p+1)
                                all_record.append(track['record'][i_index])

                        all_positions.append(track['position_record'][i_index+1])
                        all_frames_detection_flag.append(True)
                        all_frames.append(last_frame_index)
                        all_record.append(track['record'][i_index+1])


                inference_track_record[scene_token][track_id]['all_frames']=all_frames
                inference_track_record[scene_token][track_id]['all_frames_detection_flag']=all_frames_detection_flag
                inference_track_record[scene_token][track_id]['all_positions']=all_positions
                inference_track_record[scene_token][track_id]['all_record']=all_record
        
        #plot the tracks
        for frame_idx, frame_token in enumerate(ordered_frames):
            ego_position_of_this_frame=egoposition[scene_token][str(frame_idx)]
            sensor_calibration_data_of_this_frame=sensor_calibration_data[scene_token][str(frame_idx)]
            out_file_directory_for_this_scene=create_scene_folder_name(scene_token, out_file_directory_for_this_experiment)   
            directory=out_file_directory_for_this_scene+'/{}.png'.format(frame_idx)
            _, ax = plt.subplots(1, 1, figsize=(15, 15))
        
            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')
         
            for track_id in list(inference_track_record[scene_token].keys()):
                track=inference_track_record[scene_token][track_id]
                c= np.array(get_inference_colormap(track['tracking_name']))/255.0
                initial_frame=track['initial_frame']
                terminal_frame=track['frame_record'][-1]
        
                if initial_frame==terminal_frame:
                    # plot clutter
                    if initial_frame==frame_idx:
                        position=track['position_record'][0]
                        ax.scatter(position[0],position[1],color='k', s=50, alpha=0.5)
                elif frame_idx in track['all_frames']:
                    # when the track is alive
                    position_index=track['all_frames'].index(frame_idx)
                    if position_index==0:
                        target=track['record'][0]
                        box_translation=track['position_record'][0]
                        target_box = Box(box_translation,target['size'],Quaternion(target['rotation']))
                        target_box.render(ax, view=np.eye(4), colors=(c,c,c))
                    else:
                        for i in range(position_index+1)[1:]:
                            actual_frame=track['all_frames'][i]
                            detected_flag=track['all_frames_detection_flag'][i]
                            actual_position=track['all_positions'][i]
                            previous_actual_frame=track['all_frames'][i-1]
                            previous_detected_flag=track['all_frames_detection_flag'][i-1]
                            previous_actual_position=track['all_positions'][i-1]
                            if detected_flag==True:
                                ax.arrow(previous_actual_position[0], previous_actual_position[1], actual_position[0]-previous_actual_position[0], actual_position[1]-previous_actual_position[1], color=c)
                                if i == position_index:
                                    #target=track['record'][track['frame_record'].index(actual_frame)]
                                    target=track['all_record'][i]
                                    box_translation=track['position_record'][track['frame_record'].index(actual_frame)]
                                    target_box = Box(box_translation,target['size'],Quaternion(target['rotation']))
                                    target_box.render(ax, view=np.eye(4), colors=(c,c,c))

                            else:
                                ax.arrow(previous_actual_position[0], previous_actual_position[1], actual_position[0]-previous_actual_position[0], actual_position[1]-previous_actual_position[1], color=c,linestyle=':')
                                if i == position_index:
                                    #for p_index, p in enumerate(track['frame_record']):
                                    #    if p_index>0 and p_index<len(track['frame_record'])-1:
                                    #        if i>p and i<track['frame_record'][p_index+1]:
                                    #            target=track['record'][p_index]
                                    #target=track['record'][track['frame_record'].index(actual_frame)]
                                    target=track['all_record'][i]
                                    box_translation=actual_position
                                    target_box = Box(box_translation,target['size'],Quaternion(target['rotation']))
                                    target_box.render(ax, view=np.eye(4), colors=(c,c,c), linewidth=2,linestyle=':')


                elif frame_idx>terminal_frame:
                    # when the track is alive
                    position_index=len(track['all_frames'])-1

                    for i in range(position_index+1)[1:]:
                        actual_frame=track['all_frames'][i]
                        detected_flag=track['all_frames_detection_flag'][i]
                        actual_position=track['all_positions'][i]
                        previous_actual_frame=track['all_frames'][i-1]
                        previous_detected_flag=track['all_frames_detection_flag'][i-1]
                        previous_actual_position=track['all_positions'][i-1]
                        if detected_flag==True:
                            ax.arrow(previous_actual_position[0], previous_actual_position[1], actual_position[0]-previous_actual_position[0], actual_position[1]-previous_actual_position[1], color=c)
                            if i == position_index:
                                target=track['record'][track['frame_record'].index(actual_frame)]
                                box_translation=track['position_record'][track['frame_record'].index(actual_frame)]
                                ax.text(box_translation[0], box_translation[1],'T', fontsize = 7, color=c)
                        else:
                            ax.arrow(previous_actual_position[0], previous_actual_position[1], actual_position[0]-previous_actual_position[0], actual_position[1]-previous_actual_position[1], color=c,linestyle=':')
                            if i == position_index:
                                box_translation=actual_position
                                ax.text(box_translation[0], box_translation[1],'T', fontsize = 7, color=c)        

            # plot ground truth
            for track_id in list(gt_track_record_of_the_scene.keys()):
                track=gt_track_record_of_the_scene[track_id]
                initial_frame=track['initial_frame']
                c= np.array(get_inference_colormap(track['tracking_name']))/255.0
                terminal_frame=track['frame_record'][-1]
                if initial_frame==frame_idx:
                    target=track['record'][0]
                    #target_box = Box([target[0], target[1], target[2]],[target[3], target[4], target[5]],Quaternion([target[6], target[7], target[8], target[9]]))
                    target_box = Box(target['translation'],target['size'],Quaternion(target['rotation']))
                    target_box.translate(-np.array(ego_position_of_this_frame[:3]))
                    target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
                    center_bottom=target_box.center 
                    ax.scatter(center_bottom[0],center_bottom[1],color=c, s=15)
                    track['position_record']=[]
                    track['position_record'].append(center_bottom)
                else:
                    if frame_idx>initial_frame and frame_idx<=terminal_frame:
                        if frame_idx in track['frame_record']:
                            position_idx=track['frame_record'].index(frame_idx)
                            target=track['record'][position_idx]
                            #target_box = Box([target[0], target[1], target[2]],[target[3], target[4], target[5]],Quaternion([target[6], target[7], target[8], target[9]]))
                            target_box = Box(target['translation'],target['size'],Quaternion(target['rotation']))
                            
                            target_box.translate(-np.array(ego_position_of_this_frame[:3]))
                            target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
                            center_bottom=target_box.center
                            track['position_record'].append(center_bottom) 
                            ax.scatter(center_bottom[0],center_bottom[1],color=c, s=5)
                            for j in range(len(track['position_record'])-1):
                                ax.arrow(track['position_record'][j][0],track['position_record'][j][1],track['position_record'][j+1][0]-track['position_record'][j][0],track['position_record'][j+1][1]-track['position_record'][j][1], color=c, linewidth=5, alpha=0.1)
                    if frame_idx>terminal_frame:
                        for j in range(len(track['position_record'])-1):
                            ax.arrow(track['position_record'][j][0],track['position_record'][j][1],track['position_record'][j+1][0]-track['position_record'][j][0],track['position_record'][j+1][1]-track['position_record'][j][1], color=c, linewidth=5, alpha=0.1)
                    

            
            # Limit visible range.
            limit=70
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
         
            plt.savefig(directory, bbox_inches='tight', pad_inches=0, dpi=200)
            print("generating {}".format(directory)) 
            plt.close()       

    #render_tracker_result_of_a_scene(ordered_frames, ground_truth_bboxes[scene_token], ground_truth_type[scene_token], tracking_result_all,egoposition[scene_token], sensor_calibration_data[scene_token],out_file_directory_for_this_scene)
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
    