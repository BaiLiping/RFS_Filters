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
from utils.utils import get_inference_colormap, Box
from numpyencoder import NumpyEncoder
import copy

def render_tracker_result(last_frame_index_pointer, gt_track_record_of_the_scene, frame_idx, inference_track_record_of_the_scene,ego_record, sensor_calibration_data_of_this_frame,out_directory):
    _, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='red')
    '''
    # go through the record for the first time
    for track_id in list(inference_track_record_of_the_scene.keys()):
        initial_frame=inference_track_record_of_the_scene[track_id]['initial_frame']
        terminal_frame=inference_track_record_of_the_scene[track_id]['frame_record'][-1]
        if initial_frame==frame_idx:
            target=inference_track_record_of_the_scene[track_id]['record'][0]
            target_box = Box(target['translation'],target['size'],Quaternion(target['rotation']))
            target_box.translate(-np.array(ego_record[:3]))
            target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
            center_bottom=target_box.center 
            inference_track_record_of_the_scene[track_id]['position_record']=[]
            inference_track_record_of_the_scene[track_id]['position_record'].append(center_bottom)
        else:
            if frame_idx>initial_frame and frame_idx<=terminal_frame:
                if frame_idx in inference_track_record_of_the_scene[track_id]['frame_record']:
                    position_idx=inference_track_record_of_the_scene[track_id]['frame_record'].index(frame_idx)
                    target=inference_track_record_of_the_scene[track_id]['record'][position_idx]
                    target_box = Box(target['translation'],target['size'],Quaternion(target['rotation']))
                    target_box.translate(-np.array(ego_record[:3]))
                    target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
                    center_bottom=target_box.center
                    inference_track_record_of_the_scene[track_id]['position_record'].append(center_bottom)

    
    # go through the record second time to plot the thing
    for track_id in list(inference_track_record_of_the_scene.keys()):
        track=inference_track_record_of_the_scene[track_id]
        initial_frame=track['initial_frame']
        terminal_frame=track['frame_record'][-1]
        hidden_frame=[]
        hidden_frame_position=[]

        if terminal_frame>len(track['frame_record'])-1:

            for i_index, i in enumerate(track['frame_record']):
                last_frame_index=i
                if i_index+1<len(track['frame_record']):
                    next_frame_index=track['frame_record'][i_index+1] 
                    position_difference=next_frame_index-last_frame_index
                    next_position=track['position_record'][track['frame_record'].index(next_frame_index)]
                    last_position=track['position_record'][track['frame_record'].index(last_frame_index)]
                    delta_x=next_position[0]-last_position[0]/position_difference
                    delta_y=next_position[1]-last_position[1]/position_difference
                    
                    for p in range(position_difference-1):
                        hidden_frame_position.append([last_position[0]+delta_x*(p+1), last_position[1]+delta_y*(p+1)])
                        hidden_frame.append(i+p+1)
            
            inference_track_record_of_the_scene[track_id]['hidden_frame']=hidden_frame
            inference_track_record_of_the_scene[track_id]['hidden_frame_position']=hidden_frame_position

    '''
    for track_id in list(inference_track_record_of_the_scene.keys()):
        track=inference_track_record_of_the_scene[track_id]
        initial_frame=track['initial_frame']
        terminal_frame=track['frame_record'][-1]

        if initial_frame==terminal_frame:
            if initial_frame==frame_idx:
                center_bottom=track['position_record'][0]
                ax.scatter(center_bottom[0],center_bottom[1],color='k')
        else:
            if initial_frame==frame_idx:
                target=track['record'][0]
                last_frame_index_pointer=frame_idx
                #target_box = Box([target[0], target[1], target[2]],[target[3], target[4], target[5]],Quaternion([target[6], target[7], target[8], target[9]]))
                new_translation=track['position_record'][0]
                target_box = Box(new_translation,target['size'],Quaternion(target['rotation']))
                c= np.array(get_inference_colormap(target['tracking_name']))/255.0
                target_box.render(ax, view=np.eye(4), colors=(c,c,c))
            else:
                if frame_idx>initial_frame and frame_idx<=terminal_frame:
                    if frame_idx in track['frame_record']:
                        last_frame_index_pointer=frame_idx
                        position_idx=track['frame_record'].index(frame_idx)
                        target=track['record'][position_idx]
                        new_translation=track['position_record'][position_idx]
                        target_box = Box(new_translation,target['size'],Quaternion(target['rotation']))
                        c= np.array(get_inference_colormap(target['tracking_name']))/255.0
                        target_box.render(ax, view=np.eye(4), colors=(c,c,c))
                        for j in range(position_idx):
                            ax.arrow(track['position_record'][j][0],track['position_record'][j][1],track['position_record'][j+1][0]-track['position_record'][j][0],track['position_record'][j+1][1]-track['position_record'][j][1], color=c)
                      
                    # plot the hidden state with dashed lines
                    else:
                        position_difference=frame_idx-last_frame_index_pointer
                        new_hidden_frame_position=[]

                        
                        new_hidden_frame_position.append(track['hidden_frame_position'][position_difference-1][0])
                        new_hidden_frame_position.append(track['hidden_frame_position'][position_difference-1][1])
                        new_hidden_frame_position.append(0)

                        position_idx=track['frame_record'].index(last_frame_index_pointer)
                        target=track['record'][position_idx]
                        c= np.array(get_inference_colormap(target['tracking_name']))/255.0
                        target_box=Box(new_hidden_frame_position,target['size'],Quaternion(target['rotation']) )
                        target_box.render(ax, view=np.eye(4), colors=(c,c,c), linewidth=1)
                        for j in range(position_idx):
                            ax.arrow(track['position_record'][j][0],track['position_record'][j][1],track['position_record'][j+1][0]-track['position_record'][j][0],track['position_record'][j+1][1]-track['position_record'][j][1], color=c)
                        for z in range(position_difference):
                            ax.arrow(track['hidden_frame_position'][position_idx+z][0],track['hidden_frame_position'][position_idx+z][1],track['hidden_frame_position'][position_idx+z+1][0]-track['hidden_frame_position'][position_idx+z][0],track['hidden_frame_position'][position_idx+z+1][1]-track['hidden_frame_position'][position_idx+z][1], color=c)
        
    # plot ground truth
    for track_id in list(gt_track_record_of_the_scene.keys()):
        track=gt_track_record_of_the_scene[track_id]
        initial_frame=track['initial_frame']
        terminal_frame=track['frame_record'][-1]
        if initial_frame==frame_idx:
            target=track['record'][0]
            #target_box = Box([target[0], target[1], target[2]],[target[3], target[4], target[5]],Quaternion([target[6], target[7], target[8], target[9]]))
            target_box = Box(target['translation'],target['size'],Quaternion(target['rotation']))
            c= np.array(get_inference_colormap(target['tracking_name']))/255.0
            target_box.translate(-np.array(ego_record[:3]))
            target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
            center_bottom=target_box.center 
            ax.scatter(center_bottom[0],center_bottom[1],color=c)
            track['position_record']=[]
            track['position_record'].append(center_bottom)
        else:
            if frame_idx>initial_frame and frame_idx<=terminal_frame:
                if frame_idx in track['frame_record']:
                    position_idx=track['frame_record'].index(frame_idx)
                    target=track['record'][position_idx]
                    #target_box = Box([target[0], target[1], target[2]],[target[3], target[4], target[5]],Quaternion([target[6], target[7], target[8], target[9]]))
                    target_box = Box(target['translation'],target['size'],Quaternion(target['rotation']))
                    c= np.array(get_inference_colormap(target['tracking_name']))/255.0
                    target_box.translate(-np.array(ego_record[:3]))
                    target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
                    center_bottom=target_box.center
                    track['position_record'].append(center_bottom) 
                    ax.scatter(center_bottom[0],center_bottom[1],color=c)
                    for j in range(len(track['position_record'])-1):
                        ax.arrow(track['position_record'][j][0],track['position_record'][j][1],track['position_record'][j+1][0]-track['position_record'][j][0],track['position_record'][j+1][1]-track['position_record'][j][1], color=c, alpha=0.2)
    
    # Limit visible range.
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
 
    plt.savefig(out_directory, bbox_inches='tight', pad_inches=0, dpi=200)

    return last_frame_index_pointer