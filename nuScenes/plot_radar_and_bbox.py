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

from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from utils.utils import get_inference_colormap


def visualize_radar_and_bbox(Z_k,radar_points_of_this_frame,sensor_calibration_data_of_this_frame, ego_record,frame_idx,classification):
    viewpoint = np.eye(4)  
    fig, ax = plt.subplots(1,figsize=(12,12))
    #ax=axes[0]
    position=np.array(radar_points_of_this_frame)[:3, :]

    position_rotated=np.dot(Quaternion(ego_record[3:]).rotation_matrix, position)
    position_rotated=np.dot(Quaternion(sensor_calibration_data_of_this_frame[3:]).rotation_matrix, position_rotated)
    points = view_points(position_rotated, viewpoint, normalize=False)

    velocity=np.array(radar_points_of_this_frame)[8:10, :]
    velocity_rotated=np.dot(Quaternion(ego_record[3:]).rotation_matrix, [velocity[0],velocity[1],0])
    velocity_rotated=np.dot(Quaternion(sensor_calibration_data_of_this_frame[3:]).rotation_matrix, velocity_rotated)


    #dists = np.sqrt(np.sum(np.array(position_rotated[:2,:]) ** 2, axis=0))
    #colors = np.minimum(1, dists / 60 / np.sqrt(2))
    colors='k'
    point_scale = 3
    scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
    # Show velocities.
    #points_vel = view_points(position_rotated + velocity_rotated, viewpoint, normalize=False)
    deltas_vel = velocity_rotated
    #deltas_vel = 6 * deltas_vel  # Arbitrary scaling
    #max_delta = 20
    #deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
    #colors_rgba = scatter.to_rgba(colors)
    for i in range(points.shape[1]):
        ax.arrow(points[0, i], points[1, i], 6*deltas_vel[0][i], 6*deltas_vel[1][i], color='k')
    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='red')
    target_box_record=[]

    # Show bounding boxes
    for idx, target in enumerate(Z_k):
        translation = target['translation']
        size=target['size']
        rotation=Quaternion(target['rotation'])
        target_box = Box(translation,size,rotation)
        
      
        target_box.translate(-np.array(ego_record[:3]))
        
        target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3]))  
        #target_box.rotate(Quaternion(ego_record[3:]).inverse)
        #target_box.rotate(Quaternion(sensor_calibration_data_of_this_frame[3:]).inverse)
        center_bottom=target_box.center
        c= np.array(get_inference_colormap(target['detection_name']))/255.0
        target_box_record.append(target_box)
        target_box.render(ax, view=np.eye(4), colors=(c,c,c))
        position1=2
        position2=-3
        if target['detection_name']=='pedestrian':
            #ax.text(center_bottom[0], center_bottom[1]+position1,text , fontsize = 7, color=c)
            ax.text(center_bottom[0], center_bottom[1],np.round(target['detection_score'],2) , fontsize = 15, color=c)
        elif target['detection_name']=='truck':
            ax.text(center_bottom[0], center_bottom[1]-position2,np.round(target['detection_score'],2) , fontsize = 15, color=c)
        else:
            ax.text(center_bottom[0], center_bottom[1]+position2,np.round(target['detection_score'],2) , fontsize = 15, color=c)
        #velocity= np.dot(Quaternion(ego_record[3:]).rotation_matrix, [target['velocity'][0],target['velocity'][0],0])        
        #velocity=np.dot(Quaternion(sensor_calibration_data_of_this_frame[3:]).rotation_matrix, velocity)
        #ax.arrow(center_bottom[0], center_bottom[1], 6*velocity[0], 6*velocity[1], color='red')
        #ax.text(center_bottom[0]+3, center_bottom[1],np.round(target['velocity'][0],2) , fontsize = 15, color=c)
        #ax.text(center_bottom[0]-3, center_bottom[1],np.round(target['velocity'][1],2) , fontsize = 15, color=c)
        #ax.scatter(center_bottom[0], center_bottom[1],color=c)
        # the following part is only applicable to PHD filter
        #ax.text(target[0][0], target[1][0]-2, '{:.2f}'.format(estimated_target_position['w'][idx]) , fontsize = 20)
    # Limit visible range.
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
 
    plt.savefig('/home/bailiping/Desktop/radar/{}_{}.png'.format(classification, frame_idx), bbox_inches='tight', pad_inches=0, dpi=200)

def points_in_box(box: 'Box', points: np.ndarray, wlh_factor: float = 1.0):
    """
    Checks whether points are inside the box.
    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask