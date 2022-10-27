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


def render_tracker_result(ground_truth, ground_truth_type_for_this_frame, Z_k,ego_record, sensor_calibration_data_of_this_frame,out_directory):
    _, ax = plt.subplots(1, 1, figsize=(12, 12))

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
        c= np.array(get_inference_colormap(target['tracking_name']))/255.0
        target_box_record.append(target_box)
        target_box.render(ax, view=np.eye(4), colors=(c,c,c))
        position1=2
        position2=-3
        text='{}'.format(target['tracking_id'])
        if target['tracking_name']=='pedestrian':
            ax.text(center_bottom[0], center_bottom[1],np.round(target['tracking_score'],2) , fontsize = 15, color=c)
        elif target['tracking_name']=='truck':
            ax.text(center_bottom[0], center_bottom[1]+position1,text , fontsize = 15, color=c)
            ax.text(center_bottom[0], center_bottom[1]-position2,np.round(target['tracking_score'],2) , fontsize = 15, color=c)
        else:
            ax.text(center_bottom[0], center_bottom[1]-position1,text , fontsize = 15, color=c)
            ax.text(center_bottom[0], center_bottom[1]+position2,np.round(target['tracking_score'],2) , fontsize = 15, color=c)

    
    for idx, target in enumerate(ground_truth):
        target_box = Box([target[0], target[1], target[2]],[target[3], target[4], target[5]],Quaternion([target[6], target[7], target[8], target[9]]))
        c= np.array(get_inference_colormap(ground_truth_type_for_this_frame[idx]))/255.0
        target_box.translate(-np.array(ego_record[:3]))
        target_box.translate(-np.array(sensor_calibration_data_of_this_frame[:3])) 
        center_bottom=target_box.center 
        ax.scatter(center_bottom[0],center_bottom[1],color=c)
  
    # Limit visible range.
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
 
    plt.savefig(out_directory, bbox_inches='tight', pad_inches=0, dpi=200)