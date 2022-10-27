'''
the utility functions to support the filters
for specifics, read the function note.
In general, it includes the customized class datastructure, initializating function etc.
'''

import os
import copy
import shutil
from tokenize import Number
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
import os, numpy as np, json
import math
import numpy as np
from copy import deepcopy
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numba
from scipy.spatial import ConvexHull
import numpy as np
from os import path
import glob
from tqdm import tqdm
import pickle
from copy import deepcopy
import json


import copy
import os.path as osp
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict

import cv2
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, create_lidarseg_legend
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import view_points, transform_matrix


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
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)

    def render_cv2(self,
                   im: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 2) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im,
                         (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])),
                         color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(im,
                     (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                     colors[2][::-1], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(im,
                 (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                 colors[0][::-1], linewidth)

    def copy(self) -> 'Box':
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)


def compute_birth_rate(Z_k, classification):
    base_rate={'bicycle': 0.001, 'car':0.1, 'truck':0.1, 'motorcycle':0.01, 'trailer': 0.1, 'bus':0.1, 'pedestrian':0.01}
    threshold_distance={'bicycle': 10, 'car':10, 'truck':10,'motorcycle': 10, 'trailer':10, 'bus':10, 'pedestrian':5}
    number_of_objects_max={'bicycle': 3, 'car':35, 'truck':25, 'motorcycle':3, 'trailer': 4, 'bus':3, 'pedestrian':30}
    number_of_objects_min={'bicycle': 2, 'car':0, 'truck':0, 'motorcycle':2, 'trailer': 2, 'bus':2, 'pedestrian':1}


    crowd_birth_rate=[]
    distance=np.zeros((len(Z_k), len(Z_k)))
    for index_1, z_1 in enumerate(Z_k):
        for index_2, z_2 in enumerate(Z_k): 
            euclidean_distance=np.sqrt(np.power(z_1['translation'][0]-z_2['translation'][0],2)+np.power(z_1['translation'][1]-z_2['translation'][1],2))
            if index_2!=index_1:
                distance[index_1][index_2]=euclidean_distance
    for index_3, z_3 in enumerate(Z_k):
        distance_log=distance[index_3]
        within_range=np.where(np.array(distance_log)<=threshold_distance[classification])
        number_of_measurements_nearby=len(within_range[0])
        if number_of_measurements_nearby>number_of_objects_max[classification]:
            birth_rate=base_rate[classification]*10
        elif number_of_measurements_nearby>number_of_objects_min[classification] and number_of_measurements_nearby<=number_of_objects_max[classification]:
            birth_rate=base_rate[classification]
        else:
            birth_rate=base_rate[classification]*0.1
        crowd_birth_rate.append(birth_rate)

    return crowd_birth_rate 

def compute_birth_rate_location(Z_k,egoposition, classification):
    base_rate={'bicycle': 0.001, 'car':0.1, 'truck':0.1, 'motorcycle':0.01, 'trailer': 0.1, 'bus':0.1, 'pedestrian':0.01}
    max_distance={'bicycle': 20, 'car':20, 'truck':20, 'motorcycle':20, 'trailer': 20, 'bus':30, 'pedestrian':0}
    min_distance={'bicycle': 10, 'car':10, 'truck':10, 'motorcycle':10, 'trailer': 10, 'bus':20, 'pedestrian':0}


    distance_birth_rate=[]
    distance=np.zeros(len(Z_k))
    for index_1, z_1 in enumerate(Z_k):
        euclidean_distance=np.sqrt(np.power(egoposition[0]-z_1['translation'][0],2)+np.power(egoposition[1]-z_1['translation'][1],2))
    
        if euclidean_distance>max_distance[classification]:
            birth_rate=base_rate[classification]
        elif euclidean_distance>min_distance[classification] and euclidean_distance<=max_distance[classification]:
            birth_rate=base_rate[classification]*0.1
        else:
            birth_rate=base_rate[classification]*0.01
        distance_birth_rate.append(birth_rate)

    return distance_birth_rate 

def compute_duplicated_detection(Z_k):
    skip_z_index=[]
    Z_k_new=[]
    for index_1, z_1 in enumerate(Z_k):
        if index_1 not in skip_z_index: 
            duplicated_detection=[]
            for index_2, z_2 in enumerate(Z_k): 
                twodiou_result, threediou_result=iou3d(nu_array2mot_bbox(z_1), nu_array2mot_bbox(z_2))
                if threediou_result>0.5 and index_2!=index_1:
                    skip_z_index.append(index_2)
                    duplicated_detection.append(copy.deepcopy(z_2))
                    #print('{} and {} are duplicated detection'.format(z_1['detection_name'],z_2['detection_name']))
            z_1['duplicated_detection']=duplicated_detection
            Z_k_new.append(z_1)
    return Z_k_new 

def fuse_duplicated_track(Z_k):
    Z_k_out=[]
    for z in Z_k:
        if len(z['duplicated_detection'])>0:
            probabilities=[]
            classes=[]
            probabilities.append(z['detection_score'])
            classes.append(z['detection_name'])
            for z_1 in z['duplicated_detection']:
                probabilities.append(z['detection_score'])
                classes.append(z['detection_name'])
            highest_index=np.argmax(probabilities)
            average_score=np.sum(probabilities)/len(probabilities)
            # replace the detection score with average_Score
            z['detection_score']=average_score
            z['detection_name']=classes[highest_index]

        Z_k_out.append(z)
    return Z_k_out


def compute_duplicated_detection_2d(Z_k):
    skip_z_index=[]
    Z_k_new=[]
    for index_1, z_1 in enumerate(Z_k):
        if index_1 not in skip_z_index: 
            duplicated_detection=[]
            for index_2, z_2 in enumerate(Z_k): 
                twodiou_result, threediou_result=iou3d(nu_array2mot_bbox(z_1), nu_array2mot_bbox(z_2))
                if threediou_result>0.5 and index_2!=index_1:
                    skip_z_index.append(index_2)
                    duplicated_detection.append(copy.deepcopy(z_2))
                    #print('{} and {} are duplicated detection'.format(z_1['detection_name'],z_2['detection_name']))
            
            z_1['duplicated_detection']=duplicated_detection
            Z_k_new.append(z_1)
    return Z_k_new   

def compute_overlapping_score_to_gt(Z_k, ground_truth):
    overlapping_score_twod=np.zeros((len(Z_k), len(ground_truth)))
    overlapping_score_threed=np.zeros((len(Z_k), len(ground_truth)))
    for z_index, z in enumerate(Z_k):
        for gt_index, gt in enumerate(ground_truth):
            twodiou_result, threediou_result=iou3d(nu_array2mot_bbox(z), nu_array2mot_bbox(gt))
            overlapping_score_twod[z_index][gt_index]=twodiou_result
            overlapping_score_threed[z_index][gt_index]=threediou_result
    return overlapping_score_twod,overlapping_score_threed

def visualize_duplicated_detection(Z_k):
    duplicated_z=[]
    for idx, z in enumerate(Z_k):
        if len(z['duplicated_detection'])>0:
            duplicated_z.append(z)
            for z1 in z['duplicated_detection']:
                duplicated_z.append(z1)
    return duplicated_z

classifications_index = {'bicycle':0,'motorcycle':1,  'trailer':2, 'truck':3,'bus':4,'pedestrian':5,'car':6}

def fill_frame_result(frame_result,estimation, frame_token):
    for idx in range(len(estimation['mean'])):
        instance_info = {}
        instance_info['sample_token'] = frame_token
        translation_of_this_target = [estimation['mean'][idx][0][0],
                                      estimation['mean'][idx][1][0], estimation['elevation'][idx]]
        instance_info['translation'] = translation_of_this_target
        instance_info['size'] = estimation['size'][idx]
        instance_info['rotation'] = estimation['rotation'][idx]
        instance_info['velocity'] = [estimation['mean']
                                     [idx][2][0], estimation['mean'][idx][3][0]]
        #instance_info['tracking_id'] = estimation['classification'][idx]+'_'+str(
        #    estimation['id'][idx])
        instance_info['tracking_id'] = str(classifications_index[estimation['classification'][idx]])+'_'+str(estimation['id'][idx])
        #instance_info['tracking_id']=str(estimation['id'][idx])
        instance_info['tracking_name'] = estimation['classification'][idx]
        instance_info['tracking_score']=estimation['detection_score'][idx]
        #results_all['results'][frame_token].append(instance_info)
        frame_result.append(instance_info)
    return frame_result 


def gen_measurement_all(estimated_bboxes_at_current_frame):
        
    # read parameters
    with open('/home/zhubinglab/Desktop/nuScenes_Tracker/configs/pmbmgnn_parameters.json', 'r') as f:
        parameters=json.load(f)
    Z_k=[]
    for classification in ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']:
        birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classification, parameters)
         
        for box_index, box in enumerate(estimated_bboxes_at_current_frame):
            '''
            if classification=='pedestrian':
                if box['detection_name']==classification:
                    if box['translation'][2]<1.5:
                        Z_k.append(box)
                        
            else:
                if box['detection_name']==classification and box['detection_score']>detection_score_thr:
                    if box['translation'][2]<3:
                        Z_k.append(box)
            '''
            if box['detection_name']==classification and box['detection_score']>detection_score_thr:
                Z_k.append(box)
            
    return Z_k






def compute_box_3d(obj):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.yaw)    

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.X
    corners_3d[1,:] = corners_3d[1,:] + obj.Y
    corners_3d[2,:] = corners_3d[2,:] + obj.Z
    # print('cornsers_3d: ', corners_3d)

    return np.transpose(corners_3d)

def box3doverlap(aa, bb, criterion='union'):
	aa_3d = compute_box_3d(aa)
	bb_3d = compute_box_3d(bb)

	iou3d, iou2d = box3d_iou(aa_3d, bb_3d, criterion=criterion)
	# print(iou3d)
	# print(iou2d)
	return iou3d

class BBox:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, o=None):
        self.x = x      # center x
        self.y = y      # center y
        self.z = z      # center z
        self.h = h      # height
        self.w = w      # width
        self.l = l      # length
        self.o = o      # orientation
        self.s = None   # detection score
    
    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.o, self.l, self.w, self.h, self.s)
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.o}
    
    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def array2bbox(cls, data):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @classmethod
    def dict2bbox(cls, data):
        bbox = BBox()
        bbox.x = data['center_x']
        bbox.y = data['center_y']
        bbox.z = data['center_z']
        bbox.h = data['height']
        bbox.w = data['width']
        bbox.l = data['length']
        bbox.o = data['heading']
        if 'score' in data.keys():
            bbox.s = data['score']
        return bbox
    
    @classmethod
    def copy_bbox(cls, bboxa, bboxb):
        bboxa.x = bboxb.x
        bboxa.y = bboxb.y
        bboxa.z = bboxb.z
        bboxa.l = bboxb.l
        bboxa.w = bboxb.w
        bboxa.h = bboxb.h
        bboxa.o = bboxb.o
        bboxa.s = bboxb.s
        return
    
    @classmethod
    def box2corners2d(cls, bbox):
        """ the coordinates for bottom corners
        """
        bottom_center = np.array([bbox.x, bbox.y, bbox.z - bbox.h / 2])
        cos, sin = np.cos(bbox.o), np.sin(bbox.o)
        pc0 = np.array([bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc1 = np.array([bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1
    
        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]
    
    @classmethod
    def box2corners3d(cls, bbox):
        """ the coordinates for bottom corners
        """
        center = np.array([bbox.x, bbox.y, bbox.z])
        bottom_corners = np.array(BBox.box2corners2d(bbox))
        up_corners = 2 * center - bottom_corners
        corners = np.concatenate([up_corners, bottom_corners], axis=0)
        return corners.tolist()
    
    @classmethod
    def motion2bbox(cls, bbox, motion):
        result = deepcopy(bbox)
        result.x += motion[0]
        result.y += motion[1]
        result.z += motion[2]
        result.o += motion[3]
        return result
    
    @classmethod
    def set_bbox_size(cls, bbox, size_array):
        result = deepcopy(bbox)
        result.l, result.w, result.h = size_array
        return result
    
    @classmethod
    def set_bbox_with_states(cls, prev_bbox, state_array):
        prev_array = BBox.bbox2array(prev_bbox)
        prev_array[:4] += state_array[:4]
        prev_array[4:] = state_array[4:]
        bbox = BBox.array2bbox(prev_array)
        return bbox 
    
    @classmethod
    def box_pts2world(cls, ego_matrix, pcs):
        new_pcs = np.concatenate((pcs,
                                  np.ones(pcs.shape[0])[:, np.newaxis]),
                                  axis=1)
        new_pcs = ego_matrix @ new_pcs.T
        new_pcs = new_pcs.T[:, :3]
        return new_pcs
    
    @classmethod
    def edge2yaw(cls, center, edge):
        vec = edge - center
        yaw = np.arccos(vec[0] / np.linalg.norm(vec))
        if vec[1] < 0:
            yaw = -yaw
        return yaw
    
    @classmethod
    def bbox2world(cls, ego_matrix, box):
        # center and corners
        corners = np.array(BBox.box2corners2d(box))
        center = BBox.bbox2array(box)[:3][np.newaxis, :]
        center = BBox.box_pts2world(ego_matrix, center)[0]
        corners = BBox.box_pts2world(ego_matrix, corners)
        # heading
        edge_mid_point = (corners[0] + corners[1]) / 2
        yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])
        
        result = deepcopy(box)
        result.x, result.y, result.z = center
        result.o = yaw
        return result
__all__ = ['pc_in_box', 'downsample', 'pc_in_box_2D',
           'apply_motion_to_points', 'make_transformation_matrix',
           'iou2d', 'iou3d', 'pc2world', 'giou2d', 'giou3d', 
           'back_step_det', 'm_distance', 'velo2world', 'score_rectification']


def velo2world(ego_matrix, velo):
    """ transform local velocity [x, y] to global
    """
    new_velo = velo[:, np.newaxis]
    new_velo = ego_matrix[:2, :2] @ new_velo
    return new_velo[:, 0]


def apply_motion_to_points(points, motion, pre_move=0):
    transformation_matrix = make_transformation_matrix(motion)
    points = deepcopy(points)
    points = points + pre_move
    new_points = np.concatenate((points,
                                 np.ones(points.shape[0])[:, np.newaxis]),
                                 axis=1)

    new_points = transformation_matrix @ new_points.T
    new_points = new_points.T[:, :3]
    new_points -= pre_move
    return new_points


@numba.njit
def downsample(points, voxel_size=0.05):
    sample_dict = dict()
    for i in range(points.shape[0]):
        point_coord = np.floor(points[i] / voxel_size)
        sample_dict[(int(point_coord[0]), int(point_coord[1]), int(point_coord[2]))] = True
    res = np.zeros((len(sample_dict), 3), dtype=np.float32)
    idx = 0
    for k, v in sample_dict.items():
        res[idx, 0] = k[0] * voxel_size + voxel_size / 2
        res[idx, 1] = k[1] * voxel_size + voxel_size / 2
        res[idx, 2] = k[2] * voxel_size + voxel_size / 2
        idx += 1
    return res

def pc_in_box(box, pc, box_scaling=1.5):
    center_x, center_y, length, width = \
        box.x, box.y, box.l, box.w
    center_z, height = box.z, box.h
    yaw = box.o
    return pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.5):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)
        rz = np.abs(pc[i, 2] - center_z)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2) and rz < (height * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def pc_in_box_2D(box, pc, box_scaling=1.0):
    center_x, center_y, length, width = \
        box.x, box.y, box.l, box.w
    center_z, height = box.z, box.h
    yaw = box.o
    return pc_in_box_2D_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_2D_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.0):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def make_transformation_matrix(motion):
    x, y, z, theta = motion
    transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                                      [np.sin(theta),  np.cos(theta), 0, y],
                                      [0            ,  0            , 1, z],
                                      [0            ,  0            , 0, 1]])
    return transformation_matrix


def iou2d(box_a, box_b):
    boxa_corners = np.array(BBox.box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap = reca.intersection(recb).area
    area_a = reca.area
    area_b = recb.area
    iou = overlap / (area_a + area_b - overlap + 1e-10)
    return iou


def iou3d(box_a, box_b):
    boxa_corners = np.array(BBox.box2corners2d(box_a))
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap_area = reca.intersection(recb).area
    iou_2d = overlap_area / (reca.area + recb.area - overlap_area)

    ha, hb = box_a.h, box_b.h
    za, zb = box_a.z, box_b.z
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    overlap_volume = overlap_area * overlap_height
    union_volume = box_a.w * box_a.l * ha + box_b.w * box_b.l * hb - overlap_volume
    iou_3d = overlap_volume / (union_volume + 1e-5)

    return iou_2d,iou_3d


def pc2world(ego_matrix, pcs):
    new_pcs = np.concatenate((pcs,
                              np.ones(pcs.shape[0])[:, np.newaxis]),
                              axis=1)
    new_pcs = ego_matrix @ new_pcs.T
    new_pcs = new_pcs.T[:, :3]
    return new_pcs


def giou2d(box_a: BBox, box_b: BBox):
    boxa_corners = np.array(BBox.box2corners2d(box_a))
    boxb_corners = np.array(BBox.box2corners2d(box_b))
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    
    # compute intersection and union
    I = reca.intersection(recb).area
    U = box_a.w * box_a.l + box_b.w * box_b.l - I

    # compute the convex area
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area

    # compute giou
    return I / U - (C - U) / C
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

def nu_array2mot_bbox(b):
    translation=b['translation']
    size=b['size']
    rotation=b['rotation']

    nu_box = Box(translation, size, Quaternion(rotation))
    mot_bbox = BBox(
        x=nu_box.center[0], y=nu_box.center[1], z=nu_box.center[2],
        w=nu_box.wlh[0], l=nu_box.wlh[1], h=nu_box.wlh[2],
        o=nu_box.orientation.yaw_pitch_roll[0]
    )
    if 'detection_score' in b.keys():
        mot_bbox.s = b['detection_score']
    if 'tracking_score' in b.keys():
        mot_bbox.s = b['tracking_score']
    return mot_bbox

def giou3d(z, track):
    '''
    first need to convert z and track to BBox format
    '''
    z=nu_array2mot_bbox(z)
    track=nu_array2mot_bbox(track)

    boxa_corners = np.array(BBox.box2corners2d(z))[:, :2]
    boxb_corners = np.array(BBox.box2corners2d(track))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    ha, hb = z.h, track.h
    za, zb = z.z, track.z
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    union_height = max((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2))
    
    # compute intersection and union
    I = reca.intersection(recb).area * overlap_height
    U = z.w * z.l * ha + track.w * track.l * hb - I

    # compute the convex area
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area * union_height

    # compute giou
    giou = I / U - (C - U) / C
    return giou


def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area


def back_step_det(det: BBox, velo, time_lag):
    result = BBox()
    BBox.copy_bbox(result, det)
    result.x -= (time_lag * velo[0])
    result.y -= (time_lag * velo[1])
    return result


def diff_orientation_correction(diff):
    """
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    """
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    return diff


def m_distance(det, trk, trk_inv_innovation_matrix=None):
    det_array = BBox.bbox2array(det)[:7]
    trk_array = BBox.bbox2array(trk)[:7]
    
    diff = np.expand_dims(det_array - trk_array, axis=1)
    corrected_yaw_diff = diff_orientation_correction(diff[3])
    diff[3] = corrected_yaw_diff

    if trk_inv_innovation_matrix is not None:
        result = \
            np.sqrt(np.matmul(np.matmul(diff.T, trk_inv_innovation_matrix), diff)[0][0])
    else:
        result = np.sqrt(np.dot(diff.T, diff))
    return result


def score_rectification(dets, gts):
    """ rectify the scores of detections according to their 3d iou with gts
    """
    result = deepcopy(dets)
    
    if len(gts) == 0:
        for i, _ in enumerate(dets):
            result[i].s = 0.0
        return result

    if len(dets) == 0:
        return result

    iou_matrix = np.zeros((len(dets), len(gts)))
    for i, d in enumerate(dets):
        for j, g in enumerate(gts):
            iou_matrix[i, j] = iou3d(d, g)[1]
    max_index = np.argmax(iou_matrix, axis=1)
    max_iou = np.max(iou_matrix, axis=1)
    index = list(reversed(sorted(range(len(dets)), key=lambda k:max_iou[k])))

    matched_gt = []
    for i in index:
        if max_iou[i] >= 0.1 and max_index[i] not in matched_gt:
            result[i].s = max_iou[i]
            matched_gt.append(max_index[i])
        elif max_iou[i] >= 0.1 and max_index[i] in matched_gt:
            result[i].s = 0.2
        else:
            result[i].s = 0.05

    return result


# Video Generating function
def imagetovideo(image_path, num_images, video_path):
    image_folder = image_path # make sure to use your folder
    images = [img for img in os.listdir(image_folder)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    # set the flip per second varible
    fps = 2
    # setting the frame width, height width
    height, width, layers = frame.shape  
    # save video as mp4
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height)) 
  
    # Appending the images to the video one by one
    for i in range(num_images):
        video.write(cv2.imread(os.path.join(image_folder, '{}.png'.format(i)))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated

# generate video from the readout data
def generate_video(log_token, out_file_directory_for_this_log):
    # readout the images from input data
    # generate the input path for this log file
    num_of_images = len(os.listdir(out_file_directory_for_this_log))
    video_path = out_file_directory_for_this_log+'/{}.mp4'.format(log_token)
    imagetovideo(out_file_directory_for_this_log, num_of_images,video_path)


# extract data from the database
def generate_visualization(nuscenes_data, root_directory_for_out_path):
    '''
    nuscenes_data is the database readout, it is a NuScenes object as defined by data_extraction file
    root_directory_for_out_path is the input parameter from configure file
    '''
    # read out scenes of this dataset
    scenes=nuscenes_data.scene
    # read out frames of this dataset
    frames=nuscenes_data.sample

    # read out time stamp information from inference data
    
    for scene in scenes:
        # get the token of this scene
        scene_token=scene['token']
        # get the name of this scene
        scene_name=scene['name']
        # generate the out file directory for this scene
        out_file_directory_for_this_scene = os.path.join(root_directory_for_out_path,scene_name)
        # if the directory exist, then delete and make a new directory
        if os.path.exists(out_file_directory_for_this_scene):
            print('erasing existing data')
            shutil.rmtree(out_file_directory_for_this_scene, ignore_errors=True)
        os.mkdir(out_file_directory_for_this_scene)
        
        # get all the frames associated with this scene
        frames_for_this_scene = []
        for frame in frames:
            if frame['scene_token']==scene_token:
                frames_for_this_scene.append(frame)
        
        # set the frames in corret order
        # notice that NuScenes database does not provide the numerical ordering of the frames
        # it provides previous and next frame token information
        unordered_frames = copy.deepcopy(frames_for_this_scene)
        ordered_frames=[]
        # looping until the unordered frame is an empty set
        while len(unordered_frames)!=0:
            for current_frame in unordered_frames:
                # if it is the first frame
                if current_frame['prev']=='':
                    ordered_frames.append(current_frame)
                    # set current token
                    current_frame_token_of_current_scene = current_frame['token']
                    unordered_frames.remove(current_frame)
        
                # find the next frame of current frame
                if current_frame['prev']==current_frame_token_of_current_scene:
                    ordered_frames.append(current_frame)
                    # reset current frame
                    current_frame_token_of_current_scene=current_frame['token']
                    unordered_frames.remove(current_frame)

        # get the data from ordered frame list
        for idx in range(len(ordered_frames)):
            # notice this is a customized function, so please do not use code from the official dev-kit   
            nuscenes_data.render_sample(ordered_frames[idx]['token'],out_path=out_file_directory_for_this_scene+'/{}.png'.format(idx),verbose=False)        
            plt.close('all')
    
# extract data from the database
def generate_inference_visualization(nuscenes_data,inference_result, nsweeps, root_directory_for_out_path):
    '''
    nuscenes_data is the database readout, it is a NuScenes object as defined by data_extraction file
    root_directory_for_out_path is the input parameter from configure file
    '''
    # read out scenes of this dataset
    scenes=nuscenes_data.scene
    # read out frames of this dataset
    frames=nuscenes_data.sample
    log=nuscenes_data.log
    for log_file in log:
        log_token = log_file['token']
        out_file_directory_for_this_log = os.path.join(root_directory_for_out_path,log_token)
        # if the directory exist, then delete and make a new directory
        if os.path.exists(out_file_directory_for_this_log):
            print('Erasing existing data for log {}'.format(log_token))
            shutil.rmtree(out_file_directory_for_this_log, ignore_errors=True)
        os.mkdir(out_file_directory_for_this_log)

    # read out time stamp information from inference data
    for scene in scenes:
        # get the token of this scene
        scene_token=scene['token']
        # get the log token
        scene_log_token = scene['log_token']
        out_file_directory_for_this_log = os.path.join(root_directory_for_out_path,scene_log_token)
        # generate the out file directory for this scene

        # get all the frames associated with this scene
        frames_for_this_scene = []
        for frame in frames:
            if frame['scene_token']==scene_token:
                frames_for_this_scene.append(frame)
        
        # set the frames in corret order
        # notice that NuScenes database does not provide the numerical ordering of the frames
        # it provides previous and next frame token information
        unordered_frames = copy.deepcopy(frames_for_this_scene)
        ordered_frames=[]
        # looping until the unordered frame is an empty set
        while len(unordered_frames)!=0:
            for current_frame in unordered_frames:
                # if it is the first frame
                if current_frame['prev']=='':
                    ordered_frames.append(current_frame)
                    # set current token
                    current_frame_token_of_current_scene = current_frame['token']
                    unordered_frames.remove(current_frame)
        
                # find the next frame of current frame
                if current_frame['prev']==current_frame_token_of_current_scene:
                    ordered_frames.append(current_frame)
                    # reset current frame
                    current_frame_token_of_current_scene=current_frame['token']
                    unordered_frames.remove(current_frame)

        num_of_images = len(os.listdir(out_file_directory_for_this_log))

        # get the data from ordered frame list
        for idx, frame in enumerate(ordered_frames):
            # notice this is a customized function, so please do not use code from the official dev-kit   
            nuscenes_data.render_inference_sample(inference_result,frame['token'],nsweeps=nsweeps,out_path=out_file_directory_for_this_log+'/{}.png'.format(num_of_images+idx),verbose=False)        
            plt.close('all')
    
    for log_file in log:
        log_token = log_file['token']
        out_file_directory_for_this_log = os.path.join(root_directory_for_out_path,log_token)
        generate_video(log_token,out_file_directory_for_this_log)


def get_inference_colormap(class_name):
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """
    classname_to_color = {  # RGB.
        "car": (0, 0, 230),  # Blue
        "truck": (70, 130, 180),  # Steelblue
        "trailer": (138, 43, 226),  # Blueviolet
        "bus":(0,255,0),  # lime
        "construction_vehicle": (255,255,0),  # Gold
        "bicycle": (0, 175, 0),  # Green
        "motorcycle": (0, 0, 128),  # Navy,
        "pedestrian":(255, 69, 0),  # Orangered.
        "traffic_cone": (255,0,255), #magenta
        "barrier": (173,255,47),  # greenyello
    }

    return classname_to_color[class_name]


def boxes_iou_bev(box_a, box_b):
    translation_of_box_a=[box_a['translation'][0],box_a['translation'][1],box_a['translation'][2]]
    size_of_box_a=box_a['size']
    rotation_of_box_a=Quaternion(box_a['rotation'])
    box_a_in_Box_format = Box(translation_of_box_a,size_of_box_a,rotation_of_box_a)

    translation_of_box_b=[box_b['translation'][0],box_b['translation'][1],box_b['translation'][2]]
    size_of_box_b=box_b['size']
    rotation_of_box_b=Quaternion(box_b['rotation'])
    box_b_in_Box_format = Box(translation_of_box_b,size_of_box_b,rotation_of_box_b)

    x1=box_a_in_Box_format.center[0]
    y1=box_a_in_Box_format.center[1]

    w1=box_a_in_Box_format.wlh[0]
    l1=box_a_in_Box_format.wlh[1]
    
    x2=box_b_in_Box_format.center[0]
    y2=box_b_in_Box_format.center[1]

    w2=box_b_in_Box_format.wlh[0]
    l2=box_b_in_Box_format.wlh[1]


    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1+w1/2, x2+w2/2)
    yB = min(y1+l1/2, y2+l2/2)

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs(w1 * l1)
    boxBArea = abs(w2 * l2)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def compute_trajectory(frame_index, distance, estimates_previous_frame, estimates_this_frame):
    '''
    The strategy for assigning a track id is this:
    1. match the estimations of this frame to estimations of previous frame. The criterium for matching is the definition of distance.
    2. if there are unmatched current estimation, then give it a new id tag
    3. if there are unmatched previous estimation, terminate that id tag
    4. the id tag is stored as the key associated with each estimation
    '''
    T=0.5
    I = T*np.eye(2, dtype=np.float64)
    F=np.eye(4, dtype=np.float64)
    F[0:2, 2:4] = I
    sigma_v = 1     # Standard deviation of the process noise.
    Q1 = np.array([[T ** 4 / 4, T ** 3 / 2], [T ** 3 / 2, T ** 2]], dtype=np.float64)
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    Q = sigma_v ** 2 * Q  # Covariance of process noise
    
    if len(estimates_previous_frame['mean'])==0:
        return estimates_this_frame
    else:
        num_previous_estimation = len(estimates_previous_frame['mean'])
        #max_id_of_previous_frame = estimates_previous_frame['maximum_id']
        num_current_estimation = len(estimates_this_frame['mean'])
        #if num_previous_estimation == 0: # All the tracks are false tracks
        #    estimates_this_frame['tag']=[]
        #    estimates_this_frame['maximum_id']=max_id_of_previous_frame
        #    for i in range(num_current_estimation):
        #        # initiate the tag id
        #        estimates_this_frame['tag'].append(i)
        #    estimates_this_frame['maximum_id']=num_current_estimation-1
        #elif num_current_estimation == 0: # All the targets are missed
            # terminate all the tags
        #    estimates_this_frame['tag']=[]
        #    if 'maximum_id' in estimates_previous_frame.keys():
        #        # the maxiumu tag id remain the same
        #        estimates_this_frame['maximum_id']=estimates_previous_frame['maximum_id']
        #    else:
        #        estimates_this_frame['maximum_id']=0
        #else: # There are elements in both sets. Compute cost matrix
        cost_matrix = np.zeros((num_previous_estimation, num_current_estimation))
        for n_previous in range(num_previous_estimation):
            for n_current in range(num_current_estimation):
                if distance=='Euclidean distance':
                    predicted_position=F.dot(estimates_previous_frame['mean'][n_previous])
                    current_cost = (predicted_position[0]-estimates_this_frame['mean'][n_current][0])**2+(predicted_position[1]-estimates_this_frame['mean'][n_current][1])**2
                    cost_matrix[n_previous,n_current] = np.min([current_cost, 20])
            # use the linear sum assignment algorithm to get the best assignment option
            previous_frame_assignment, current_frame_assignment = linear_sum_assignment(cost_matrix)
            # succession of tag for matched previous and current estimations
            previous_to_current_assigments = dict()
            current_to_previous_assignments=dict()
            #initiate tag list
            estimates_this_frame['id']=[-1 for x in range(num_current_estimation)]
            for previous_idx, current_idx in zip(previous_frame_assignment, current_frame_assignment):
                if cost_matrix[previous_idx, current_idx] < 20:
                    previous_to_current_assigments[previous_idx] = current_idx
                    current_to_previous_assignments[current_idx]=previous_idx
                    estimates_this_frame['id'][current_idx]=estimates_previous_frame['id'][previous_idx]
                    estimates_this_frame['classification'][current_idx]=estimates_previous_frame['classification'][previous_idx]
            # initiate new id
            estimates_this_frame['max_id']=estimates_previous_frame['max_id']
            previous_max=estimates_previous_frame['max_id']
            max=previous_max
            for current_index in range(len(estimates_this_frame['mean'])):
                if current_index not in current_to_previous_assignments:
                    max+=1
                    estimates_this_frame['id'][current_index]=max
                    estimates_this_frame['max_id']=max
            
        return estimates_this_frame
    
def readout_parameters(classification, parameters):
    # readout parameters
    parameters_for_this_classification=parameters[classification]
    difference_in_z=parameters_for_this_classification['difference_in_z']
    birth_rate=parameters_for_this_classification['birth_rate']
    P_s=parameters_for_this_classification['p_s']
    P_d=parameters_for_this_classification['p_d']
    use_ds_as_pd=parameters_for_this_classification['use_ds_as_pd']
    clutter_rate=parameters_for_this_classification['clutter_rate']
    bernoulli_gating=parameters_for_this_classification['bernoulli_gating']
    extraction_thr=parameters_for_this_classification['extraction_thr']
    ber_thr=parameters_for_this_classification['ber_thr']
    poi_thr=parameters_for_this_classification['poi_thr']
    eB_thr=parameters_for_this_classification['eB_thr']
    detection_score_thr=parameters_for_this_classification['detection_score_thr']
    nms_score = parameters_for_this_classification['nms_score']
    confidence_score = parameters_for_this_classification['confidence_score']
    P_init = parameters_for_this_classification['P_init']
    return difference_in_z,birth_rate, P_s,P_d, use_ds_as_pd, clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init

def readout_gnn_parameters(classification, parameters):
    # readout gnn parameters
    parameters_for_this_classification=parameters[classification]
    gating=parameters_for_this_classification['gating']
    P_d=parameters_for_this_classification['p_d']
    clutter_rate=parameters_for_this_classification['clutter_rate']
    detection_score_thr=parameters_for_this_classification['detection_score_thr']
    nms_score = parameters_for_this_classification['nms_score']
    death_counter_kill=parameters_for_this_classification['death_counter_kill']
    birth_counter_born=parameters_for_this_classification['birth_counter_born']
    death_initiation=parameters_for_this_classification['death_initiation']
    birth_initiation=parameters_for_this_classification['birth_initiation']



    return gating,P_d, clutter_rate,detection_score_thr, nms_score, death_counter_kill,birth_counter_born,death_initiation,birth_initiation




def initiate_submission_file(orderedframe):
    submission={}
    submission['meta']={}
    submission['meta']["use_camera"]=False
    submission['meta']["use_lidar"]=True
    submission['meta']["use_radar"]=False
    submission['meta']["use_map"]=False
    submission['meta']["use_external"]=False
    submission['results']={}
    for scene_token in orderedframe.keys():
        frames=orderedframe[scene_token]
        for frame_token in frames:
            submission['results'][frame_token]=[]
    return submission

def initiate_submission_file_mini(frames,estimated_bboxes_data_over_all_frames):
    submission={}
    submission['meta']={}
    submission['meta']["use_camera"]=False
    submission['meta']["use_lidar"]=True
    submission['meta']["use_radar"]=False
    submission['meta']["use_map"]=False
    submission['meta']["use_external"]=False
    submission['results']={}
    for frame in frames:
        frame_token=frame['token']
        if frame_token in estimated_bboxes_data_over_all_frames.keys():
            submission['results'][frame_token]=[]
    return submission

def initiate_classification_submission_file(classification):
    submission={}
    submission['meta']={}
    submission['meta']["use_camera"]=False
    submission['meta']["use_lidar"]=True
    submission['meta']["use_radar"]=False
    submission['meta']["use_map"]=False
    submission['meta']["use_external"]=False
    submission['meta']["classification"]=classification
    submission['results']={}
    return submission



def predict_for_all(Z_k, lag_time):
    F = np.eye(4, dtype=np.float64)
    I = lag_time*np.eye(2, dtype=np.float64)
    F[0:2, 2:4] = I
    '''
    instance_info['translation'] = translation_of_this_target
    instance_info['size'] = estimatedStates_for_this_classification['size'][idx]
    instance_info['rotation'] = estimatedStates_for_this_classification['rotation'][idx]
    instance_info['velocity'] = [estimatedStates_for_this_classification['mean']
                                 [idx][2][0], estimatedStates_for_this_classification['mean'][idx][3][0]]
    instance_info['tracking_id'] = estimatedStates_for_this_classification['classification'][idx]+'_'+str(
        estimatedStates_for_this_classification['id'][idx])
    instance_info['tracking_name'] = estimatedStates_for_this_classification['classification'][idx]
    instance_info['tracking_score']=estimatedStates_for_this_classification['detection_score'][idx]
    '''
    Z_k_new=[]
    if len(Z_k)>0:
        for z in Z_k:
            position=z['translation'][:2]
            for i in range(2):
                position.append(z['velocity'][i])
            new_position=F.dot(position)
            for j in range(2):
                z['translation'][j]=new_position[j]
            Z_k_new.append(z)
    return Z_k_new

def incorporate_track(Z_k,predict_previous_frame):  
    twodiou_register=np.zeros((len(Z_k),len(predict_previous_frame)))
    threediou_register=np.zeros((len(Z_k),len(predict_previous_frame)))
    Z_k_out=[]
    if len(Z_k)>0 and len(predict_previous_frame)>0:
        for index_1, z_1 in enumerate(Z_k): 
            for index_2, z_2 in enumerate(predict_previous_frame):
                twodiou_result, threediou_result=iou3d(nu_array2mot_bbox(z_1), nu_array2mot_bbox(z_2))
                twodiou_register[index_1][index_2]=twodiou_result
                threediou_register[index_1][index_2]=threediou_result
       
        for z_index, z in enumerate(Z_k):
            associated_tracks_score=threediou_register[z_index]
            best_associated_track_index=np.argmax(associated_tracks_score)
            best_association_score=associated_tracks_score[best_associated_track_index]
            best_associated_track=predict_previous_frame[best_associated_track_index]
    
            if best_association_score>0.5:
                if len(z['duplicated_detection'])>0:
                    probabilities=[]
                    classes=[]
                    probabilities.append(z['detection_score'])
                    classes.append(z['detection_name'])
                    for z_1 in z['duplicated_detection']:
                        probabilities.append(z['detection_score'])
                        classes.append(z['detection_name'])
                    # incorporate information from previous track if it exist
                    if best_associated_track['tracking_name'] in classes:
                        index_for_previous_track_classification=classes.index(best_associated_track['tracking_name'])
                        for index,probability in enumerate(probabilities):
                            if index!=index_for_previous_track_classification:
                                probabilities[index]=probability*0.3
                    # adjust detection name
                    normalized_probabilities=probabilities/np.sum(probabilities)
                    highest_class_index=np.argmax(probabilities)
                    if normalized_probabilities[highest_class_index]>0.8:
                        highest_class=classes[highest_class_index]
                        z['detection_name']=highest_class
                        Z_k_out.append(z)
                    else:
                        Z_k_out.append(z)
                        for z_1 in z['duplicated_detection']:
                            Z_k_out.append(z_1)
                else:
                    if best_associated_track['tracking_name']=='truck' or best_associated_track['tracking_name']=='bus' or best_associated_track['tracking_name']=='car':
                        if z['detection_name']!=best_associated_track['tracking_name']:
                            z['detection_name']=best_associated_track['tracking_name']
                    Z_k_out.append(z)
            else:
                if len(z['duplicated_detection'])>0:
                    probabilities=[]
                    classes=[]
                    probabilities.append(z['detection_score'])
                    classes.append(z['detection_name'])
                    for z_1 in z['duplicated_detection']:
                        probabilities.append(z['detection_score'])
                        classes.append(z['detection_name'])
    
                    # adjust detection name
                    normalized_probabilities=probabilities/np.sum(probabilities)
                    highest_class_index=np.argmax(probabilities)
                    if normalized_probabilities[highest_class_index]>0.8:
                        highest_class=classes[highest_class_index]
                        z['detection_name']=highest_class
                        Z_k_out.append(z)
                    else:
                        Z_k_out.append(z)
                        for z_1 in z['duplicated_detection']:
                            Z_k_out.append(z_1)
                else:
                    Z_k_out.append(z)
    return Z_k_out

def create_experiment_folder(root_directory_for_dataset, time,comment):
    result_path=os.path.join(root_directory_for_dataset, 'experiment_result')
    # generate the out file directory for this scene
    timefolder=os.path.join(result_path,time)
    experiment_folder=timefolder+'_'+comment
    #shutil.rmtree(timefolder, ignore_errors=True)
    if os.path.exists(experiment_folder):
        pass
    else:
        os.makedirs(experiment_folder)
    return experiment_folder
def create_scene_folder(scene, experiment_folder):
    # get the token of this scene
    scene_token=scene['token']
    # get the name of this scene
    scene_name=scene['name']
    #print('this is {}'.format(scene_name))
    out_file_directory_for_this_scene = os.path.join(experiment_folder,scene_name)
    # if the directory exist, then delete and make a new directory
    if os.path.exists(out_file_directory_for_this_scene):
        pass
    else:
        os.mkdir(out_file_directory_for_this_scene)
    return out_file_directory_for_this_scene

def create_scene_folder_name(scene_name, experiment_folder):
    # get the token of this scene
    # get the name of this scene
    #print('this is {}'.format(scene_name))
    out_file_directory_for_this_scene = os.path.join(experiment_folder,scene_name)
    # if the directory exist, then delete and make a new directory
    if os.path.exists(out_file_directory_for_this_scene):
        pass
    else:
        os.mkdir(out_file_directory_for_this_scene)
    return out_file_directory_for_this_scene


def create_classification_folder(classification,scene_folder):
    out_file_directory_for_this_scene_classfication = os.path.join(scene_folder, classification)
    # if the directory exist, then delete and make a new directory
    if os.path.exists(out_file_directory_for_this_scene_classfication):
        pass
    else:
        os.mkdir(out_file_directory_for_this_scene_classfication)
    return out_file_directory_for_this_scene_classfication

def gen_ordered_frames(scene,frames):
    # get all the frames associated with this scene
    frames_for_this_scene = []
    for frame in frames:
        if frame['scene_token']==scene['token']:
            frames_for_this_scene.append(frame)
            # set the frames in corret order
            # notice that NuScenes database does not provide the numerical ordering of the frames
            # it provides previous and next frame token information
            unordered_frames = copy.deepcopy(frames_for_this_scene)
            ordered_frames=[]
            # looping until the unordered frame is an empty set
            while len(unordered_frames)!=0:
                for current_frame in unordered_frames:
                    # if it is the first frame
                    if current_frame['prev']=='':
                        ordered_frames.append(current_frame)
                        # set current token
                        current_frame_token_of_current_scene = current_frame['token']
                        unordered_frames.remove(current_frame)
            
                    # find the next frame of current frame
                    if current_frame['prev']==current_frame_token_of_current_scene:
                        ordered_frames.append(current_frame)
                        # reset current frame
                        current_frame_token_of_current_scene=current_frame['token']
                        unordered_frames.remove(current_frame)
    return ordered_frames

def gen_measurement_of_this_class(detection_score_thr,estimated_bboxes_at_current_frame, classification):
    Z_k=[]
    for box_index, box in enumerate(estimated_bboxes_at_current_frame):
        if box['detection_name']==classification:
            if box['detection_score']>detection_score_thr:
                Z_k.append(box)
    return Z_k

def gen_measurement_all(estimated_bboxes_at_current_frame):
        
    # read parameters
    with open('/home/zhubinglab/Desktop/nuScenes_Tracker/configs/pmbmgnn_parameters.json', 'r') as f:
        parameters=json.load(f)
    Z_k=[]
    for classification in ['bicycle','motorcycle',  'trailer', 'truck','bus','pedestrian','car']:
        z_adjustment,birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classification, parameters)
        for box_index, box in enumerate(estimated_bboxes_at_current_frame):
            if box['detection_name']==classification and box['detection_score']>detection_score_thr:
                Z_k.append(box)
    return Z_k

def instance_info2bbox_array(info):
    translation = info.center.tolist()
    size = info.wlh.tolist()
    rotation = info.orientation.q.tolist()
    return translation + size + rotation

class BBoxCoarseFilter:
    def __init__(self, grid_size, scaler=100):
        self.gsize = grid_size
        self.scaler = 100
        self.bbox_dict = dict()
    
    def bboxes2dict(self, bboxes):
        for i, bbox in enumerate(bboxes):
            grid_keys = self.compute_bbox_key(bbox)
            for key in grid_keys:
                if key not in self.bbox_dict.keys():
                    self.bbox_dict[key] = set([i])
                else:
                    self.bbox_dict[key].add(i)
        return
        
    def compute_bbox_key(self, bbox):
        corners = np.asarray(BBox.box2corners2d(bbox))
        min_keys = np.floor(np.min(corners, axis=0) / self.gsize).astype(np.int)
        max_keys = np.floor(np.max(corners, axis=0) / self.gsize).astype(np.int)
        
        # enumerate all the corners
        grid_keys = [
            self.scaler * min_keys[0] + min_keys[1],
            self.scaler * min_keys[0] + max_keys[1],
            self.scaler * max_keys[0] + min_keys[1],
            self.scaler * max_keys[0] + max_keys[1]
        ]
        return grid_keys
    
    def related_bboxes(self, bbox):
        """ return the list of related bboxes
        """ 
        result = set()
        grid_keys = self.compute_bbox_key(bbox)
        for key in grid_keys:
            if key in self.bbox_dict.keys():
                result.update(self.bbox_dict[key])
        return list(result)
    
    def clear(self):
        self.bbox_dict = dict()

def weird_bbox(bbox):
    if bbox.l <= 0 or bbox.w <= 0 or bbox.h <= 0:
        return True
    else:
        return False

def nms(dets, threshold=0.1, threshold_high=1.0, threshold_yaw=0.3):
    """ 
    keep the bboxes with overlap <= threshold
    """
    dets_new=[]
    for det in dets:
        dets_new.append(nu_array2mot_bbox(det))
    dets=dets_new

    dets_coarse_filter = BBoxCoarseFilter(grid_size=100, scaler=100)
    dets_coarse_filter.bboxes2dict(dets)
    scores = np.asarray([det.s for det in dets])
    yaws = np.asarray([det.o for det in dets])
    order = np.argsort(scores)[::-1]
    
    result_indexes = list()
    while order.size > 0:
        index = order[0]

        if weird_bbox(dets[index]):
            order = order[1:]
            continue

        # locate the related bboxes
        filter_indexes = dets_coarse_filter.related_bboxes(dets[index])
        in_mask = np.isin(order, filter_indexes)
        related_idxes = order[in_mask]

        # compute the ious
        bbox_num = len(related_idxes)
        ious = np.zeros(bbox_num)
        for i, idx in enumerate(related_idxes):
            iou_result = iou3d(dets[index], dets[idx])[1]
            ious[i]=iou_result
        related_inds = np.where(ious > threshold)
        related_inds_vote = np.where(ious > threshold_high)
        order_vote = related_idxes[related_inds_vote]

        if len(order_vote) >= 2:
            # keep the bboxes with similar yaw
            if order_vote.shape[0] <= 2:
                score_index = np.argmax(scores[order_vote])
                median_yaw = yaws[order_vote][score_index]
            elif order_vote.shape[0] % 2 == 0:
                tmp_yaw = yaws[order_vote].copy()
                tmp_yaw = np.append(tmp_yaw, yaws[order_vote][0])
                median_yaw = np.median(tmp_yaw)
            else:
                median_yaw = np.median(yaws[order_vote])
            yaw_vote = np.where(np.abs(yaws[order_vote] - median_yaw) % (2 * np.pi) < threshold_yaw)[0]
            order_vote = order_vote[yaw_vote]
            
            # start weighted voting
            vote_score_sum = np.sum(scores[order_vote])
            det_arrays = list()
            for idx in order_vote:
                det_arrays.append(BBox.bbox2array(dets[idx])[np.newaxis, :])
            det_arrays = np.vstack(det_arrays)
            avg_bbox_array = np.sum(scores[order_vote][:, np.newaxis] * det_arrays, axis=0) / vote_score_sum
            bbox = BBox.array2bbox(avg_bbox_array)
            bbox.s = scores[index]
            result_indexes.append(index)
        else:
            result_indexes.append(index)

        # delete the overlapped bboxes
        delete_idxes = related_idxes[related_inds]
        in_mask = np.isin(order, delete_idxes, invert=True)
        order = order[in_mask]

    return result_indexes


def cross_classification_nms(dets, threshold=0.1, threshold_high=1.0, threshold_yaw=0.3):
    """ 
    keep the bboxes with overlap <= threshold
    """
    dets_new=[]
    for det in dets:
        dets_new.append(nu_array2mot_bbox(det))
    dets=dets_new

    dets_coarse_filter = BBoxCoarseFilter(grid_size=100, scaler=100)
    dets_coarse_filter.bboxes2dict(dets)
    scores = np.asarray([det.s for det in dets])
    yaws = np.asarray([det.o for det in dets])
    order = np.argsort(scores)[::-1]
    
    result_indexes = list()
    while order.size > 0:
        index = order[0]

        if weird_bbox(dets[index]):
            order = order[1:]
            continue

        # locate the related bboxes
        filter_indexes = dets_coarse_filter.related_bboxes(dets[index])
        in_mask = np.isin(order, filter_indexes)
        related_idxes = order[in_mask]

        # compute the ious
        bbox_num = len(related_idxes)
        ious = np.zeros(bbox_num)
        for i, idx in enumerate(related_idxes):
            ious[i] = iou3d(dets[index], dets[idx])[1]
        related_inds = np.where(ious > threshold)
        related_inds_vote = np.where(ious > threshold_high)
        order_vote = related_idxes[related_inds_vote]

        if len(order_vote) >= 2:
            # keep the bboxes with similar yaw
            if order_vote.shape[0] <= 2:
                score_index = np.argmax(scores[order_vote])
                median_yaw = yaws[order_vote][score_index]
            elif order_vote.shape[0] % 2 == 0:
                tmp_yaw = yaws[order_vote].copy()
                tmp_yaw = np.append(tmp_yaw, yaws[order_vote][0])
                median_yaw = np.median(tmp_yaw)
            else:
                median_yaw = np.median(yaws[order_vote])
            yaw_vote = np.where(np.abs(yaws[order_vote] - median_yaw) % (2 * np.pi) < threshold_yaw)[0]
            order_vote = order_vote[yaw_vote]
            
            # start weighted voting
            vote_score_sum = np.sum(scores[order_vote])
            det_arrays = list()
            for idx in order_vote:
                det_arrays.append(BBox.bbox2array(dets[idx])[np.newaxis, :])
            det_arrays = np.vstack(det_arrays)
            avg_bbox_array = np.sum(scores[order_vote][:, np.newaxis] * det_arrays, axis=0) / vote_score_sum
            bbox = BBox.array2bbox(avg_bbox_array)
            bbox.s = scores[index]
            result_indexes.append(index)
        else:
            result_indexes.append(index)

        # delete the overlapped bboxes
        delete_idxes = related_idxes[related_inds]
        in_mask = np.isin(order, delete_idxes, invert=True)
        order = order[in_mask]

    return result_indexes


def associate_dets_to_tracks(dets, tracks, mode, asso, 
    dist_threshold=0.9, trk_innovation_matrix=None):
    """ associate the tracks with detections
    """
    if mode == 'bipartite':
        matched_indices, dist_matrix = \
            bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    elif mode == 'greedy':
        matched_indices, dist_matrix = \
            greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    unmatched_dets = list()
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    unmatched_tracks = list()
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)
    
    matches = list()
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > dist_threshold:
            unmatched_dets.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(2))
    return matches, np.array(unmatched_dets), np.array(unmatched_tracks)


def bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    if asso == 'iou':
        dist_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'giou':
        dist_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'm_dis':
        dist_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        dist_matrix = compute_m_distance(dets, tracks, None)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)
    return matched_indices, dist_matrix


def greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    """ it's ok to use iou in bipartite
        but greedy is only for m_distance
    """
    matched_indices = list()
    
    # compute the distance matrix
    if asso == 'm_dis':
        distance_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        distance_matrix = compute_m_distance(dets, tracks, None)
    elif asso == 'iou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'giou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    num_dets, num_trks = distance_matrix.shape

    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_dets
    tracking_id_matches_to_detection_id = [-1] * num_trks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])
    if len(matched_indices) == 0:
        matched_indices = np.empty((0, 2))
    else:
        matched_indices = np.asarray(matched_indices)
    return matched_indices, distance_matrix


def compute_m_distance(dets, tracks, trk_innovation_matrix):
    """ compute l2 or mahalanobis distance
        when the input trk_innovation_matrix is None, compute L2 distance (euler)
        else compute mahalanobis distance
        return dist_matrix: numpy array [len(dets), len(tracks)]
    """
    euler_dis = (trk_innovation_matrix is None) # is use euler distance
    if not euler_dis:
        trk_inv_inn_matrices = [np.linalg.inv(m) for m in trk_innovation_matrix]
    dist_matrix = np.empty((len(dets), len(tracks)))

    for i, det in enumerate(dets):
        for j, trk in enumerate(tracks):
            if euler_dis:
                dist_matrix[i, j] = m_distance(det, trk)
            else:
                dist_matrix[i, j] = m_distance(det, trk, trk_inv_inn_matrices[j])
    return dist_matrix


def compute_iou_distance(dets, tracks, asso='iou'):
    iou_matrix = np.zeros((len(dets), len(tracks)))
    for d, det in enumerate(dets):
        for t, trk in enumerate(tracks):
            if asso == 'iou':
                iou_matrix[d, t] = iou3d(det, trk)[1]
            elif asso == 'giou':
                iou_matrix[d, t] = giou3d(det, trk)
    dist_matrix = 1 - iou_matrix
    return dist_matrix


def weird_bbox(bbox):
    if bbox.l <= 0 or bbox.w <= 0 or bbox.h <= 0:
        return True
    else:
        return False

class BBoxCoarseFilter:
    def __init__(self, grid_size, scaler=100):
        self.gsize = grid_size
        self.scaler = 100
        self.bbox_dict = dict()
    
    def bboxes2dict(self, bboxes):
        for i, bbox in enumerate(bboxes):
            grid_keys = self.compute_bbox_key(bbox)
            for key in grid_keys:
                if key not in self.bbox_dict.keys():
                    self.bbox_dict[key] = set([i])
                else:
                    self.bbox_dict[key].add(i)
        return
        
    def compute_bbox_key(self, bbox):
        corners = np.asarray(BBox.box2corners2d(bbox))
        min_keys = np.floor(np.min(corners, axis=0) / self.gsize).astype(np.int)
        max_keys = np.floor(np.max(corners, axis=0) / self.gsize).astype(np.int)
        
        # enumerate all the corners
        grid_keys = [
            self.scaler * min_keys[0] + min_keys[1],
            self.scaler * min_keys[0] + max_keys[1],
            self.scaler * max_keys[0] + min_keys[1],
            self.scaler * max_keys[0] + max_keys[1]
        ]
        return grid_keys
    
    def related_bboxes(self, bbox):
        """ return the list of related bboxes
        """ 
        result = set()
        grid_keys = self.compute_bbox_key(bbox)
        for key in grid_keys:
            if key in self.bbox_dict.keys():
                result.update(self.bbox_dict[key])
        return list(result)
    
    def clear(self):
        self.bbox_dict = dict()