from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
import mmcv
from nuscenes.nuscenes import NuScenes

import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode
import pickle
from utils.inference import inference
import utils.preprocessing_of_dataset as nuscenes_converter
import json
from utils.utils import boxes_iou_bev


def nuscenes_data_prep(output_root_path,
                       data_root_path,
                       version,
                       max_sweeps,
                       info_prefix):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(output_root_path, data_root_path, version=version, max_sweeps=max_sweeps, info_prefix=info_prefix)

    #info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    #nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)


def output_to_nusc_box(bboxes_for_this_inference_frame, labels_for_this_inference_frame, scores_for_this_inference_frame,remove_overlapping_bounding_boxes=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
            `1[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = bboxes_for_this_inference_frame
    labels = labels_for_this_inference_frame
    scores = scores_for_this_inference_frame
    box_to_be_removed = []
    box_list = []
    for i in range(len(box3d)):
        # only keep the box if it is not in the to be removed list
        if box3d[i].bev.tolist() not in box_to_be_removed:
            box_gravity_center = box3d[i].gravity_center.numpy()[0]
            box_dims = box3d[i].dims.numpy()[0]
            box_yaw = box3d[i].yaw.numpy()[0]
            box_yaw = -box_yaw - np.pi / 2
            score = scores[i]
    
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
            velocity = (*box3d.tensor[i, 7:9], 0.0)
            #velo_val = np.linalg.norm(box3d[i, 7:9])
            #velo_ori = box3d[i, 6]
            #velocity = (velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
            box = NuScenesBox(
                box_gravity_center,
                box_dims,
                quat,
                label=labels[i],
                velocity=velocity,
                score=score)
            box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(info,
                             boxes):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        #  filter det in ego.
        '''
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        '''
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
def removeprefix(input_string, prefix):
    return input_string[len(prefix):]

def format_bbox(data_infos, inference_results, extra_path_info, inference_result_in_nuscenes_format_file_path,remove_overlapping_bounding_boxes=True):
    """Convert the results to the standard format.

     Args:
        results (list[dict]): Testing results of the dataset.
        jsonfile_prefix (str): The prefix of the output jsonfile.
            You can specify the output directory/filename by
            modifying the jsonfile_prefix. Default: None.

     Returns:
        str: Path of the output json file.
    """
    nusc_annos = {}
    #mapped_class_names = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle','bicycle', 'motorcycle', 'pedestrian', 'traffic_cone','barrier')
    mapped_class_names = ('car', 'truck', 'trailer', 'bus','bicycle', 'motorcycle', 'pedestrian')
    print('Start to convert detection format...')
    for dataset_information_index in range(len(data_infos)):

        annos = []
        bboxes_for_this_frame = []
        labels_for_this_frame = []
        scores_for_this_frame = []

        for idx, inference_frame_data_path in enumerate(inference_results['file_name']):
            data_lidar_path = removeprefix(data_infos[dataset_information_index]['lidar_path'], extra_path_info)
            data_lidar_path = removeprefix(data_lidar_path, '/')
            if inference_frame_data_path == data_lidar_path:
                # read out the inference information for this frame
                bboxes_for_this_frame = inference_results['bboxes'][idx]
                labels_for_this_frame = inference_results['labels'][idx]
                scores_for_this_frame = inference_results['scores'][idx]
                break

        boxes = output_to_nusc_box(bboxes_for_this_frame, labels_for_this_frame,scores_for_this_frame,remove_overlapping_bounding_boxes=True)
        boxes = lidar_nusc_box_to_global(data_infos[dataset_information_index], boxes)
        # the token for a certain frame, which is specified by sample token
        sample_token = data_infos[dataset_information_index]['token']

        for i, box in enumerate(boxes):
            if box.label== 4 or box.label==8 or box.label==9:
                pass
            else:
                if box.label>4:
                    box.label-=1
                score_of_this_box=scores_for_this_frame[i].numpy()
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
    
                nusc_anno = dict(
                    sample_token=sample_token,
                    is_key_frame='yes',
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=score_of_this_box,
                    attribute_name=attr)
                annos.append(nusc_anno)
       
        nusc_annos[sample_token] = annos

    res_path = inference_result_in_nuscenes_format_file_path
    print('Result writes to', res_path)
    mmcv.dump(nusc_annos, res_path)
    return res_path
