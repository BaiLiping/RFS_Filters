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

class NuScenes:
    """
    Database class for nuScenes to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                            'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))

        # Explicitly assign tables to help the IDE determine valid class members.
        self.category = self.__load_table__('category')
        self.attribute = self.__load_table__('attribute')
        self.visibility = self.__load_table__('visibility')
        self.instance = self.__load_table__('instance')
        self.sensor = self.__load_table__('sensor')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        self.ego_pose = self.__load_table__('ego_pose')
        self.log = self.__load_table__('log')
        self.scene = self.__load_table__('scene')
        self.sample = self.__load_table__('sample')
        self.sample_data = self.__load_table__('sample_data')
        self.sample_annotation = self.__load_table__('sample_annotation')
        self.map = self.__load_table__('map')
        # a list of tracking record
        self.tracking_record=[]

        lidar_tasks = [t for t in ['lidarseg', 'panoptic'] if osp.exists(osp.join(self.table_root, t + '.json'))]
        if len(lidar_tasks) > 0:
            self.lidarseg_idx2name_mapping = dict()
            self.lidarseg_name2idx_mapping = dict()
            self.load_lidarseg_cat_name_mapping()
        for i, lidar_task in enumerate(lidar_tasks):
            if self.verbose:
                print(f'Loading nuScenes-{lidar_task}...')
            if lidar_task == 'lidarseg':
                self.lidarseg = self.__load_table__(lidar_task)
            else:
                self.panoptic = self.__load_table__(lidar_task)

            setattr(self, lidar_task, self.__load_table__(lidar_task))
            label_files = os.listdir(os.path.join(self.dataroot, lidar_task, self.version))
            num_label_files = len([name for name in label_files if (name.endswith('.bin') or name.endswith('.npz'))])
            num_lidarseg_recs = len(getattr(self, lidar_task))
            assert num_lidarseg_recs == num_label_files, \
                f'Error: there are {num_label_files} label files but {num_lidarseg_recs} {lidar_task} records.'
            self.table_names.append(lidar_task)

        # If available, also load the image_annotations table created by export_2d_annotations_as_json().
        if osp.exists(osp.join(self.table_root, 'image_annotations.json')):
            self.image_annotations = self.__load_table__('image_annotations')

        # Initialize map mask for each map record.
        for map_record in self.map:
            map_record['mask'] = MapMask(osp.join(self.dataroot, map_record['filename']), resolution=map_resolution)

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

        # Initialize NuScenesExplorer class.
        self.explorer = NuScenesExplorer(self)

    @property
    def table_root(self) -> str:
        """ Returns the folder where the tables are stored for the relevant version. """
        return osp.join(self.dataroot, self.version)

    def __load_table__(self, table_name) -> dict:
        """ Loads a table. """
        with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table

    def load_lidarseg_cat_name_mapping(self):
        """ Create mapping from class index to class name, and vice versa, for easy lookup later on """
        for lidarseg_category in self.category:
            # Check that the category records contain both the keys 'name' and 'index'.
            assert 'index' in lidarseg_category.keys(), \
                'Please use the category.json that comes with nuScenes-lidarseg, and not the old category.json.'

            self.lidarseg_idx2name_mapping[lidarseg_category['index']] = lidarseg_category['name']
            self.lidarseg_name2idx_mapping[lidarseg_category['name']] = lidarseg_category['index']

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        for ann_record in self.sample_annotation:
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

        # Add reverse indices from log records to map records.
        if 'log_tokens' not in self.map[0].keys():
            raise Exception('Error: log_tokens not in map table. This code is not compatible with the teaser dataset.')
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record['log_tokens']:
                log_to_map[log_token] = map_record['token']
        for log_record in self.log:
            log_record['map_token'] = log_to_map[log_record['token']]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]

    def field2token(self, table_name: str, field: str, query) -> List[str]:
        """
        This function queries all records for a certain field value, and returns the tokens for the matching records.
        Warning: this runs in linear time.
        :param table_name: Table name.
        :param field: Field name. See README.md for details.
        :param query: Query to match against. Needs to type match the content of the query field.
        :return: List of tokens for the matching records.
        """
        matches = []
        for member in getattr(self, table_name):
            if member[field] == query:
                matches.append(member['token'])
        return matches

    def get_sample_data_path(self, sample_data_token: str) -> str:
        """ Returns the path to a sample_data. """

        sd_record = self.get('sample_data', sample_data_token)
        return osp.join(self.dataroot, sd_record['filename'])

    def get_sample_data(self, sample_data_token: str,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        selected_anntokens: List[str] = None,
                        use_flat_vehicle_coordinates: bool = False) -> \
            Tuple[str, List[Box], np.array]:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param selected_anntokens: If provided only return the selected annotation.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                             aligned to z-plane in the world.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.get('sensor', cs_record['sensor_token'])
        pose_record = self.get('ego_pose', sd_record['ego_pose_token'])

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(self.get_box, selected_anntokens))
        else:
            boxes = self.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not \
                    box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def get_box(self, sample_annotation_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get('sample_annotation', sample_annotation_token)
        return Box(record['translation'], record['size'], Quaternion(record['rotation']),
                   name=record['category_name'], token=record['token'])
    def get_boxes(self, sample_data_token: str) -> List[Box]:
        """
        Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
        keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
        sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
        sample_data was captured.
        :param sample_data_token: Unique sample_data identifier.
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        curr_sample_record = self.get('sample', sd_record['sample_token'])

        if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record['anns']))

        else:
            prev_sample_record = self.get('sample', curr_sample_record['prev'])

            curr_ann_recs = [self.get('sample_annotation', token) for token in curr_sample_record['anns']]
            prev_ann_recs = [self.get('sample_annotation', token) for token in prev_sample_record['anns']]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

            t0 = prev_sample_record['timestamp']
            t1 = curr_sample_record['timestamp']
            t = sd_record['timestamp']

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec['instance_token'] in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                    # Interpolate center.
                    center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                                 curr_ann_rec['translation'])]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                                q1=Quaternion(curr_ann_rec['rotation']),
                                                amount=(t - t0) / (t1 - t0))

                    box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                              token=curr_ann_rec['token'])
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box(curr_ann_rec['token'])

                boxes.append(box)
        return boxes

    def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        """
        Estimate the velocity for an annotation.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
        :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
        """

        current = self.get('sample_annotation', sample_annotation_token)
        has_prev = current['prev'] != ''
        has_next = current['next'] != ''

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.get('sample_annotation', current['prev'])
        else:
            first = current

        if has_next:
            last = self.get('sample_annotation', current['next'])
        else:
            last = current

        pos_last = np.array(last['translation'])
        pos_first = np.array(first['translation'])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * self.get('sample', last['sample_token'])['timestamp']
        time_first = 1e-6 * self.get('sample', first['sample_token'])['timestamp']
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff


    def render_sample(self, sample_token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      verbose: bool = True) -> None:
        self.explorer.render_sample(sample_token, box_vis_level, nsweeps=nsweeps, out_path=out_path,verbose=verbose)

    def render_sample_data(self, sample_data_token: str, with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY, axes_limit: float =60, ax: Axes = None,
                           nsweeps: int = 1, out_path: str = None, bird_eye_view_with_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           verbose: bool = True) -> None:
        self.explorer.render_sample_data(sample_data_token, with_anns, box_vis_level, axes_limit, ax, nsweeps=nsweeps,
                                         out_path=out_path,
                                         bird_eye_view_with_map=bird_eye_view_with_map,
                                         use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                         verbose=verbose)
       
    def render_inference_sample(self,
                      inference_result, 
                      sample_token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      verbose: bool = True) -> None:
        self.explorer.render_inference_sample(inference_result,sample_token, box_vis_level, nsweeps=nsweeps, out_path=out_path,verbose=verbose)

    def render_inference_sample_data(self, sample_data_token: str, with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY, axes_limit: float = 60, ax: Axes = None,
                           nsweeps: int = 1, out_path: str = None, bird_eye_view_with_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           verbose: bool = True) -> None:
        self.explorer.render_inference_sample_data(sample_data_token, with_anns, box_vis_level, axes_limit, ax, nsweeps=nsweeps,
                                         out_path=out_path,
                                         bird_eye_view_with_map=bird_eye_view_with_map,
                                         use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                         verbose=verbose)

    def render_egoposes_on_map(self, log_location: str, scene_tokens: List = None, out_path: str = None) -> None:
        self.explorer.render_egoposes_on_map(log_location, scene_tokens, out_path=out_path)

    def render_target_position(self, thr, estimation, sample_data_token: str, with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY, axes_limit: float = 60, ax: Axes = None,
                           nsweeps: int = 1, out_path: str = None, bird_eye_view_with_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           verbose: bool = True) -> None:
        self.explorer.render_target_position(thr, estimation, sample_data_token, with_anns, box_vis_level, axes_limit, ax, nsweeps=nsweeps,
                                         out_path=out_path,
                                         bird_eye_view_with_map=bird_eye_view_with_map,
                                         use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                         verbose=verbose)
    def render_tracker_result(self, thr, ground_truth_bboxes,ground_truth_type_for_this_frame,inference_result, sample_token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      bird_eye_view_with_map = False,
                      verbose: bool = True) -> None:
        self.explorer.render_tracker_result(thr, ground_truth_bboxes,ground_truth_type_for_this_frame,inference_result, sample_token, box_vis_level, nsweeps=nsweeps, out_path=out_path,bird_eye_view_with_map=bird_eye_view_with_map,verbose=verbose)


    def render_egoposes_on_map(self, log_location: str, scene_tokens: List = None, out_path: str = None) -> None:
        self.explorer.render_egoposes_on_map(log_location, scene_tokens, out_path=out_path)

class NuScenesExplorer:
    """ Helper class to list and visualize NuScenes data. These are meant to serve as tutorials and templates for
    working with the data. """

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        self.tracking_record=[]

    def render_sample(self,
                      token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      verbose: bool = True) -> None:
        """
        Render all LIDAR and camera sample_data in sample along with annotations.
        :param token: Sample token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param show_lidarseg: Whether to show lidar segmentations labels or not.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to show the rendered sample in a window or not.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        record = self.nusc.get('sample', token)

        # Separate RADAR from LIDAR and vision.
        radar_data = {}
        camera_data = {}
        lidar_data = {}
        for channel, token in record['data'].items():
            sd_record = self.nusc.get('sample_data', token)
            sensor_modality = sd_record['sensor_modality']

            if sensor_modality == 'camera':
                camera_data[channel] = token
            elif sensor_modality == 'lidar':
                lidar_data[channel] = token
            else:
                radar_data[channel] = token

        # Create plots.
        num_radar_plots = 1 if len(radar_data) > 0 else 0
        num_lidar_plots = 1 if len(lidar_data) > 0 else 0
        n = num_radar_plots + len(camera_data) + num_lidar_plots
        cols = 2
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 24))

        # Plot radars into a single subplot.
        if len(radar_data) > 0:
            ax = axes[0, 0]
            for i, (_, sd_token) in enumerate(radar_data.items()):
                self.render_sample_data(sd_token, with_anns=i == 0, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,
                                        verbose=False)
            ax.set_title('Fused RADARs')

        # Plot lidar into a single subplot.
        if len(lidar_data) > 0:
            for (_, sd_token), ax in zip(lidar_data.items(), axes.flatten()[num_radar_plots:]):
                self.render_sample_data(sd_token, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,verbose=False)

        # Plot cameras in separate subplots.
        for (_, sd_token), ax in zip(camera_data.items(), axes.flatten()[num_radar_plots + num_lidar_plots:]):

            self.render_sample_data(sd_token, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps, verbose=False)

        # Change plot settings and write to disk.
        axes.flatten()[-1].axis('off')
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        if out_path is not None:
            plt.savefig(out_path)

        if verbose:
            plt.show()

    def render_sample_data(self,
                           sample_data_token: str,
                           with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 60,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           bird_eye_view_with_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           verbose: bool = True) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param bird_eye_view_with_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to display the image after it is rendered.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """

        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':

                # Get aggregated lidar point cloud in lidar frame.
                pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan,nsweeps=nsweeps)
                velocities = None
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if bird_eye_view_with_map:
                assert use_flat_vehicle_coordinates, 'Error: bird_eye_view_with_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

            # Show velocities.
            if sensor_modality == 'radar':
                points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
                deltas_vel = points_vel - points
                deltas_vel = 6 * deltas_vel  # Arbitrary scaling
                max_delta = 20
                deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
                colors_rgba = scatter.to_rgba(colors)
                for i in range(points.shape[1]):
                    ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')

            # Get boxes in lidar frame.
            _, boxes, _ = self.nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                    use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)
        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,
                                                                           box_vis_level=box_vis_level)
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis('off')
        #ax.set_title('{} {labels_type}'.format(
        #    sd_record['channel'], labels_type='(predictions)'))
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

        if verbose:
            plt.show()

    def render_inference_sample(self,
                      inference_result_in_nuscenes_format,
                      token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      verbose: bool = True) -> None:
        """
        Render all LIDAR and camera sample_data in sample along with annotations.
        :param token: Sample token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param verbose: Whether to show the rendered sample in a window or not.
        """
        # get all the information with the same sample token
        record = self.nusc.get('sample', token)

        # Separate RADAR from LIDAR and vision.
        radar_data = {}
        camera_data = {}
        lidar_data = {}
        for channel, sample_data_channel_token in record['data'].items():
            sd_record = self.nusc.get('sample_data', sample_data_channel_token)
            sensor_modality = sd_record['sensor_modality']
            
            # sort the sample data token with various sensing modalities
            if sensor_modality == 'camera':
                camera_data[channel] = sample_data_channel_token
            elif sensor_modality == 'lidar':
                lidar_data[channel] = sample_data_channel_token
            else:
                radar_data[channel] = sample_data_channel_token

        # Create plots.
        num_radar_plots = 1
        num_lidar_plots = 1
        num_camera_plot = 1
        n = num_radar_plots + num_camera_plot + num_lidar_plots
        cols = n
        fig, axes = plt.subplots(1, cols,figsize=(24,12))
        
        # Plot radars into a single subplot.
        # radar data would be the first plat for this frame
        ax = axes[0]
        # there are five radars
        for i, (_, sd_token) in enumerate(radar_data.items()):
            self.render_inference_sample_data(inference_result_in_nuscenes_format,sd_token, with_anns=i == 0, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,verbose=False)
            ax.set_title('FUSED 5 RADARS',fontsize=40)
        
        # Plot lidar into a single subplot.
        ax = axes[1]
        for (_, sd_token), ax in zip(lidar_data.items(), ax):
            self.render_inference_sample_data(inference_result_in_nuscenes_format,sd_token, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,verbose=False)
            ax.set_title("TOP LIDAR",fontsize=40)
        
        # Plot cameras in separate subplots.        
        ax = axes[2]
        camera_sensor = 'CAM_FRONT'
        camera_data = self.nusc.get('sample_data', record['data'][camera_sensor])
        self.render_inference_sample_data(inference_result_in_nuscenes_format,camera_data['token'], box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps, verbose=False)        
        ax.set_title("FRONT CAMERA",fontsize=40)

        # Change plot settings and write to disk.
        axes.flatten()[-1].axis('off')
        plt.tight_layout()
        '''
        "car": (255, 69, 0),  # Orangered.
        "truck": (70, 130, 180),  # Steelblue
        "trailer": (255,0,255), #magenta
        "bus":(0,255,0),  # lime
        "construction_vehicle": (255,255,0),  # Gold
        "bicycle": (0, 175, 0),  # Green
        "motorcycle": (0, 0, 128),  # Navy,
        "pedestrian":((0, 0, 230),  # Blue
        "traffic_cone": (138, 43, 226),  # Blueviolet
        "barrier": (173,255,47),  # greenyello
        '''
        car_patch = mpatches.Patch(color=(255/255, 69/255, 0/255), label='car')
        truck_patch = mpatches.Patch(color=(70/255, 130/255, 180/255), label='truck')
        trailer_patch = mpatches.Patch(color=(255/255, 0/255, 255/255), label='trailer')
        bus_patch = mpatches.Patch(color=(219/255, 112/255, 147/255), label='bus')
        construction_vehicle_patch = mpatches.Patch(color=(255/255, 255/255, 0), label='construction vehicle')
        bicycle_patch = mpatches.Patch(color=(0, 175/255, 0), label='bicycle')
        motorcycle_patch = mpatches.Patch(color=(0, 0, 128/255), label='motorcycle')
        pedestrian_patch = mpatches.Patch(color=(0,0, 230/255), label='pedestrian')
        traffic_cone_patch = mpatches.Patch(color=(138/255, 43/255, 226/255), label='traffic cone')
        barrier_patch = mpatches.Patch(color=(173/255,255/255,47/255), label='barrier')  

        fig.subplots_adjust(wspace=0, hspace=0)
        fig.legend(handles=[car_patch, truck_patch,trailer_patch,bus_patch,construction_vehicle_patch,bicycle_patch,motorcycle_patch,pedestrian_patch,traffic_cone_patch,barrier_patch],loc='lower left',prop={'size': 20},ncol=3)

        if out_path is not None:
            plt.savefig(out_path)

        if verbose:
            plt.show()


    def render_ego_centric_map(self,
                               sample_data_token: str,
                               axes_limit: float =60,
                               ax: Axes = None) -> None:
        """
        Render map centered around the associated ego pose.
        :param sample_data_token: Sample_data token.
        :param axes_limit: Axes limit measured in meters.
        :param ax: Axes onto which to render.
        """

        def crop_image(image: np.array,
                       x_px: int,
                       y_px: int,
                       axes_limit_px: int) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        # Get data.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sample = self.nusc.get('sample', sd_record['sample_token'])
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        map_ = self.nusc.get('map', log['map_token'])
        map_mask = map_['mask']
        pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

        # Retrieve and crop mask.
        pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
        scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
        mask_raster = map_mask.mask()
        cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

        # Rotate image.
        ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0])
        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

        # Crop image.
        ego_centric_map = crop_image(rotated_cropped,
                                     int(rotated_cropped.shape[1] / 2),
                                     int(rotated_cropped.shape[0] / 2),
                                     scaled_limit_px)

        # Init axes and show image.
        # Set background to white and foreground (semantic prior) to gray.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))
        ego_centric_map[ego_centric_map == map_mask.foreground] = 125
        ego_centric_map[ego_centric_map == map_mask.background] = 255
        ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit],
                  cmap='gray', vmin=0, vmax=255)

    def render_inference_sample_data(self,
                           inference_result_in_nuscenes_format,
                           sample_data_token: str,
                           with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 60,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           bird_eye_view_with_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           verbose: bool = False) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param bird_eye_view_with_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param verbose: Whether to display the image after it is rendered.

        """

        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.nusc.get('sensor', cs_record['sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        
        if sensor_modality == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':
                # Get aggregated lidar point cloud in lidar frame.
                pc, _ = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan,nsweeps=nsweeps)
                velocities = None
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, _ = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)
                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(24, 12))

            # Render map if requested.
            if bird_eye_view_with_map:
                assert use_flat_vehicle_coordinates, 'Error: bird_eye_view_with_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

            # Show velocities.
            if sensor_modality == 'radar':
                points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
                deltas_vel = points_vel - points
                deltas_vel = 6 * deltas_vel  # Arbitrary scaling
                max_delta = 20
                deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
                colors_rgba = scatter.to_rgba(colors)
                for i in range(points.shape[1]):
                    ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')
            sample_token = self.nusc.get('sample_data', sample_data_token)['sample_token']
            boxes = inference_result_in_nuscenes_format

            # Show boxes.
            for box in boxes:
                new_bbox = Box(box['translation'], box['size'], Quaternion(box['rotation']),name=box['tracking_name'], token=box['sample_token'])
                if use_flat_vehicle_coordinates:
                    # Move box to ego vehicle coord system parallel to world z plane.
                    yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                    new_bbox.translate(-np.array(pose_record['translation']))
                    new_bbox.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
                else:
                    # Move box to ego vehicle coord system.
                    new_bbox.translate(-np.array(pose_record['translation']))
                    new_bbox.rotate(Quaternion(pose_record['rotation']).inverse)
    
                    #  Move box to sensor coord system.
                    new_bbox.translate(-np.array(cs_record['translation']))
                    new_bbox.rotate(Quaternion(cs_record['rotation']).inverse)

                c= np.array(get_inference_colormap(new_bbox.name))/255.0
                new_bbox.render(ax, view=np.eye(4), colors=(c, c, c))
 
            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)
        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,box_vis_level=box_vis_level)
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(24, 12))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            sample_token = self.nusc.get('sample_data', sample_data_token)['sample_token']
            boxes = inference_result_in_nuscenes_format
            # Show boxes.
            for box in boxes:
                new_bbox = Box(box['translation'], box['size'], Quaternion(box['rotation']),name=box['tracking_name'], token=box['sample_token'])
                # Move box to ego vehicle coord system.
                new_bbox.translate(-np.array(pose_record['translation']))
                new_bbox.rotate(Quaternion(pose_record['rotation']).inverse)
                #  Move box to sensor coord system.
                new_bbox.translate(-np.array(cs_record['translation']))
                new_bbox.rotate(Quaternion(cs_record['rotation']).inverse)
            
                if sensor_record['modality'] == 'camera' and not \
                        box_in_image(new_bbox, cam_intrinsic, imsize, vis_level=box_vis_level):
                    continue

                c= np.array(get_inference_colormap(new_bbox.name))/255.0
                new_bbox.render(ax, view=camera_intrinsic,  normalize=True,colors=(c, c, c))
            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis('off')
        #ax.set_title('{} {labels_type}'.format(
        #    sd_record['channel'], labels_type='(predictions)'))
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

        if verbose:
            plt.show()

    def render_target_position(self,
                           thr,
                           ground_truth,ground_truth_type_for_this_frame,
                           inference_result_in_nuscenes_format,
                           sample_data_token: str,
                           with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 60,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           bird_eye_view_with_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           verbose: bool = False) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param bird_eye_view_with_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param verbose: Whether to display the image after it is rendered.

        """
        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.nusc.get('sensor', cs_record['sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        

        if sensor_modality == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':
                # Get aggregated lidar point cloud in lidar frame.
                pc, _ = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan,nsweeps=nsweeps)
                velocities = None
            
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, _ = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)
                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])
            
            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(12, 12))

            # Render map if requested.
            if bird_eye_view_with_map:
                assert use_flat_vehicle_coordinates, 'Error: bird_eye_view_with_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)
            
            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
            

            # Show velocities.
            if sensor_modality == 'radar':
                points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
                deltas_vel = points_vel - points
                deltas_vel = 6 * deltas_vel  # Arbitrary scaling
                max_delta = 20
                deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
                colors_rgba = scatter.to_rgba(colors)
                for i in range(points.shape[1]):
                    ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])
            
            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')
            sample_token = self.nusc.get('sample_data', sample_data_token)['sample_token']

            # render estimation postion
            target_box_record=[]
            for idx, target in enumerate(inference_result_in_nuscenes_format):
                translation = [target['translation'][0], target['translation'][1], target['translation'][2]]
                #size =[1,1,1]
                size=target['size']
                
                rotation=Quaternion(target['rotation'])
                target_box = Box(translation,size,rotation)


                # Move box to ego vehicle coord system.
                target_box.translate(-np.array(pose_record['translation']))
                #target_box.rotate(Quaternion(pose_record['rotation']).inverse)
                #  Move box to sensor coord system.
                #target_box.translate(-np.array(cs_record['translation']))  
                #target_box.rotate(Quaternion(cs_record['rotation']).inverse)
                
                # Move box to ego vehicle coord system.
                #target_box.translate(-np.array(pose_record['translation']))
                #target_box.rotate(Quaternion(pose_record['rotation']).inverse)
                #  Move box to sensor coord system.
                #target_box.translate(-np.array(cs_record['translation']))  
                #target_box.rotate(Quaternion(cs_record['rotation']).inverse)
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]


                '''
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                target_box.translate(-np.array(pose_record['translation']))
                target_box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
                # Move box to ego vehicle coord system.
                #target_box.translate(-np.array(pose_record['translation']))
                #target_box.rotate(Quaternion(pose_record['rotation']).inverse)
                #  Move box to sensor coord system.
                #target_box.translate(-np.array(cs_record['translation']))  
                #target_box.rotate(Quaternion(cs_record['rotation']).inverse)
                '''

                #corners = view_points(ax, view=np.eye(4), normalize=True)[:2, :]
                
                #center_bottom = np.mean(target_box.corners.T[[2, 3, 7, 6]], axis=0)
                center_bottom=target_box.center
                #c=np.array([0, 0, 0])/255.0
                #c= np.array(get_inference_colormap(target['tracking_name']))/255.0
                c= np.array(get_inference_colormap(target['tracking_name']))/255.0

                #ax.plot(target_box.center[0], target_box.center[1], 'o', color='red')

                target_box_record.append(target_box)
                target_box.render(ax, view=np.eye(4), colors=(c,c,c))
                text='{}'.format(target['tracking_id'])
                #position1=np.random.uniform(-2,2)
                #position2=np.random.uniform(-2,2)
                position1=2
                position2=-3
                if target['tracking_name']=='pedestrian':
                    #ax.text(center_bottom[0], center_bottom[1]+position1,text , fontsize = 7, color=c)
                    ax.text(center_bottom[0], center_bottom[1],np.round(target['tracking_score'],2) , fontsize = 15, color=c)
                elif target['tracking_name']=='truck':
                    ax.text(center_bottom[0], center_bottom[1]-position1,text , fontsize = 15, color=c)
                    ax.text(center_bottom[0], center_bottom[1]-position2,np.round(target['tracking_score'],2) , fontsize = 15, color=c)
                else:
                    ax.text(center_bottom[0], center_bottom[1]+position1,text , fontsize = 15, color=c)
                    ax.text(center_bottom[0], center_bottom[1]+position2,np.round(target['tracking_score'],2) , fontsize = 15, color=c)
                
                velocity=[target['velocity'][0],target['velocity'][0],0]
                velocity=np.dot(Quaternion(cs_record['rotation']).inverse.rotation_matrix, velocity)
                velocity= np.dot(Quaternion(pose_record['rotation']).inverse.rotation_matrix, velocity)
                
                
                ax.arrow(center_bottom[0], center_bottom[1], 6*velocity[0], 6*velocity[1], color='red')
                #ax.text(center_bottom[0]+3, center_bottom[1],np.round(target['velocity'][0],2) , fontsize = 15, color=c)
                #ax.text(center_bottom[0]-3, center_bottom[1],np.round(target['velocity'][1],2) , fontsize = 15, color=c)


                #ax.scatter(center_bottom[0], center_bottom[1],color=c)
                # the following part is only applicable to PHD filter
                #ax.text(target[0][0], target[1][0]-2, '{:.2f}'.format(estimated_target_position['w'][idx]) , fontsize = 20)
            '''
            #boxes = inference_result_in_nuscenes_format[sample_token]
            boxes = inference_result_in_nuscenes_format

            # Show boxes
            box_record = []
            for box in boxes:
                #if True:

                if box['tracking_score']>thr:
                    new_bbox = Box(box['translation'], box['size'], Quaternion(box['rotation']),name=box['tracking_name'], token=box['sample_token'])
                    
                    #if use_flat_vehicle_coordinates:
                    #    # Move box to ego vehicle coord system parallel to world z plane.
                    yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                    new_bbox.translate(-np.array(pose_record['translation']))
                    new_bbox.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
                    #else:
                    # Move box to ego vehicle coord system.
                    #new_bbox.translate(-np.array(pose_record['translation']))
                    #new_bbox.rotate(Quaternion(pose_record['rotation']).inverse)
                    #  Move box to sensor coord system.
                    #new_bbox.translate(-np.array(cs_record['translation']))  
                    #new_bbox.rotate(Quaternion(cs_record['rotation']).inverse)
                    
                    c= np.array(get_inference_colormap(new_bbox.name))/255.0
                    new_bbox.render(ax, view=np.eye(4), colors=(c, c, c))
                    box_record.append(new_bbox)
            '''
                    
            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)
        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,box_vis_level=box_vis_level)
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(24, 12))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            sample_token = self.nusc.get('sample_data', sample_data_token)['sample_token']
            #boxes = inference_result_in_nuscenes_format
            boxes = inference_result_in_nuscenes_format
            
            # render estimation postion
            for idx, target in enumerate(inference_result_in_nuscenes_format):
                target_box = Box([target['translation'][0], target['translation'][1], target['translation'][2]],target['size'],Quaternion(target['rotation']))
                
                # Move box to ego vehicle coord system.
                target_box.translate(-np.array(pose_record['translation']))
                target_box.rotate(Quaternion(pose_record['rotation']).inverse)
                #  Move box to sensor coord system.
                target_box.translate(-np.array(cs_record['translation']))  
                target_box.rotate(Quaternion(cs_record['rotation']).inverse)

                c= np.array(get_inference_colormap(target['tracking_name']))/255.0
                #c=np.array((0,0,0))/255.0
                if sensor_record['modality'] == 'camera' and not \
                        box_in_image(target_box, cam_intrinsic, imsize, vis_level=box_vis_level):
                    continue
                corners=target_box.render(ax,  view=camera_intrinsic,  normalize=True, colors=(c,c,c))
                corners = view_points(target_box.corners(), view=camera_intrinsic, normalize=True)[:2, :]
                center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
                
                #text='{}'.format(target['tracking_id'])
                #ax.scatter(center_bottom[0],center_bottom[1],color=c)
                #ax.text(center_bottom[0], center_bottom[1]+2,text , fontsize = 25,color=c)
                #ax.text(center_bottom[0], center_bottom[1]-2, '{:.2f}'.format(estimated_target_position['w'][idx]) , fontsize = 20)
            '''
            # Show boxes.
            for box in boxes:
                #if True:
                if box['tracking_score']>thr:
                    new_bbox = Box(box['translation'], box['size'], Quaternion(box['rotation']),name=box['tracking_name'], token=box['sample_token'])
                    
                    #  Move box to ego vehicle coord system.
                    new_bbox.translate(-np.array(pose_record['translation']))
                    new_bbox.rotate(Quaternion(pose_record['rotation']).inverse)
                    #  Move box to sensor coord system.
                    new_bbox.translate(-np.array(cs_record['translation']))
                    new_bbox.rotate(Quaternion(cs_record['rotation']).inverse)
                
                
                    if sensor_record['modality'] == 'camera' and not \
                            box_in_image(new_bbox, cam_intrinsic, imsize, vis_level=box_vis_level):
                        continue
    
                    c= np.array(get_inference_colormap(new_bbox.name))/255.0
                    new_bbox.render(ax, view=camera_intrinsic,  normalize=True,colors=(c, c, c))
            '''
            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")
        
        '''
        for idx, target in enumerate(ground_truth):
            target_box = Box([target[0], target[1], target[2]],[target[3], target[4], target[5]],Quaternion([target[6], target[7], target[8], target[9]]))
            c= np.array(get_inference_colormap(ground_truth_type_for_this_frame[idx]))/255.0
            # Move box to ego vehicle coord system.
            target_box.translate(-np.array(pose_record['translation']))
            #target_box.rotate(Quaternion(pose_record['rotation']).inverse)
            #target_box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            #  Move box to sensor coord system.
            target_box.translate(-np.array(cs_record['translation'])) 
            center_bottom=target_box.center 
            #target_box.rotate(Quaternion(cs_record['rotation']).inverse)
            #c= 'k'     
            #corners=target_box.render(ax,  view=camera_intrinsic,  normalize=True, colors=(c,c,c))
            #target_box.render(ax, view=np.eye(4), colors=(c, c, c))
            
            #text='{}'.format(target['tracking_id'])
            ax.scatter(center_bottom[0],center_bottom[1],color=c)
            #ax.text(center_bottom[0], center_bottom[1]+2,text , fontsize = 25,color=c)
            #ax.text(center_bottom[0], center_bottom[1]-2, '{:.2f}'.format(estimated_target_position['w'][idx]) , fontsize = 20)
        '''

        ax.axis('off')
        #ax.set_title('{} {labels_type}'.format(
        #    sd_record['channel'], labels_type='(predictions)'))
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)

        if verbose:
            plt.show()

    def render_tracker_result(self,
                      thr, 
                      ground_truth_bboxes,ground_truth_type_for_this_frame, 
                      inference_result_in_nuscenes_format,
                      token: str,
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      nsweeps: int = 1,
                      out_path: str = None,
                      bird_eye_view_with_map= False,
                      verbose: bool = True) -> None:
        """
        Render all LIDAR and camera sample_data in sample along with annotations.
        :param token: Sample token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param verbose: Whether to show the rendered sample in a window or not.
        """

        # get the estimated target positions
        estimatedStates_for_current_frame=inference_result_in_nuscenes_format

        # get all the information with the same sample token
        record = self.nusc.get('sample', token)
        
        # Separate RADAR from LIDAR and vision.
        radar_data = {}
        camera_data = {}
        lidar_data = {}
        for channel, sample_data_channel_token in record['data'].items():
            sd_record = self.nusc.get('sample_data', sample_data_channel_token)
            sensor_modality = sd_record['sensor_modality']
            
            # sort the sample data token with various sensing modalities
            if sensor_modality == 'camera':
                camera_data[channel] = sample_data_channel_token
            elif sensor_modality == 'lidar':
                lidar_data[channel] = sample_data_channel_token
            else:
                radar_data[channel] = sample_data_channel_token

        # Create plots.
        num_radar_plots = 1
        num_lidar_plots = 1
        num_camera_plot = 1
        n = num_radar_plots + num_camera_plot + num_lidar_plots
        cols = n
        """
        fig, axes = plt.subplots(2, 2,figsize=(24,24))
        
        # Plot radars into a single subplot.
        # radar data would be the first plat for this frame
        ax = axes[0][0]
        # there are five radars
        for i, (_, sd_token) in enumerate(radar_data.items()):
            self.render_target_position(thr, ground_truth_bboxes,ground_truth_type_for_this_frame,inference_result_in_nuscenes_format,sd_token, with_anns=i == 0, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,bird_eye_view_with_map=bird_eye_view_with_map,verbose=False)
            ax.set_title('FUSED 5 RADARS',fontsize=40)


        # Plot lidar into a single subplot.
        ax = axes[0][1]
        for i, (_, sd_token) in enumerate(lidar_data.items()):
            self.render_target_position(thr,  ground_truth_bboxes,ground_truth_type_for_this_frame,inference_result_in_nuscenes_format,sd_token, box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,bird_eye_view_with_map=bird_eye_view_with_map,verbose=False)
            ax.set_title("TOP LIDAR",fontsize=40)
        # Plot cameras in separate subplots.        
        ax = axes[1][0]
        camera_sensor = 'CAM_FRONT'
        camera_data = self.nusc.get('sample_data', record['data'][camera_sensor])
        self.render_target_position(thr,  ground_truth_bboxes,ground_truth_type_for_this_frame,inference_result_in_nuscenes_format,camera_data['token'], box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,verbose=False)        
        ax.set_title("FRONT CAMERA",fontsize=40)

        ax = axes[1][1]
        camera_sensor = 'CAM_BACK'
        camera_data = self.nusc.get('sample_data', record['data'][camera_sensor])
        self.render_target_position(thr,  ground_truth_bboxes,ground_truth_type_for_this_frame,inference_result_in_nuscenes_format,camera_data['token'], box_vis_level=box_vis_level, ax=ax, nsweeps=nsweeps,verbose=False)        
        ax.set_title("BACK CAMERA",fontsize=40)
        
        # Change plot settings and write to disk.
        axes.flatten()[-1].axis('off')
        plt.tight_layout()
        
        '''
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
        '''

        car_patch = mpatches.Patch(color=(0/255, 0/255, 230/255), label='car')
        truck_patch = mpatches.Patch(color=(70/255, 130/255, 180/255), label='truck')
        trailer_patch = mpatches.Patch(color=(138/255, 43/255, 226/255), label='trailer')
        bus_patch = mpatches.Patch(color=(0/255, 255/255, 0/255), label='bus')
        construction_vehicle_patch = mpatches.Patch(color=(255/255, 255/255, 0), label='construction vehicle')
        bicycle_patch = mpatches.Patch(color=(0, 175/255, 0), label='bicycle')
        motorcycle_patch = mpatches.Patch(color=(0, 0, 128/255), label='motorcycle')
        pedestrian_patch = mpatches.Patch(color=(255/255,69/255, 0/255), label='pedestrian')
        traffic_cone_patch = mpatches.Patch(color=(255/255, 0/255, 255/255), label='traffic cone')
        barrier_patch = mpatches.Patch(color=(173/255,255/255,47/255), label='barrier')  
        #fig = plt.plot(figsize=(24,24))
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.legend(handles=[car_patch, truck_patch,trailer_patch,bus_patch,construction_vehicle_patch,bicycle_patch,motorcycle_patch,pedestrian_patch,traffic_cone_patch,barrier_patch],loc='lower left',prop={'size': 20},ncol=3)

        if out_path is not None:
            plt.savefig(out_path)

        if verbose:
            plt.show()
        """
        fig = plt.plot(figsize=(24,24))
        
        lidar_data = {}
        for channel, sample_data_channel_token in record['data'].items():
            sd_record = self.nusc.get('sample_data', sample_data_channel_token)
            sensor_modality = sd_record['sensor_modality']
            

            if sensor_modality == 'lidar':
                lidar_data[channel] = sample_data_channel_token
        
        for i, (_, sd_token) in enumerate(lidar_data.items()):
            self.render_target_position(thr, ground_truth_bboxes,ground_truth_type_for_this_frame,inference_result_in_nuscenes_format,sd_token, box_vis_level=box_vis_level, ax=None, nsweeps=nsweeps,bird_eye_view_with_map=bird_eye_view_with_map,verbose=False)
            #fig.set_title("TOP LIDAR",fontsize=40)
        
        '''
        car_patch = mpatches.Patch(color=(0/255, 0/255, 230/255), label='car')
        truck_patch = mpatches.Patch(color=(70/255, 130/255, 180/255), label='truck')
        trailer_patch = mpatches.Patch(color=(138/255, 43/255, 226/255), label='trailer')
        bus_patch = mpatches.Patch(color=(0/255, 255/255, 0/255), label='bus')
        construction_vehicle_patch = mpatches.Patch(color=(255/255, 255/255, 0), label='construction vehicle')
        bicycle_patch = mpatches.Patch(color=(0, 175/255, 0), label='bicycle')
        motorcycle_patch = mpatches.Patch(color=(0, 0, 128/255), label='motorcycle')
        pedestrian_patch = mpatches.Patch(color=(255/255,69/255, 0/255), label='pedestrian')
        traffic_cone_patch = mpatches.Patch(color=(255/255, 0/255, 255/255), label='traffic cone')
        barrier_patch = mpatches.Patch(color=(173/255,255/255,47/255), label='barrier')  
        #fig = plt.plot(figsize=(24,24))
        plt.legend(handles=[car_patch, truck_patch,trailer_patch,bus_patch,construction_vehicle_patch,bicycle_patch,motorcycle_patch,pedestrian_patch,traffic_cone_patch,barrier_patch],loc='best',prop={'size': 10},ncol=3)
        '''
        if out_path is not None:
            plt.savefig(out_path)

        if verbose:
            plt.show()
        
    def render_egoposes_on_map(self,
                               log_location: str,
                               scene_tokens,
                               close_dist = 100,
                               color_fg = (167, 174, 186),
                               color_bg = (255, 255, 255),
                               out_path: str = None) -> None:
        """
        Renders ego poses a the map. These can be filtered by location or scene.
        :param log_location: Name of the location, e.g. "singapore-onenorth", "singapore-hollandvillage",
                             "singapore-queenstown' and "boston-seaport".
        :param scene_tokens: Optional list of scene tokens.
        :param close_dist: Distance in meters for an ego pose to be considered within range of another ego pose.
        :param color_fg: Color of the semantic prior in RGB format (ignored if map is RGB).
        :param color_bg: Color of the non-semantic prior in RGB format (ignored if map is RGB).
        :param out_path: Optional path to save the rendered figure to disk.
        """
        # Get logs by location.
        log_tokens = [log['token'] for log in self.nusc.log if log['location'] == log_location]
        assert len(log_tokens) > 0, 'Error: This split has 0 scenes for location %s!' % log_location

        # Filter scenes.
        scene_tokens_location = [e['token'] for e in self.nusc.scene if e['log_token'] in log_tokens]
        if scene_tokens is not None:
            scene_tokens_location = [t for t in scene_tokens_location if t in scene_tokens]
        if len(scene_tokens_location) == 0:
            print('Warning: Found 0 valid scenes for location %s!' % log_location)

        map_poses = []
        map_mask = None

        print('Adding ego poses to map...')
        for scene_token in tqdm(scene_tokens_location):

            # Get records from the database.
            scene_record = self.nusc.get('scene', scene_token)
            log_record = self.nusc.get('log', scene_record['log_token'])
            map_record = self.nusc.get('map', log_record['map_token'])
            map_mask = map_record['mask']

            # For each sample in the scene, store the ego pose.
            sample_tokens = self.nusc.field2token('sample', 'scene_token', scene_token)
            for sample_token in sample_tokens:
                sample_record = self.nusc.get('sample', sample_token)

                # Poses are associated with the sample_data. Here we use the lidar sample_data.
                sample_data_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
                pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])

                # Calculate the pose on the map and append.
                map_poses.append(np.concatenate(
                    map_mask.to_pixel_coords(pose_record['translation'][0], pose_record['translation'][1])))

        # Compute number of close ego poses.
        print('Creating plot...')
        map_poses = np.vstack(map_poses)
        dists = sklearn.metrics.pairwise.euclidean_distances(map_poses * map_mask.resolution)
        close_poses = np.sum(dists < close_dist, axis=0)

        if len(np.array(map_mask.mask()).shape) == 3 and np.array(map_mask.mask()).shape[2] == 3:
            # RGB Colour maps.
            mask = map_mask.mask()
        else:
            # Monochrome maps.
            # Set the colors for the mask.
            mask = Image.fromarray(map_mask.mask())
            mask = np.array(mask)

            maskr = color_fg[0] * np.ones(np.shape(mask), dtype=np.uint8)
            maskr[mask == 0] = color_bg[0]
            maskg = color_fg[1] * np.ones(np.shape(mask), dtype=np.uint8)
            maskg[mask == 0] = color_bg[1]
            maskb = color_fg[2] * np.ones(np.shape(mask), dtype=np.uint8)
            maskb[mask == 0] = color_bg[2]
            mask = np.concatenate((np.expand_dims(maskr, axis=2),
                                   np.expand_dims(maskg, axis=2),
                                   np.expand_dims(maskb, axis=2)), axis=2)

        # Plot.
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(mask)
        title = 'Number of ego poses within {}m in {}'.format(close_dist, log_location)
        ax.set_title(title, color='k')
        sc = ax.scatter(map_poses[:, 0], map_poses[:, 1], s=10, c=close_poses)
        color_bar = plt.colorbar(sc, fraction=0.025, pad=0.04)
        plt.rcParams['figure.facecolor'] = 'black'
        color_bar_ticklabels = plt.getp(color_bar.ax.axes, 'yticklabels')
        plt.setp(color_bar_ticklabels, color='k')
        plt.rcParams['figure.facecolor'] = 'white'  # Reset for future plots.

        if out_path is not None:
            plt.savefig(out_path)

