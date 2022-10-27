# nuScenes dev-kit.
# Code written by Holger Caesar, Caglayan Dicle, Varun Bankiti, and Alex Lang, 2019.

import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from pyquaternion import Quaternion

from evaluate.common.render import setup_axis
from evaluate.tracking.data_classes import TrackingBox, TrackingMetricDataList
from evaluate.tracking.data_classes import TrackingConfig
from evaluate.util.lidar_data import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
import matplotlib.patches as mpatches

Axis = Any


def summary_plot(cfg: TrackingConfig,
                 md_list: TrackingMetricDataList,
                 ncols: int = 2,
                 savepath: str = None) -> None:
    """
    Creates a summary plot with which includes all traditional metrics for each class.
    :param cfg: A TrackingConfig object.
    :param md_list: TrackingMetricDataList instance.
    :param ncols: How many columns the resulting plot should have.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Select metrics and setup plot.
    rel_metrics = ['motar', 'motp', 'mota', 'recall', 'mt', 'ml', 'faf', 'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
    n_metrics = len(rel_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5 * ncols, 5 * nrows))

    # For each metric, plot all the classes in one diagram.
    for ind, metric_name in enumerate(rel_metrics):
        row = ind // ncols
        col = np.mod(ind, ncols)
        recall_metric_curve(cfg, md_list, metric_name, ax=axes[row, col])

    # Set layout with little white space and save to disk.
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def recall_metric_curve(cfg: TrackingConfig,
                        md_list: TrackingMetricDataList,
                        metric_name: str,
                        savepath: str = None,
                        ax: Axis = None) -> None:
    """
    Plot the recall versus metric curve for the given metric.
    :param cfg: A TrackingConfig object.
    :param md_list: TrackingMetricDataList instance.
    :param metric_name: The name of the metric to plot.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render or None to create a new axis.
    """
    min_recall = cfg.min_recall  # Minimum recall value from config.
    # Setup plot.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    ax = setup_axis(xlabel='Recall', ylabel=metric_name.upper(),
                    xlim=1, ylim=None, min_recall=min_recall, ax=ax, show_spines='bottomleft')

    # Plot the recall vs. precision curve for each detection class.
    for tracking_name, md in md_list.md.items():
        # Get values.
        confidence = md.confidence
        recalls = md.recall_hypo
        values = md.get_metric(metric_name)

        # Filter unachieved recall thresholds.
        valid = np.where(np.logical_not(np.isnan(confidence)))[0]
        if len(valid) == 0:
            continue
        first_valid = valid[0]
        assert not np.isnan(confidence[-1])
        recalls = recalls[first_valid:]
        values = values[first_valid:]

        ax.plot(recalls,
                values,
                label='%s' % cfg.pretty_tracking_names[tracking_name],
                color=cfg.tracking_colors[tracking_name])

    # Scale count statistics and FAF logarithmically.
    if metric_name in ['mt', 'ml', 'faf', 'tp', 'fp', 'fn', 'ids', 'frag']:
        ax.set_yscale('symlog')

    if metric_name in ['amota', 'motar', 'recall', 'mota']:
        # Some metrics have an upper bound of 1.
        ax.set_ylim(0, 1)
    elif metric_name != 'motp':
        # For all other metrics except MOTP we set a lower bound of 0.
        ax.set_ylim(bottom=0)

    ax.legend(loc='upper right', borderaxespad=0)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


class TrackingRenderer:
    """
    Class that renders the tracking results in BEV and saves them to a folder.
    """
    def __init__(self,scene_id, save_path):
        """
        :param save_path:  Output path to save the renderings.
        """
        self.save_path = save_path
        self.id2color = {}  # The color of each track.
        self.scene_id=scene_id

    def render(self, nusc, events: DataFrame, timestamp: int, frame_gt: List[TrackingBox], frame_pred: List[TrackingBox]) \
            -> None:
        """
        Render function for a given scene timestamp
        :param events: motmetrics events for that particular
        :param timestamp: timestamp for the rendering
        :param frame_gt: list of ground truth boxes
        :param frame_pred: list of prediction boxes
        """
        # Init.
        #print('Rendering {}'.format(timestamp))
        switches = events[events.Type == 'SWITCH']
        switch_ids = switches.HId.values
        switch_dis = switches.D.values
        fp=events[events.Type == 'FP']
        fp_ids = fp.HId.values 
        fp_dis=fp.D.values
        fn=events[events.Type=='MISS']
        fn_ids = fn.OId.values
        match=events[events.Type=='MATCH']
        match_ids=match.HId.values
        match_dis=match.D.values

        fig, ax = plt.subplots(figsize=(12,12))
       
        #ground_truth = mpatches.Patch(color=(0/255, 0/255, 0/255), label='ground truth')
        #false_nagative = mpatches.Patch(color=(70/255, 130/255, 180/255), label='false negative')
        #ID_switch = mpatches.Patch(label='ID_switch, random colour')
        #matched = mpatches.Patch(label='matched, random colour with distance shown')
        
        #fig.legend(handles=[ground_truth, false_nagative, ID_switch, matched],loc='upper left',prop={'size': 20},ncol=1)

        # Plot GT boxes.
        for b in frame_gt:
            box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
            #sample_token=b.sample_token
            
            if b.tracking_id in fn_ids:
                color = 'b'
                text='fn'
                center_bottom=box.center
                box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=1,linestyle='dashed')
                ax.text(center_bottom[0], center_bottom[1],text , fontsize = 15, color=color)
                #ax.scatter(center_bottom[0], center_bottom[1],color=color)
            else:
                color = 'k'
                box.render(ax, view=np.eye(4), colors=(color, color, color))

        # Plot predicted boxes.
        for b in frame_pred:
            box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)

            # Determine color for this tracking id.
            if b.tracking_id not in self.id2color.keys():
                self.id2color[b.tracking_id] = (float(hash(str(b.tracking_id)+ 'r') % 256) / 255,
                                                float(hash(str(b.tracking_id) + 'g') % 256) / 255,
                                                float(hash(str(b.tracking_id) + 'b') % 256) / 255)

            # Render box. Highlight identity switches in red.
            if b.tracking_id in switch_ids:
                color = self.id2color[b.tracking_id]
                idx=switch_ids.tolist().index(b.tracking_id)
                id=b.tracking_id
                text='id switch'+str(round(switch_dis[idx], 2))
                text2=str(round(b.tracking_score, 2))
                text3=str(id)
                box.render(ax, view=np.eye(4), colors=(color, color, color),linewidth=1)
                center_bottom=box.center
                ax.text(center_bottom[0], center_bottom[1]+1,text , fontsize = 15, color=color)
                ax.text(center_bottom[0], center_bottom[1],text3 , fontsize = 7, color=color)
                ax.text(center_bottom[0], center_bottom[1]-1,text2 , fontsize = 7, color=color)
                #ax.scatter(center_bottom[0], center_bottom[1],color=color)

            elif b.tracking_id in fp_ids:
                color = self.id2color[b.tracking_id]
                idx=fp_ids.tolist().index(b.tracking_id)
                id=b.tracking_id
                text='fp'
                text3=str(id)
                text2=str(round(b.tracking_score, 2))
                box.render(ax, view=np.eye(4), colors=(color, color, color),linewidth=1,linestyle=':')
                center_bottom=box.center
                ax.text(center_bottom[0], center_bottom[1]+1,text , fontsize = 15, color=color)
                ax.text(center_bottom[0], center_bottom[1],text3 , fontsize = 7, color=color)
                ax.text(center_bottom[0], center_bottom[1]-1,text2 , fontsize = 7, color=color)
                #ax.scatter(center_bottom[0], center_bottom[1],color=color)
            elif b.tracking_id in match_ids:
                color = self.id2color[b.tracking_id]
                idx=match_ids.tolist().index(b.tracking_id)
                id=match_ids[idx]
                text=str(round(match_dis[idx], 2))
                text3=str(id)
                text2=str(round(b.tracking_score, 2))
                box.render(ax, view=np.eye(4), colors=(color, color, color),linewidth=1)
                center_bottom=box.center
                ax.text(center_bottom[0], center_bottom[1]+1,text , fontsize = 15, color=color)
                ax.text(center_bottom[0], center_bottom[1],text3 , fontsize = 7, color=color)
                ax.text(center_bottom[0], center_bottom[1]-1,text2 , fontsize = 7, color=color)

            else:
                color = self.id2color[b.tracking_id]
                box.render(ax, view=np.eye(4), colors=(color, color, color), linestyle='dashed')

        '''
        # Plot lidar point cloud
        scenes = nusc.scene
        samples = nusc.sample
        sample_data=nusc.sample_data
        samples_for_this_scene=[x for x in samples if x['scene_token']==self.scene_id]
        for x in samples_for_this_scene:
            if x['timestamp']==timestamp:
                sample_token=x['token']
                break

        #sample_token = frame_gt[0].sample_token
        record = nusc.get('sample', sample_token)
        lidar_data = {}
        for channel, sample_data_channel_token in record['data'].items():
            if channel == 'LIDAR_TOP':
                sd_record = nusc.get('sample_data', sample_data_channel_token)
                lidar_data[channel] = sample_data_channel_token
                sample_rec = nusc.get('sample', sd_record['sample_token'])
                chan = sd_record['channel']
        pc = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, 'LIDAR_TOP',nsweeps=1)

        #points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
        points = pc.points[:2,:]
        colors = 'k'
        point_scale = 0.01
        ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
        '''

        # Plot ego pose.
        plt.scatter(0, 0, s=96, facecolors='none', edgecolors='k', marker='o')
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)

        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, '{}.png'.format(timestamp)))
        plt.close(fig)
