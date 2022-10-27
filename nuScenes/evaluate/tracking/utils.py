# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

import unittest
import warnings
from typing import Optional, Dict

import numpy as np
import sklearn
import os

try:
    import motmetrics
    from motmetrics.metrics import MetricsHost
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as motmetrics was not found!')

from evaluate.tracking.data_classes import TrackingMetrics
from evaluate.tracking.mot import MOTAccumulatorCustom
from evaluate.tracking.render import TrackingRenderer
from evaluate.tracking.metrics import motar, mota_custom, motp_custom, faf, track_initialization_duration, \
    longest_gap_duration, num_fragmentations_custom


def category_to_tracking_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes tracking classes.
    :param category_name: Generic nuScenes class.
    :return: nuScenes tracking class.
    """
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
        'vehicle.truck': 'truck'
    }

    if category_name in tracking_mapping:
        return tracking_mapping[category_name]
    else:
        return None


def metric_name_to_print_format(metric_name) -> str:
    """
    Get the standard print format (numerical precision) for each metric.
    :param metric_name: The lowercase metric name.
    :return: The print format.
    """
    if metric_name in ['amota', 'amotp', 'motar', 'recall', 'mota', 'motp']:
        print_format = '%.3f'
    elif metric_name in ['tid', 'lgd']:
        print_format = '%.2f'
    elif metric_name in ['faf']:
        print_format = '%.1f'
    else:
        print_format = '%d'
    return print_format


def print_final_metrics(metrics: TrackingMetrics) -> None:
    """
    Print metrics to stdout.
    :param metrics: The output of evaluate().
    """
    print('\n### Final results ###')

    # Print per-class metrics.
    metric_names = metrics.label_metrics.keys()
    print('\nPer-class results:')
    print('\t\t', end='')
    print('\t'.join([m.upper() for m in metric_names]))

    class_names = metrics.class_names
    max_name_length = 7
    for class_name in class_names:
        print_class_name = class_name[:max_name_length].ljust(max_name_length + 1)
        print('%s' % print_class_name, end='')

        for metric_name in metric_names:
            val = metrics.label_metrics[metric_name][class_name]
            print_format = '%f' if np.isnan(val) else metric_name_to_print_format(metric_name)
            print('\t%s' % (print_format % val), end='')

        print()

    # Print high-level metrics.
    print('\nAggregated results:')
    for metric_name in metric_names:
        val = metrics.compute_metric(metric_name, 'all')
        print_format = metric_name_to_print_format(metric_name)
        print('%s\t%s' % (metric_name.upper(), print_format % val))

    print('Eval time: %.1fs' % metrics.eval_time)
    print()


def print_threshold_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print only a subset of the metrics for the current class and threshold.
    :param metrics: A dictionary representation of the metrics.
    """
    # Specify threshold name and metrics.
    assert len(metrics['mota_custom'].keys()) == 1
    threshold_str = list(metrics['mota_custom'].keys())[0]
    mota=metrics['mota_custom'][threshold_str]
    motar_val = metrics['motar'][threshold_str]
    motp = metrics['motp_custom'][threshold_str]
    recall = metrics['recall'][threshold_str]
    num_frames = metrics['num_frames'][threshold_str]
    num_objects = metrics['num_objects'][threshold_str]
    num_predictions = metrics['num_predictions'][threshold_str]
    num_false_positives = metrics['num_false_positives'][threshold_str]
    num_misses = metrics['num_misses'][threshold_str]
    num_switches = metrics['num_switches'][threshold_str]
    num_matches = metrics['num_matches'][threshold_str]

    # Print.
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s'
          % ('\t','MOTAR','MOTA' ,'MOTP', 'Recall', 'Frames',
             'GT', 'GT-Mtch', 'GT-Miss', 'GT-IDS',
             'Pred', 'Pred-TP', 'Pred-FP', 'Pred-IDS',))
    print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d'
          % (threshold_str, motar_val,mota, motp, recall, num_frames,
             num_objects, num_matches, num_misses, num_switches,
             num_predictions, num_matches, num_false_positives, num_switches))
    #print('MOTA {}'.format(str(round(mota,2))))
    print()

    # Check metrics for consistency.
    assert num_objects == num_matches + num_misses + num_switches
    assert num_predictions == num_matches + num_false_positives + num_switches


def create_motmetrics() -> MetricsHost:
    """
    Creates a MetricsHost and populates it with default and custom metrics.
    It does not populate the global metrics which are more time consuming.
    :return The initialized MetricsHost object with default MOT metrics.
    """
    # Create new metrics host object.
    mh = MetricsHost()

    # Suppress deprecation warning from py-motmetrics.
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # Register standard metrics.
    fields = [
        'num_frames', 'obj_frequencies', 'num_matches', 'num_switches', 'num_false_positives', 'num_misses',
        'num_detections', 'num_objects', 'num_predictions', 'mostly_tracked', 'mostly_lost', 'num_fragmentations',
        'motp', 'mota', 'precision', 'recall', 'track_ratios'
    ]
    for field in fields:
        mh.register(getattr(motmetrics.metrics, field), formatter='{:d}'.format)

    # Reenable deprecation warning.
    warnings.filterwarnings('default', category=DeprecationWarning)

    # Register custom metrics.
    # Specify all inputs to avoid errors incompatibility between type hints and py-motmetric's introspection.
    mh.register(motar, ['num_matches', 'num_misses', 'num_switches', 'num_false_positives', 'num_objects'],
                formatter='{:.2%}'.format, name='motar')
    mh.register(mota_custom, ['num_misses', 'num_switches', 'num_false_positives', 'num_objects'],
                formatter='{:.2%}'.format, name='mota_custom')
    mh.register(motp_custom, ['num_detections'],
                formatter='{:.2%}'.format, name='motp_custom')
    mh.register(num_fragmentations_custom, ['obj_frequencies'],
                formatter='{:.2%}'.format, name='num_fragmentations_custom')
    mh.register(faf, ['num_false_positives', 'num_frames'],
                formatter='{:.2%}'.format, name='faf')
    mh.register(track_initialization_duration, ['obj_frequencies'],
                formatter='{:.2%}'.format, name='tid')
    mh.register(longest_gap_duration, ['obj_frequencies'],
                formatter='{:.2%}'.format, name='lgd')

    return mh

def render_for_the_best_threshold(nusc, class_name, threshold, tracks_gt, tracks_pred,threshold_render_classes,threshold_output_dir,dist_th_tp):
    # Groundtruth and tracker contain lists for every single frame containing lists detections.
    for scene_id in tracks_gt.keys():
        # Initialize accumulator and frame_id for this scene
        acc = MOTAccumulatorCustom()
        frame_id = 0  # Frame ids must be unique across all scenes
        # Retrieve GT and preds.
        scene_tracks_gt = tracks_gt[scene_id]
        scene_tracks_pred = tracks_pred[scene_id]
        # Visualize the boxes in this frame.
        if class_name in threshold_render_classes:
            save_path = os.path.join(threshold_output_dir,'threshold',str(scene_id), class_name, str(round(threshold,2)))
            os.makedirs(save_path, exist_ok=True)
            renderer = TrackingRenderer(scene_id, save_path)
        else:
            renderer = None
        for timestamp in scene_tracks_gt.keys():
            # Select only the current class.
            frame_gt = scene_tracks_gt[timestamp]
            frame_pred = scene_tracks_pred[timestamp]
            frame_gt = [f for f in frame_gt if f.tracking_name == class_name]
            frame_pred = [f for f in frame_pred if f.tracking_name == class_name and f.tracking_score >= threshold]
            # Abort if there are neither GT nor pred boxes.
            gt_ids = [gg.tracking_id for gg in frame_gt]
            pred_ids = [tt.tracking_id for tt in frame_pred]
            if len(gt_ids) == 0 and len(pred_ids) == 0:
                continue
            # Calculate distances.
            # Note that the distance function is hard-coded to achieve significant speedups via vectorization.
            if len(frame_gt) == 0 or len(frame_pred) == 0:
                distances = np.ones((0, 0))
            else:
                gt_boxes = np.array([b.translation[:2] for b in frame_gt])
                pred_boxes = np.array([b.translation[:2]
                                      for b in frame_pred])
                distances = sklearn.metrics.pairwise.euclidean_distances(
                    gt_boxes, pred_boxes)
            # Distances that are larger than the threshold won't be associated.
            assert len(distances) == 0 or not np.all(np.isnan(distances))
            distances[distances >= dist_th_tp] = np.nan
            # Accumulate results.
            # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
            acc.update(gt_ids, pred_ids, distances, frameid=frame_id)
            # Store scores of matches, which are used to determine recall thresholds.
            events = acc.events.loc[frame_id]
            # Render the boxes in this frame.
            if class_name in threshold_render_classes:
                renderer.render(nusc,events, timestamp, frame_gt, frame_pred)
            frame_id += 1