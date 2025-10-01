from scipy.optimize import linear_sum_assignment
from helper.kalman_tracker import KalmanTracker
import numpy as np
from helper.utils import diou

class MultiObjectTracker:
    def __init__(self, max_lost=20, line_position=710):
        self.trackers = {}
        self.lost_frames = {}
        self.prev_centers = {}
        self.next_id = 0
        self.max_lost = max_lost
        self.count = 0
        self.line_position = line_position
        self.crossed_ids = set()

    def helper_center_fun(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2)/2.0, (y1 + y2)/2.0)

    def update_prev(self, detections):
        tracker_ids = list(self.trackers.keys())
        predicted_boxes = []
        for tid in tracker_ids:
            pred_box = self.trackers[tid].predict()
            predicted_boxes.append(pred_box)

        assigned_dets = set()
        assigned_trks = set()

        if len(detections) > 0 and len(predicted_boxes) > 0:
            cost_matrix = np.zeros((len(detections), len(predicted_boxes)))

            for d in range(len(detections)):
                for t in range(len(predicted_boxes)):
                    di = diou(detections[d], predicted_boxes[t])
                    cost_matrix[d, t] = 1-di

            row_idx, col_idx = linear_sum_assignment(cost_matrix)

            for d, t in zip(row_idx, col_idx):
                di = 1 - cost_matrix[d, t]
                if di > 0.1:
                    tid = tracker_ids[t]
                    self.trackers[tid].update(detections[d])  # update with full bbox
                    self.lost_frames[tid] = 0
                    assigned_dets.add(d)
                    assigned_trks.add(tid)

                    #Line crossing check
                    prev_cy = self.prev_centers.get(tid, None)
                    cx, cy = self.helper_center_fun(detections[d])
                    if prev_cy is not None and (prev_cy < self.line_position <= cy) and (tid not in self.crossed_ids):
                        self.count += 1
                        self.crossed_ids.add(tid)
                    self.prev_centers[tid] = cy

        for d in range(len(detections)):
            if d not in assigned_dets:
                self.trackers[self.next_id] = KalmanTracker()
                self.trackers[self.next_id].update(detections[d])  # initialize with full bbox
                cx, cy = self.helper_center_fun(detections[d])
                self.prev_centers[self.next_id] = cy
                self.lost_frames[self.next_id] = 0
                self.next_id += 1

        for tid in list(self.trackers.keys()):
            if tid not in assigned_trks:
                self.lost_frames[tid] = self.lost_frames.get(tid, 0) + 1

        to_delete = []
        for tid, lost in self.lost_frames.items():
            if lost > self.max_lost:
                to_delete.append(tid)

        for tid in to_delete:
            self.trackers.pop(tid, None)
            self.lost_frames.pop(tid, None)
            self.prev_centers.pop(tid, None)
            self.crossed_ids.discard(tid)

        results = {}
        for tid, tracker in self.trackers.items():
            results[tid] = tracker.get_state()  # [x1, y1, x2, y2]

        return results, self.count
