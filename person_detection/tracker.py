"""Kalman filter-based multi-person tracker.

Each person gets a stable track_id that persists across frames, including
recovery after short occlusions (up to max_age frames).

State design: 36-dim constant-velocity model with dt=1.
  - 18 position dims: [cx, cy, w, h, kp0x, kp0y, kp1x, kp1y, ..., kp6x, kp6y]
  - 18 velocity dims
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

# 7 keypoints of interest by COCO index (must match KEYPOINTS_OF_INTEREST in detect_people.py)
KP_INDICES = [0, 5, 6, 7, 8, 9, 10]

N_POS   = 18   # 4 bbox + 14 keypoint coords (7 kps × 2)
N_STATE = 36   # N_POS × 2  (position + velocity)

# ---------------------------------------------------------------------------
# Module-level matrices — built once, shared across all Track instances
# ---------------------------------------------------------------------------

# F (36×36): block_diag of 18 copies of [[1, 1], [0, 1]]
_block_f = np.array([[1.0, 1.0],
                     [0.0, 1.0]])
F = np.zeros((N_STATE, N_STATE))
for _i in range(N_POS):
    F[2*_i:2*_i+2, 2*_i:2*_i+2] = _block_f

# H (18×36): block_diag of 18 copies of [1, 0]  — extracts position only
H = np.zeros((N_POS, N_STATE))
for _i in range(N_POS):
    H[_i, 2*_i] = 1.0

# Q unit matrix (36×36): block_diag of 18 copies of outer([0.5, 1], [0.5, 1])
# Discrete Wiener process noise for CV; multiply by sigma_q² at predict time.
_q_vec = np.array([0.5, 1.0])
_q_block = np.outer(_q_vec, _q_vec)
_Q_UNIT = np.zeros((N_STATE, N_STATE))
for _i in range(N_POS):
    _Q_UNIT[2*_i:2*_i+2, 2*_i:2*_i+2] = _q_block


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def iou(a, b):
    """Intersection-over-union for two boxes in [cx, cy, w, h] format."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

class Track:
    """Single-person Kalman filter track (constant-velocity model)."""

    _id_counter = 0

    def __init__(self, meas_vec, sigma_a=1.0, r_bbox=5.0, r_kp=5.0):
        Track._id_counter += 1
        self.track_id = Track._id_counter

        self.sigma_a = sigma_a
        self.r_bbox  = r_bbox
        self.r_kp    = r_kp

        # State: position dims initialised from measurement; velocities = 0
        self.x = np.zeros(N_STATE)
        self.x[::2] = meas_vec

        # Covariance: tight on position, large on velocity (unknown initially)
        self.P = np.eye(N_STATE) * 10.0
        for _i in range(N_POS):
            self.P[2*_i+1, 2*_i+1] = 1000.0  # high uncertainty on velocities

        self.age             = 0
        self.hits            = 1
        self.hit_streak      = 1
        self.frames_since_det = 0

        # Keypoint confidences (7,); persists during coasting to avoid flicker
        self.kp_conf_7 = np.zeros(7)

    # ------------------------------------------------------------------
    def predict(self):
        """Advance state by one time step."""
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + (self.sigma_a ** 2) * _Q_UNIT
        self.age += 1
        self.frames_since_det += 1

    # ------------------------------------------------------------------
    def update(self, meas_vec, kp_mask, kp_conf_raw):
        """Kalman update with a new detection.

        Args:
            meas_vec   : (18,) measurement [cx,cy,w,h, kp0x,kp0y,...,kp6x,kp6y]
            kp_mask    : (7,)  boolean — True if keypoint confidence >= threshold
            kp_conf_raw: (7,)  raw keypoint confidences (stored for display)
        """
        # Build dynamic R — inflate unconfident keypoint dims to 1e6
        R = np.zeros((N_POS, N_POS))
        for d in range(4):          # bbox dims
            R[d, d] = self.r_bbox ** 2
        for j in range(7):          # 7 keypoints × 2 coords
            r_val = self.r_kp ** 2 if kp_mask[j] else 1e6
            R[4 + 2*j,     4 + 2*j    ] = r_val
            R[4 + 2*j + 1, 4 + 2*j + 1] = r_val

        # Standard Kalman gain and update
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = meas_vec - H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(N_STATE) - K @ H) @ self.P

        self.hits += 1
        self.hit_streak += 1
        self.frames_since_det = 0
        self.kp_conf_7 = kp_conf_raw.copy()

    # ------------------------------------------------------------------
    @property
    def bbox(self):
        """Return [cx, cy, w, h] clamped to non-negative size."""
        pos = self.x[::2][:4]
        return [pos[0], pos[1], max(float(pos[2]), 1.0), max(float(pos[3]), 1.0)]

    @property
    def keypoints(self):
        """Return (7, 2) array of keypoint positions."""
        return self.x[::2][4:].reshape(7, 2)

    # ------------------------------------------------------------------
    def is_confirmed(self, min_hits):
        return self.hits >= min_hits

    def is_dead(self, max_age):
        return self.frames_since_det > max_age


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class Tracker:
    """Hungarian + Kalman multi-person tracker."""

    def __init__(self, min_hits=3, max_age=15, iou_gate=0.1,
                 sigma_a=1.0, r_bbox=5.0, r_kp=5.0):
        self.min_hits = min_hits
        self.max_age  = max_age
        self.iou_gate = iou_gate
        self.sigma_a  = sigma_a
        self.r_bbox   = r_bbox
        self.r_kp     = r_kp
        self.tracks   = []

    def update(self, boxes_xyxy, kp_xy, kp_conf, kp_conf_thresh=0.3):
        """Run one tracking cycle.

        Args:
            boxes_xyxy : (N, 4) numpy — YOLO detections in x1y1x2y2
            kp_xy      : (N, 17, 2) numpy — all 17 COCO keypoint coords
            kp_conf    : (N, 17) numpy — all 17 COCO keypoint confidences
            kp_conf_thresh: float — below this confidence a keypoint is treated
                            as unreliable (R inflated to 1e6)

        Returns:
            List of confirmed Track objects (hits >= min_hits).
        """
        # 1. Predict all existing tracks
        for track in self.tracks:
            track.predict()

        # 2. Convert detections to cxcywh; extract the 7 kps of interest
        n_dets = len(boxes_xyxy)
        if n_dets > 0:
            boxes_cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
            boxes_cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
            boxes_w  =  boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
            boxes_h  =  boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
            boxes_cxcywh = np.stack([boxes_cx, boxes_cy, boxes_w, boxes_h], axis=1)

            kp_xy_7   = kp_xy[:, KP_INDICES, :]    # (N, 7, 2)
            kp_conf_7 = kp_conf[:, KP_INDICES]      # (N, 7)
        else:
            boxes_cxcywh = np.empty((0, 4))
            kp_xy_7      = np.empty((0, 7, 2))
            kp_conf_7    = np.empty((0, 7))

        n_tracks = len(self.tracks)

        # 3–5. Build cost matrix and run Hungarian assignment with IoU gate
        matched_track_idx = set()
        matched_det_idx   = set()
        matches           = []

        if n_tracks > 0 and n_dets > 0:
            cost = np.zeros((n_tracks, n_dets))
            for i, track in enumerate(self.tracks):
                for j in range(n_dets):
                    cost[i, j] = iou(track.bbox, boxes_cxcywh[j])

            row_ind, col_ind = linear_sum_assignment(1.0 - cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= self.iou_gate:
                    matches.append((r, c))
                    matched_track_idx.add(r)
                    matched_det_idx.add(c)

        # 6. Update matched tracks
        for track_idx, det_idx in matches:
            track       = self.tracks[track_idx]
            box         = boxes_cxcywh[det_idx]
            kp_xy_det   = kp_xy_7[det_idx]     # (7, 2)
            kp_conf_det = kp_conf_7[det_idx]   # (7,)
            kp_mask     = kp_conf_det >= kp_conf_thresh

            meas_vec = np.concatenate([box, kp_xy_det.flatten()])
            track.update(meas_vec, kp_mask, kp_conf_det)

        # 7. Age unmatched tracks (hit_streak reset; frames_since_det already
        #    incremented inside predict())
        for i in range(n_tracks):
            if i not in matched_track_idx:
                self.tracks[i].hit_streak = 0

        # 8. Spawn new tracks for unmatched detections
        for j in range(n_dets):
            if j not in matched_det_idx:
                box         = boxes_cxcywh[j]
                kp_xy_det   = kp_xy_7[j]
                kp_conf_det = kp_conf_7[j]
                meas_vec    = np.concatenate([box, kp_xy_det.flatten()])
                new_track   = Track(meas_vec, self.sigma_a, self.r_bbox, self.r_kp)
                new_track.kp_conf_7 = kp_conf_det.copy()
                self.tracks.append(new_track)

        # 9. Prune dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead(self.max_age)]

        # 10. Return confirmed tracks
        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]
