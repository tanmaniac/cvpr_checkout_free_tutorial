import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

from tracker import Tracker, KP_INDICES

VIDEO_DIR = Path("/mnt/c/Users/tanma/projects/MERL_Shopping_Dataset"
                 "/Videos_MERL_Shopping_Dataset/Videos_MERL_Shopping_Dataset")

KEYPOINTS_OF_INTEREST = {
    0:  "Head",
    5:  "L.Shldr",
    6:  "R.Shldr",
    7:  "L.Elbow",
    8:  "R.Elbow",
    9:  "L.Wrist",
    10: "R.Wrist",
}

# All COCO keypoint indices not in our interest set (eyes, ears, hips, knees, ankles)
HIDE_KEYPOINT_INDICES = list(set(range(17)) - set(KEYPOINTS_OF_INTEREST.keys()))

TRACK_COLORS_BGR = [
    (255, 56,  56),
    (56,  255, 56),
    (56,  56,  255),
    (255, 157, 56),
    (161, 56,  255),
    (56,  255, 255),
    (255, 56,  255),
    (255, 255, 56),
    (255, 128, 0),
    (0,   200, 255),
]

# Labels for the 7 keypoints in the same order as KP_INDICES
KP_LABELS = [KEYPOINTS_OF_INTEREST[i] for i in sorted(KEYPOINTS_OF_INTEREST)]


def plot_filtered(result):
    """Plot skeleton for only the 7 keypoints of interest (no YOLO boxes/labels).

    Temporarily zeros out confidence for non-interest keypoints so that
    result.plot() skips their circles and skeleton connections.
    Track-coloured boxes are drawn separately by the main loop.
    """
    kp = result.keypoints
    if kp is not None and kp.conf is not None:
        orig_data = kp.data
        masked_data = orig_data.clone()
        masked_data[:, HIDE_KEYPOINT_INDICES, 2] = 0.0  # zero confidence in channel 2
        kp.data = masked_data
        annotated = result.plot(kpt_line=True, boxes=False, labels=False)
        kp.data = orig_data
    else:
        annotated = result.plot(boxes=False, labels=False)
    return annotated


def draw_keypoints_of_interest(frame, kp_xy_7x2, kp_conf_7, kp_conf_thresh=0.3):
    """Overlay highlighted circles on the 7 keypoints of interest.

    Args:
        frame       : BGR image to draw on (modified in-place)
        kp_xy_7x2  : (7, 2) array of keypoint coordinates from the track
        kp_conf_7  : (7,)   array of raw keypoint confidences from the track
        kp_conf_thresh: minimum confidence to draw a keypoint
    """
    for label, (x, y), conf in zip(KP_LABELS, kp_xy_7x2, kp_conf_7):
        if conf < kp_conf_thresh:
            continue
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)   # yellow fill
        cv2.circle(frame, (x, y), 8, (0, 0, 0), 1)        # black outline
        cv2.putText(frame, label, (x + 9, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="MERL Person Detection with YOLO Pose Estimation")
    parser.add_argument("--video", default=None,
                        help="Path to video file (default: first .mp4 in dataset dir)")
    parser.add_argument("--model", default="yolo11m-pose.pt",
                        help="YOLO pose model weights (default: yolo11m-pose.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--max-age", type=int, default=15,
                        help="Frames a track can coast without a detection (default: 15)")
    parser.add_argument("--min-hits", type=int, default=3,
                        help="Detections before a track is shown (default: 3)")
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
    else:
        mp4_files = sorted(VIDEO_DIR.glob("*.mp4"))
        if not mp4_files:
            raise FileNotFoundError(f"No .mp4 files found in {VIDEO_DIR}")
        video_path = mp4_files[0]

    print(f"Video:    {video_path}")
    print(f"Model:    {args.model}")
    print(f"Conf:     {args.conf}")
    print(f"min-hits: {args.min_hits}  max-age: {args.max_age}")

    model = YOLO(args.model)  # auto-downloads weights on first run

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    tracker = Tracker(min_hits=args.min_hits, max_age=args.max_age)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=0, conf=args.conf, verbose=False)
        r = results[0]

        # Extract numpy arrays from YOLO result
        if r.boxes is not None and len(r.boxes):
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()      # (N, 4)
            kp_xy      = r.keypoints.xy.cpu().numpy()    # (N, 17, 2)
            kp_conf    = r.keypoints.conf.cpu().numpy()  # (N, 17)
        else:
            boxes_xyxy = np.empty((0, 4))
            kp_xy      = np.empty((0, 17, 2))
            kp_conf    = np.empty((0, 17))

        confirmed = tracker.update(boxes_xyxy, kp_xy, kp_conf)

        annotated = plot_filtered(r)   # skeleton lines only (no YOLO boxes)

        for track in confirmed:
            color = TRACK_COLORS_BGR[track.track_id % len(TRACK_COLORS_BGR)]
            cx, cy, w, h = track.bbox
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1, th + 8)
            cv2.rectangle(annotated, (x1, label_y - th - 6), (x1 + tw + 4, label_y), color, -1)
            cv2.putText(annotated, label, (x1 + 2, label_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            draw_keypoints_of_interest(annotated, track.keypoints, track.kp_conf_7)

        n_people = len(confirmed)
        cv2.putText(annotated, f"Frame {frame_idx}  |  Tracked: {n_people}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("MERL Person Detection", annotated)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()
