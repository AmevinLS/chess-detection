import argparse
from typing import *
from unittest.mock import patch

import cv2
import numpy as np
from tqdm import tqdm

from .event_detect import EventDetector, EventType
from .field_detect import FieldDetectorHough
from .piece_classify import PieceClassifier
from .piece_detect import PieceDetectorHough
from .util import draw_boxes, draw_event, draw_labels, draw_points


### TODO: Implement tracking.
def points_to_board_coords(
    sqr_centers: np.ndarray, points: np.ndarray
) -> np.ndarray:
    sqr_coords = [(i, j) for i in range(8) for j in range(8)]
    dists = np.empty((len(points), len(sqr_centers)))
    for i, point in enumerate(points):
        for j, sqr_center in enumerate(sqr_centers):
            dists[i, j] = np.sum((point - sqr_center) ** 2)
    best_sqr_inds = np.argmin(dists, axis=1)

    point_coords = np.take(sqr_coords, best_sqr_inds, axis=0)
    return point_coords


def calc_presence_matrix(
    fields_centers: np.ndarray, piece_centers: np.ndarray
) -> np.ndarray:
    # TODO: (Later probably?) improve this
    # HACK: this is pretty stupid (trying to "rotate" board here depending on the change in coordinates)
    # delta = fields_centers[1] - fields_centers[0]
    # if (delta[0] <= 0 and delta[1] >= 0) or (delta[0] <= 0 and delta[1] <= 0):
    #     field_matrix = fields_centers.reshape((8,8,2))
    #     field_matrix = field_matrix[:, ::-1]
    #     fields_centers = fields_centers.reshape(64, 2)
    # elif (delta[0] >= 0 and delta[1] >= 0) or (delta[0] >= 0 and delta[1] <= 0):
    #     field_matrix = fields_centers.reshape((8,8,2))
    #     field_matrix = field_matrix[::-1, :]
    #     fields_centers = fields_centers.reshape(64, 2)
    # elif (delta[0] <= 0 and delta[1] <= 0) or (delta[0] <= 0 and delta[1] >= 0):
    #     fields_centers = fields_centers[::-1]

    presence_matrix = np.zeros((8, 8), dtype=np.bool_)
    vert = np.repeat(np.arange(0, 8), 8)
    horiz = np.tile(np.arange(0, 8), 8)
    dists = np.linalg.norm(
        piece_centers[:, np.newaxis] - fields_centers[np.newaxis], axis=-1
    )
    field_inds = dists.argmin(axis=1)
    presence_matrix[vert[field_inds], horiz[field_inds]] = True
    return presence_matrix


def match_points(
    points_old: np.ndarray, points_new: np.ndarray, max_dist: float
) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: implement matching old points to new points
    dists = np.linalg.norm(
        points_old[:, np.newaxis] - points_new[np.newaxis], axis=-1
    )
    valid_matrix = dists < max_dist

    old_to_new = np.full(len(points_old), -1, dtype=np.int64)
    min_mask = dists == dists.min(axis=1, keepdims=True)
    inds1, inds2 = np.where(valid_matrix & min_mask)
    old_to_new[inds1] = inds2

    new_to_old = np.full(len(points_new), -1, dtype=np.int64)
    min_mask = dists == dists.min(axis=0, keepdims=True)
    inds1, inds2 = np.where(valid_matrix & min_mask)
    new_to_old[inds1] = inds2

    return old_to_new, new_to_old


class VideoProcessor:
    def __init__(self, video_path: str, tracker_type: Optional[str] = None):
        self.tracker_type = tracker_type
        match tracker_type:
            case "BOOSTING":
                self.tracker_create_func = cv2.TrackerBoosting_create
            case "BOOSTING":
                self.tracker_create_func = cv2.TrackerBoosting_create
            case "MIL":
                self.tracker_create_func = cv2.TrackerMIL_create
            case "KCF":
                self.tracker_create_func = cv2.TrackerKCF_create
            case "TLD":
                self.tracker_create_func = cv2.TrackerTLD_create
            case "MEDIANFLOW":
                self.tracker_create_func = cv2.TrackerMedianFlow_create
            case "GOTURN":
                self.tracker_create_func = cv2.TrackerGOTURN_create
            case "MOSSE":
                self.tracker_create_func = cv2.TrackerMOSSE_create
            case "CSRT":
                self.tracker_create_func = cv2.TrackerCSRT_create
            case None:
                self.tracker_create_func = None
            case _:
                raise ValueError(
                    f"'{tracker_type}' is not a valid tracker_type"
                )

        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError(f"Couldn't open video at {video_path}")

        self.vid_fps = self.video.get(cv2.CAP_PROP_FPS)
        self.vid_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        self.video.release()

    def process_and_write(
        self,
        write_path: str,
        time_interval: Tuple[Optional[float], Optional[float]] = (None, None),
        redetect_seconds: float = 1.0,
    ):
        vid_writer = cv2.VideoWriter(
            write_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            self.vid_fps,
            (self.vid_width, self.vid_height),
        )

        try:
            start_frame = 0
            end_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            if time_interval[0] is not None:
                start_frame = int(time_interval[0] * self.vid_fps)
            if time_interval[1] is not None:
                end_frame = int(time_interval[1] * self.vid_fps)

            field_detector = FieldDetectorHough(
                angle_tol=0.05, sqr_size_tol=0.1, do_rotated=True
            )
            piece_detector = PieceDetectorHough()
            piece_classifier = PieceClassifier("./images/pieces")
            event_detector = EventDetector()

            self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fields_res = None
            piece_centers = []
            piece_labels = []
            trackers = []
            bboxes = []
            event = EventType.NoEvent

            for i in tqdm(range(end_frame - start_frame)):
                if not self.video.isOpened():
                    break
                ret, frame = self.video.read()
                if not ret:
                    break

                is_redetect_iter = (
                    i % int(redetect_seconds * self.vid_fps) == 0
                )
                track_success = True
                if not is_redetect_iter:
                    bboxes = self._update_trackers(frame, trackers)
                    track_success = None not in bboxes

                if is_redetect_iter or not track_success:
                    temp_fields_res = field_detector.detect_fields_centers(
                        frame
                    )
                    if fields_res is None:
                        fields_res = temp_fields_res
                    elif temp_fields_res is not None and np.isclose(
                        temp_fields_res.field_side_size,
                        fields_res.field_side_size,
                        atol=0,
                        rtol=0.5,
                    ):
                        old_to_new, _ = match_points(
                            fields_res.field_centers,
                            temp_fields_res.field_centers,
                            fields_res.field_side_size / 2,
                        )
                        if not np.any(old_to_new == -1):
                            temp_fields_res.field_centers = (
                                temp_fields_res.field_centers[old_to_new]
                            )
                        fields_res = temp_fields_res

                    # temp_piece_centers = piece_detector.piece_centers(frame, fields_res)
                    temp_piece_centers = piece_detector.piece_centers(
                        frame, None
                    )
                    if (
                        temp_piece_centers is not None
                        and fields_res is not None
                    ):
                        piece_centers = temp_piece_centers
                        piece_labels = piece_classifier.label_points(
                            frame, piece_centers
                        )

                        if fields_res is not None:
                            new_points = calc_presence_matrix(
                                fields_res.field_centers, piece_centers
                            )
                            event_detector.push_new_points(new_points)
                            event = event_detector.detect()

                        if fields_res is not None:
                            piece_shape = (
                                int(0.8 * fields_res.field_side_size),
                                int(0.8 * fields_res.field_side_size),
                            )
                            trackers, bboxes = self._create_trackers(
                                frame, piece_centers, piece_shape
                            )

                res_frame = frame.copy()
                draw_points(
                    res_frame,
                    fields_res.field_centers if fields_res is not None else [],
                    color=(0, 255, 0),
                )
                draw_labels(
                    res_frame,
                    fields_res.field_centers,
                    [str(i) for i in range(len(fields_res.field_centers))],
                    color=(0, 255, 255),
                )
                draw_points(res_frame, piece_centers, color=(0, 0, 255))
                draw_labels(res_frame, piece_centers, piece_labels)
                draw_boxes(
                    res_frame,
                    [bbox for bbox in bboxes if bbox is not None],
                    color=(255, 0, 0),
                )
                draw_event(res_frame, event)
                vid_writer.write(res_frame)
        finally:
            vid_writer.release()

    def _create_trackers(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        piece_shape: Tuple[int, int],
    ) -> Tuple[List[cv2.Tracker], List[Tuple[int, int, int, int]]]:
        if self.tracker_type is None:
            return [], []

        piece_w, piece_h = piece_shape
        trackers = []
        bboxes = []
        for x, y in points:
            bbox = (x - piece_w // 2, y - piece_h // 2, piece_w, piece_h)

            tracker = self.tracker_create_func()
            tracker.init(frame, bbox)
            trackers.append(tracker)
            bboxes.append(bbox)
        return trackers, bboxes

    def _update_trackers(
        self, frame: np.ndarray, trackers: List[cv2.Tracker]
    ) -> List[Tuple[int, int, int, int]]:
        if self.tracker_type is None:
            return []

        bboxes = []
        for tracker in trackers:
            ok, bbox = tracker.update(frame)
            if ok:
                bboxes.append(bbox)
            else:
                bboxes.append(None)
        return bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--redetect_seconds", type=float, default=1.0)
    parser.add_argument(
        "--tracker_type",
        choices=[
            "BOOSTING",
            "MIL",
            "KCF",
            "TLD",
            "MEDIANFLOW",
            "GOTURN",
            "MOSSE",
            "CSRT",
        ],
    )

    args = parser.parse_args()

    video_processor = VideoProcessor(
        video_path=args.video_path, tracker_type=args.tracker_type
    )
    video_processor.process_and_write(
        write_path=args.output_path,
        time_interval=(args.start_time, args.end_time),
        redetect_seconds=args.redetect_seconds,
    )


if __name__ == "__main__":
    # with patch(
    #     "sys.argv",
    #     [
    #         "chess_detection.video_processing",
    #         ".\\videos\\ChangingLights2_cropped.mp4",
    #         ".\\videos\\ChangingLights2_results_temp.mp4",
    #         "--start-time", "10", "--end-time", "30", "--redetect_seconds", "1"]):
    #     main()
    main()
