import numpy as np
import cv2

from typing import *

from .util import Pipeline, f, do_gray
from .field_detect import FieldDetectionResults

def change_contrast_and_brightness(img: np.ndarray, contrast: float = 1.0, brightness: int = 0):
    return np.clip(contrast*img + brightness, 0, 255).astype(np.uint8)

# TODO: optimization of parameters + remove shadows
class PieceDetector:
    def __init__(self, img: np.ndarray, chessboard_keypoints: np.ndarray):
        pipeline = Pipeline(
            f(cv2.GaussianBlur, (3,3), 0),
            do_gray,
            f(cv2.morphologyEx, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=1),
            f(cv2.morphologyEx, cv2.MORPH_OPEN, np.ones((3,3)), iterations=1),
            f(cv2.morphologyEx, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=1),
        )
        self.board_kp = chessboard_keypoints
        self.img = pipeline(img)
        self.res_img = img.copy()

        self.detect_white_pieces()
        self.detect_black_pieces()

    def detect_white_pieces(self):
        ret = change_contrast_and_brightness(self.img, contrast=1.2, brightness=10)
        _, ret = cv2.threshold(self.img, 149, 255, cv2.THRESH_BINARY)
        
        for kp in self.board_kp:
            x = kp[0]
            y = kp[1]
            if ret[x, y] == 255:
                self.res_img = cv2.circle(self.res_img, (x, y), 35, (255, 0, 0), 5)

    def detect_black_pieces(self):
        ret = change_contrast_and_brightness(self.img, contrast=1.2, brightness=27)
        _, ret = cv2.threshold(ret, 60, 255, cv2.THRESH_BINARY_INV)

        for kp in self.board_kp:
            x = kp[0]
            y = kp[1]
            if ret[x, y] == 255:
                self.res_img = cv2.circle(self.res_img, (x, y), 35, (0, 255, 0), 5)


class PieceDetectorHough:
    def __init__(self):
        pass

    def piece_centers(self, img: np.ndarray, fields_results: Optional[FieldDetectionResults] = None):
        circles = self._houghcircles(img, fields_results=fields_results)
        if circles is None:
            return None
        piece_centers = self._joined_circles_centroids(img, circles)

        if fields_results is not None:
            inside_contour = np.zeros(piece_centers.shape[0], dtype=np.bool_)
            for i, (x, y) in enumerate(piece_centers.astype(np.int16)):
                if cv2.pointPolygonTest(fields_results.board_rect[:, np.newaxis], (x, y), False) > 0:
                    inside_contour[i] = True
            piece_centers = piece_centers[inside_contour]
        return piece_centers

    def _houghcircles(self, img: np.ndarray, fields_results: Optional[FieldDetectionResults] = None) -> np.ndarray:
        minRadius, maxRadius = 15, 75
        if fields_results is not None:
            avg_sqsize = int(fields_results.field_side_size)
            minRadius, maxRadius = avg_sqsize//10, avg_sqsize//2
        circles = cv2.HoughCircles(
            (f(cv2.GaussianBlur, (3,3), 0) + do_gray)(img),
            cv2.HOUGH_GRADIENT, 1, 20,
            param1=70, param2=35, minRadius=minRadius, maxRadius=maxRadius
        )
        if circles is None:
            return None
        return circles[0].astype(np.int64)
    
    def _joined_circles_centroids(self, img: np.ndarray, circles: np.ndarray) -> np.ndarray:
        circles_img = np.zeros(img.shape[:2], dtype=np.uint8)
        for circle in circles:
            cv2.circle(circles_img, (circle[0],circle[1]), circle[2], 255, -1)
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(circles_img)
        centroids = centroids[1:]
        centroids = centroids.astype(np.int32)
        return centroids
