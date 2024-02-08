import cv2
import numpy as np

import glob
import os
from typing import Optional, Tuple

from .util import clahe_channels, do_gray
from .field_detect import FieldDetectionResults


class PieceClassifier:
    def __init__(self, pieces_dir: str, match_method: int = cv2.TM_SQDIFF_NORMED, rescale_pieces: bool = False, do_clahe: bool = False):
        self.pieces_dir = pieces_dir
        self.piece_names = glob.glob("*", root_dir=self.pieces_dir)
        self.match_method = match_method
        self.rescale_pieces = rescale_pieces
        self.do_clahe = do_clahe

        self.arg_func = np.argmax
        if self.match_method == cv2.TM_SQDIFF_NORMED:
            self.arg_func = np.argmin
    
    def label_points(self, img: np.ndarray, points: np.ndarray, field_results: Optional[FieldDetectionResults] = None):
        piece_shape = None
        if self.rescale_pieces:
            assert field_results is not None, "'field_results' is None when 'self.rescale_pieces' is True"
            piece_shape = int(0.75 * field_results.field_side_size), int(0.75 * field_results.field_side_size)

        match_matrix = self._get_match_matrix(img, points, piece_shape)

        best_inds = self.arg_func(match_matrix, axis=1)
        return np.take(self.piece_names, best_inds)

    def _get_match_matrix(self, img: np.ndarray, points: np.ndarray, piece_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        match_results = np.zeros((len(points), len(self.piece_names)))
        for j, piece_name in enumerate(self.piece_names):
            piece_dir_path = os.path.join(self.pieces_dir, piece_name)
            piece_paths = [os.path.join(piece_dir_path, filename) for filename in glob.glob("*", root_dir=piece_dir_path)]
            piece_results = np.zeros(len(points))
            for piece_path in piece_paths:
                piece_img = cv2.imread(piece_path, cv2.IMREAD_COLOR)
                if piece_shape is not None:
                    piece_img = cv2.resize(piece_img, piece_shape, cv2.INTER_AREA)

                for i, point in enumerate(points):
                    piece_results[i] += self._match_image_at_point(img, piece_img, point)
            piece_results /= len(piece_paths)
            match_results[:, j] = piece_results
        return match_results

    def _match_image_at_point(self, source_img: np.ndarray, match_img: np.ndarray, point: np.ndarray):
        match_h, match_w = match_img.shape[:2]
        point_x, point_y = point
        top_left_y = point_y - match_h//2
        top_left_x = point_x - match_w//2

        source_patch = source_img[top_left_y:top_left_y+match_h, top_left_x:top_left_x+match_w]
        if self.do_clahe:
            source_patch = clahe_channels(do_gray(source_patch), tileGridSize=(9, 9))
            match_img = clahe_channels(do_gray(match_img), tileGridSize=(9, 9))
        try:
            res = cv2.matchTemplate(source_patch, match_img, self.match_method)
        except Exception:
            return 0
        return res[0, 0]
    
    def _match_image_whole(self, source_img: np.ndarray, match_img: np.ndarray):
        res = cv2.matchTemplate(source_img, match_img, self.match_method)
        return res
    
    def get_bestinds_whole(self, img: np.ndarray):
        PIECE_SHAPE = None
        templates_results = []
        for piece_name in self.piece_names:
            piece_dir_path = os.path.join(self.pieces_dir, piece_name)
            piece_paths = [os.path.join(piece_dir_path, filename) for filename in glob.glob("*", root_dir=piece_dir_path)]
            templates_res_l = []
            for piece_path in piece_paths:
                piece_img = cv2.imread(piece_path, cv2.IMREAD_COLOR)
                if PIECE_SHAPE is None:
                    PIECE_SHAPE = piece_img.shape[:2]
                else:
                    assert piece_img.shape[:2] == PIECE_SHAPE, "All piece images are not same shape"
                res = self._match_image_whole(img, piece_img)
                templates_res_l.append(res)
            templates_res = np.mean(templates_res_l, axis=0)
            templates_results.append(templates_res)

        best_match_inds = np.argmax(templates_results, axis=0)
        best_match_inds_padded = cv2.copyMakeBorder(best_match_inds, 0, PIECE_SHAPE[0]//2, 0, PIECE_SHAPE[1]//2, cv2.BORDER_CONSTANT)
        return best_match_inds, best_match_inds_padded, PIECE_SHAPE