import numpy as np
import cv2
from .util import *


### TODO: Idea - if the regular method is too skewed from the whole board center, then calculate squares from the whole board
from dataclasses import dataclass

@dataclass
class FieldDetectionResults:
    field_centers: np.ndarray
    board_rect: np.ndarray
    field_side_size: float

@dataclass
class FieldDetectionResults_Contours(FieldDetectionResults):
    field_contours: np.ndarray
    board_contour: np.ndarray


### FieldDetectorContour

class FieldDetectorContour:
    def __init__(self, area_diff_thresh: float = 0.2, w_h_diff_thresh: float = 0.2):
        self.area_diff_thresh = area_diff_thresh
        self.w_h_diff_thresh = w_h_diff_thresh

        self.contour_pipe = Pipeline(
            do_gray,
            f(cv2.GaussianBlur, (3, 3), 0),
            do_otsu,
            f(cv2.morphologyEx, cv2.MORPH_OPEN, np.ones((3,3)), iterations=1),
            do_invert,
            f(cv2.findContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        )

    def _calc_fields_centers_from_board_rect(self, x, y, w, h, border_prop=0.1):
        coeff = 1 - border_prop
        field_w, field_h = coeff*w/8, coeff*h/8
        points = []
        for i in range(8):
            for j in range(8):
                points.append((x+border_prop/2*w + field_w/2 + j*field_w, y+border_prop/2*h + field_h/2 + i*field_h))
        points = np.array(points, dtype=int)
        return points

    def _get_sqr_contours(self, img: np.ndarray, do_rotated: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        contours, _ = self.contour_pipe(img)
        sqr_contours = []
        areas = []
        for contour in contours:
            epsilon = 0.1*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)
            if do_rotated:
                w, h = cv2.minAreaRect(contour)[1]
            else:
                _, _, w, h = cv2.boundingRect(contour)
            if area > 100 and len(approx) == 4 and (w*h/area - 1) < self.area_diff_thresh and np.abs(w/h - 1) < self.w_h_diff_thresh: 
                sqr_contours.append(contour)
                areas.append(area)
        sqr_contours = np.array(sqr_contours, dtype=object)
        areas = np.array(areas)
        return sqr_contours, areas

    def detect_fields_centers(self, img: np.ndarray) -> Optional[FieldDetectionResults_Contours]:
        # Select only sqare contours
        sqr_contours, areas = self._get_sqr_contours(img)

        if sqr_contours.size == 0:
            return None

        area_inds_sorted = areas.argsort()
        board_contour = sqr_contours[area_inds_sorted[-1]]
        field_contours = sqr_contours[area_inds_sorted[:-1]] # Get all the squares except the large one (which is the whole board)
        
        if field_contours.size == 0:
            return None

        xs, ys, widths, heights = zip(*[cv2.boundingRect(contour) for contour in field_contours])
        avg_field_w, avg_field_h = np.mean(widths), np.mean(heights)

        left_x, top_y = np.min(xs), np.min(ys) # Top-left coordinates of the top-left field
        right_x = np.max(xs) + widths[np.argmax(xs)]
        bottom_y = np.max(ys) + heights[np.argmax(ys)]
        
        if (np.abs(right_x - left_x - 8*avg_field_w) <= 0.9*avg_field_w) and (np.abs(bottom_y - top_y - 8*avg_field_h) <= 0.9*avg_field_h):
            # The have detected fields next to each board edge, so we can use this
            print("Using fields for centers")
            avg_field_w, avg_field_h = (np.max(xs) - np.min(xs) + avg_field_w) / 8, (np.max(ys) - np.min(ys) + avg_field_h) / 8
            avg_field_w = avg_field_h = np.max([avg_field_w, avg_field_h])
            
            points = []
            for i in range(8):
                for j in range(8):
                    points.append((left_x + avg_field_w/2 + j*avg_field_w, top_y + avg_field_h/2 + i*avg_field_h))
            points = np.array(points, dtype=int)
        else:
            # We have to use the outer board contour to determine fields
            x, y, w, h = cv2.boundingRect(board_contour)
            if np.abs(w/avg_field_w - 8) > 1: # The board_contour is actually a field_contour
                return None

            print("Using board for centers")
            border_prop = 0.1
            coeff = 1 - border_prop
            field_w, field_h = coeff*w/8, coeff*h/8
            points = []
            for i in range(8):
                for j in range(8):
                    points.append((x+border_prop/2*w + field_w/2 + j*field_w, y+border_prop/2*h + field_h/2 + i*field_h))
            points = np.array(points, dtype=int)

        board_x, board_y, board_w, board_h = cv2.boundingRect(board_contour)
        board_rect = np.array([
            [board_x, board_y],
            [board_x + board_w, board_y],
            [board_x + board_w, board_y + board_h],
            [board_x, board_y + board_h]
        ], dtype=np.int32)

        return FieldDetectionResults_Contours(
            field_centers=points,
            board_rect=board_rect,
            field_side_size=np.mean([avg_field_w, avg_field_h]),
            field_contours=field_contours,
            board_contour=board_contour
        )
    

def calc_lines_mask(edges: np.ndarray, threshold: Optional[int] = None):
    if threshold is None:
        threshold = min(edges.shape) // 5
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    line_mask = np.zeros_like(edges, dtype=np.uint8)
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    return line_mask
do_lines_mask = f(calc_lines_mask)



### FieldDetectorHough

def calc_lines_p_mask(edges: np.ndarray):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    line_mask = np.zeros_like(edges, dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    return line_mask
do_lines_p_mask = f(calc_lines_p_mask)


def draw_lines(img: np.ndarray, lines: np.ndarray):
    line_image = np.copy(img)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 10000 * (-b))
            y1 = int(y0 + 10000 * (a))
            x2 = int(x0 - 10000 * (-b))
            y2 = int(y0 - 10000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return line_image


def sobel(src_image, kernel_size: int):
    grad_x = cv2.Sobel(src_image, cv2.CV_16S, 1, 0, ksize=kernel_size, scale=1,
                      delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(src_image, cv2.CV_16S, 0, 1, ksize=kernel_size, scale=1, 
                      delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


def calc_corners_mask(img_gray: np.ndarray, dilate_iters: int = 1):
    # img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_gray = sobel(img_gray, 3)
    dst = cv2.cornerHarris(img_gray, 3, 3, 0.04)
    dst = cv2.dilate(dst, None, iterations=dilate_iters)

    corner_mask = np.zeros_like(img_gray, dtype=np.uint8)
    corner_mask[(dst > 0.01*dst.max())] = 255
    return corner_mask


class Range:
    def __init__(self, lhs: int, rhs: int):
        self.lhs = lhs
        self.rhs = rhs
    
    def __contains__(self, item: int) -> bool:
        return self.lhs <= item and item <= self.rhs


class FieldDetectorHough:
    def __init__(self, angle_tol: float = 0.01, sqr_size_tol: float = 0.1, do_rotated: bool = False):
        self.angle_tol = angle_tol
        self.sqr_size_tol = sqr_size_tol
        self.field_detector_contour = FieldDetectorContour()
        self.do_rotated = do_rotated

    def detect_fields_centers(self, img: np.ndarray) -> FieldDetectionResults:
        _, areas = self.field_detector_contour._get_sqr_contours(img, do_rotated=self.do_rotated)
        if len(areas) < 1:
            return None
        area_inds_sorted = areas.argsort()
        sqr_area = areas[area_inds_sorted[(len(area_inds_sorted)-1) // 2]] # Take median sqr area
        sqr_side_size = np.sqrt(sqr_area)

        cand_points = self._get_candidate_points(img)
        good_points_inds = self._pick_good_points(cand_points, sqr_side_size)
        # field_centers, board_rect, field_side_size = self._pick_best_grid(cand_points[good_points_inds], cand_points, sqr_side_size)
        grid_res = self._pick_best_grid(cand_points[good_points_inds], cand_points, sqr_side_size)
        if grid_res is None:
            return None
        field_centers, board_rect = grid_res

        return FieldDetectionResults(
            field_centers=field_centers,
            board_rect=board_rect,
            field_side_size=sqr_side_size
        )

    def _calc_cos_sims(self, points: np.ndarray) -> np.ndarray:
        diff_matrix = (points[:, np.newaxis] - points[np.newaxis])
        norms = np.linalg.norm(diff_matrix, axis=-1)
        norms[norms == 0] = 1
        cos_sims = (diff_matrix[:, np.newaxis] * diff_matrix[:, :, np.newaxis]).sum(axis=-1) / norms[:, np.newaxis] / norms[:, :, np.newaxis]
        return cos_sims

    def _pick_best_grid(self, good_points: np.ndarray, all_points: np.ndarray, sqr_side_size: float) -> Tuple[np.ndarray, np.ndarray]:
        def eval_box(box: np.ndarray):
            dists = np.linalg.norm(box[:, np.newaxis] - all_points[np.newaxis], axis=-1)
            min_dists = dists.min(axis=1)
            return min_dists.sum()

        rect = cv2.minAreaRect(good_points[:, np.newaxis].astype(np.int32))
        box = cv2.boxPoints(rect)
        
        dim1_diff, dim2_diff = box[1] - box[0], box[2] - box[1]
        dim1_size = int(np.round(np.linalg.norm(dim1_diff) / sqr_side_size))
        dim2_size = int(np.round(np.linalg.norm(dim2_diff) / sqr_side_size))
        if dim1_size == 0 or dim2_size == 0 or dim1_size > 8 or dim2_size > 8:
            return None
        dim1_delta, dim2_delta = dim1_diff / dim1_size, dim2_diff / dim2_size

        dim1_remains, dim2_remains = 8 - dim1_size, 8 - dim2_size
        new_box = np.empty_like(box)
        boxes, evals = [], []
        for dim1_before in range(dim1_remains+1):
            dim1_after = dim1_remains - dim1_before
            for dim2_before in range(dim2_remains+1):
                dim2_after = dim2_remains - dim2_before
                new_box[0] = box[0] - dim1_before * dim1_delta - dim2_before * dim2_delta
                new_box[1] = box[1] + dim1_after * dim1_delta - dim2_before * dim2_delta
                new_box[2] = box[2] + dim1_after * dim1_delta + dim2_after * dim2_delta
                new_box[3] = box[3] - dim1_before * dim1_delta + dim2_after * dim2_delta
                boxes.append(new_box.copy())
                evals.append(eval_box(new_box))

        best_box = boxes[np.argmin(evals)]
        field_centers = []
        for i in range(8):
            for j in range(8):
                field_centers.append(best_box[0] + (i+0.5) * dim1_delta + (j+0.5) * dim2_delta)
        field_centers = np.array(field_centers, dtype=np.int32)

        return field_centers, best_box

    def _get_candidate_points(self, img: np.ndarray) -> np.ndarray:
        edges = (do_gray + do_adaptive + do_otsu_canny)(img)
        line_mask = calc_lines_mask(edges)
        corner_mask = calc_corners_mask(do_gray(img), dilate_iters=2)

        intersect_mask = cv2.bitwise_and(line_mask, corner_mask)

        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersect_mask)
        centroids = centroids[1:]
        centroids = centroids.astype(np.int32)
        return centroids 
    
    def _pick_good_points(self, points: np.ndarray, sqr_side_size: float) -> np.ndarray:
        inside_point_inds = []

        # cos_sims = np.abs(self._calc_cos_sims(points))
        cos_sims = self._calc_cos_sims(points)

        for i, point in enumerate(points):
            dists = np.linalg.norm(points - point, axis=1)
            good_dists_inds = np.where(np.isclose(dists, sqr_side_size, atol=0, rtol=self.sqr_size_tol))[0]
            good_dists = dists[good_dists_inds]
            if len(good_dists) < 4:
                continue

            good_dists_sorted_inds = good_dists.argsort()
            windows = np.lib.stride_tricks.sliding_window_view(good_dists[good_dists_sorted_inds], 4)
            windows_diffsums = (windows - windows.mean(axis=1)[..., np.newaxis]).sum(axis=1)
            best_window_ind = np.argmin(windows_diffsums)
            best_point_inds = good_dists_inds[good_dists_sorted_inds[best_window_ind: best_window_ind+4]]
            
            inds1 = np.repeat(best_point_inds, len(best_point_inds))
            inds2 = np.tile(best_point_inds, len(best_point_inds))
            same_dist_sims = cos_sims[i, inds1, inds2]
            # angle_check = np.all(
            #     np.isclose(same_dist_sims, 1, atol=self.angle_tol, rtol=0) | 
            #     np.isclose(same_dist_sims, 0, atol=self.angle_tol, rtol=0)
            # )
            angle_check = (
                np.isclose(same_dist_sims, 0, atol=self.angle_tol, rtol=0).sum() == 8 
                and np.isclose(same_dist_sims, 1, atol=self.angle_tol, rtol=0).sum() == 4
                and np.isclose(same_dist_sims, -1, atol=self.angle_tol, rtol=0).sum() == 4
            )

            if angle_check:
                inside_point_inds.append(i)
                # print(f"Point {i}")
                # print(f"{best_point_inds=}")
                # print(f"{windows[best_window_ind]=}")
                # print(f"{same_dist_sims=}")
        
        return inside_point_inds