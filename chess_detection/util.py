import numpy as np
import cv2

from typing import *


def draw_points(img: np.ndarray, points: np.ndarray, color=(0, 0, 255)):
    for x, y in points:
        img = cv2.circle(img, (x,y), radius=5, color=color, thickness=-1)
    return img


def draw_mask(img: np.ndarray, mask: np.ndarray):
    if len(img.shape) == 3:
        color = (0, 0, 255)
    else:
        color = 255
    temp_img = img.copy()
    temp_img[mask > 0] = color
    return temp_img


def draw_boxes(img: np.ndarray, bboxes: List[Tuple[int, int, int, int]], color=(255, 0, 0)):
    thickness = 3
    for bbox in bboxes:
        x, y, w, h = bbox
        img = cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)


def random_color(dims: int = 3):
    return tuple(np.random.randint(255, size=dims).tolist())


def draw_circles(img: np.ndarray, circles: np.ndarray):
    for i in circles:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    return img


def draw_labels(img: np.ndarray, points: np.ndarray, labels: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    for point, label in zip(points, labels):
        img = cv2.putText(
            img, 
            label, 
            point, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            2, 
            cv2.LINE_AA
        )
    return img


def draw_event(img: np.ndarray, message: str):
    x = 10
    # y = img.shape[1] + 100
    y = 100
    for d_x, d_y in ((-3, 0), (0, -3), (3, 0), (0, 3)):
        img = cv2.putText(
            img,
            message,
            (x+d_x, y+d_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.5,
            (0, 0, 0), # black
            8,
            cv2.LINE_AA
        )

    img = cv2.putText(
        img,
        message,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        3.5,
        (255, 255, 255), # white
        8,
        cv2.LINE_AA
    )
    return img


def pad_to_shape_centered(img: np.ndarray, shape: Tuple[int, int]):
    img_h, img_w = img.shape[:2]
    h, w = shape[0], shape[1]

    top_pad = (h - img_h) // 2
    left_pad = (w - img_w) // 2
    bottom_pad = (h - img_h - top_pad)
    right_pad = (w - img_w - left_pad)

    result = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, None, 0)
    return result


class f:
    def __init__(self, func: Callable, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, img: np.ndarray):
        return self.func(img, *self.args, **self.kwargs)
    
    def __add__(self, o):
        if isinstance(o, f):
            return Pipeline(self, o)
        elif isinstance(o, Pipeline):
            return Pipeline(self, o.func_list)


class Pipeline:
    def __init__(self, *func_list: f):
        self.func_list = func_list
    
    def __call__(self, img: np.ndarray):
        temp_img = img.copy()
        for func in self.func_list:
            temp_img = func(temp_img)
        return temp_img
    
    def __add__(self, o):
        if not isinstance(o, (Pipeline, f)):
            raise TypeError(f"Cannot add '{type(o)}' to Pipeline")
        if isinstance(o, Pipeline):
            return Pipeline(*self.func_list, *o.func_list)
        elif isinstance(o, f):
            return Pipeline(*self.func_list, o)
        

do_gray = f(cv2.cvtColor, cv2.COLOR_BGR2GRAY)
do_invert = f(cv2.bitwise_not)

final_do_contours = f(cv2.findContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

do_otsu = f(lambda img_gray: cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])

def otsu_canny(img_gray: np.ndarray):
    otsu_thresh, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(img_gray, 0.5*otsu_thresh, otsu_thresh, apertureSize=3)
    return edges
do_otsu_canny = f(otsu_canny)


def adaptive_thresh_gauss(img: np.ndarray, blocksize=3, constant=2):
    img_ad_gauss = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blocksize,
        constant,
    )
    return img_ad_gauss
do_adaptive = f(adaptive_thresh_gauss, blocksize=151)


def clahe_channels(img: np.ndarray, clipLimit=40, tileGridSize=(5, 5)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    
    if img.ndim < 3:
        return clahe.apply(img)

    eq_channels = []
    for ch in range(img.shape[-1]):
        eq_channel = clahe.apply(img[..., ch])
        eq_channels.append(eq_channel[..., np.newaxis])
    return np.concatenate(eq_channels, axis=-1)