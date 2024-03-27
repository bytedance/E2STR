import math
from typing import Dict, Tuple

import cv2
import mmcv
import numpy as np
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness

from mmocr.registry import TRANSFORMS
from mmocr.utils import (bbox2poly, crop_polygon, is_poly_inside_rect,
                         poly2bbox, poly2shapely, poly_make_valid,
                         remove_pipeline_elements, rescale_polygon,
                         shapely2poly)
from .wrappers import ImgAugWrapper


@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Randomly rotate the image, boxes, and polygons. For recognition task,
    only the image will be rotated. If set ``use_canvas`` as True, the shape of
    rotated image might be modified based on the rotated angle size, otherwise,
    the image will keep the shape before rotation.

    Required Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_polygons (optional)

    Modified Keys:

    - img
    - img_shape (optional)
    - gt_bboxes (optional)
    - gt_polygons (optional)

    Added Keys:

    - rotated_angle

    Args:
        max_angle (int): The maximum rotation angle (can be bigger than 180 or
            a negative). Defaults to 10.
        pad_with_fixed_color (bool): The flag for whether to pad rotated
            image with fixed value. Defaults to False.
        pad_value (tuple[int, int, int]): The color value for padding rotated
            image. Defaults to (0, 0, 0).
        use_canvas (bool): Whether to create a canvas for rotated image.
            Defaults to False. If set true, the image shape may be modified.
    """

    def __init__(
        self,
        max_angle: int = 10,
        pad_with_fixed_color: bool = False,
        pad_value: Tuple[int, int, int] = (0, 0, 0),
        use_canvas: bool = False,
    ) -> None:
        if not isinstance(max_angle, int):
            raise TypeError('`max_angle` should be an integer'
                            f', but got {type(max_angle)} instead')
        if not isinstance(pad_with_fixed_color, bool):
            raise TypeError('`pad_with_fixed_color` should be a bool, '
                            f'but got {type(pad_with_fixed_color)} instead')
        if not isinstance(pad_value, (list, tuple)):
            raise TypeError('`pad_value` should be a list or tuple, '
                            f'but got {type(pad_value)} instead')
        if len(pad_value) != 3:
            raise ValueError('`pad_value` should contain three integers')
        if not isinstance(pad_value[0], int) or not isinstance(
                pad_value[1], int) or not isinstance(pad_value[2], int):
            raise ValueError('`pad_value` should contain three integers')

        self.max_angle = max_angle
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value
        self.use_canvas = use_canvas

    @cache_randomness
    def _sample_angle(self, max_angle: int) -> float:
        """Sampling a random angle for rotation.

        Args:
            max_angle (int): Maximum rotation angle

        Returns:
            float: The random angle used for rotation
        """
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    @staticmethod
    def _cal_canvas_size(ori_size: Tuple[int, int],
                         degree: int) -> Tuple[int, int]:
        """Calculate the canvas size.

        Args:
            ori_size (Tuple[int, int]): The original image size (height, width)
            degree (int): The rotation angle

        Returns:
            Tuple[int, int]: The size of the canvas
        """
        assert isinstance(ori_size, tuple)
        angle = degree * math.pi / 180.0
        h, w = ori_size[:2]

        cos = math.cos(angle)
        sin = math.sin(angle)
        canvas_h = int(w * math.fabs(sin) + h * math.fabs(cos))
        canvas_w = int(w * math.fabs(cos) + h * math.fabs(sin))

        canvas_size = (canvas_h, canvas_w)
        return canvas_size

    @staticmethod
    def _rotate_points(center: Tuple[float, float],
                       points: np.array,
                       theta: float,
                       center_shift: Tuple[int, int] = (0, 0)) -> np.array:
        """Rotating a set of points according to the given theta.

        Args:
            center (Tuple[float, float]): The coordinate of the canvas center
            points (np.array): A set of points needed to be rotated
            theta (float): Rotation angle
            center_shift (Tuple[int, int]): The shifting offset of the center
                coordinate

        Returns:
            np.array: The rotated coordinates of the input points
        """
        (center_x, center_y) = center
        center_y = -center_y
        x, y = points[::2], points[1::2]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - center_x)
        y = (y - center_y)

        _x = center_x + x * cos - y * sin + center_shift[0]
        _y = -(center_y + x * sin + y * cos) + center_shift[1]

        points[::2], points[1::2] = _x, _y
        return points

    def _rotate_img(self, results: Dict) -> Tuple[int, int]:
        """Rotating the input image based on the given angle.

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            Tuple[int, int]: The shifting offset of the center point.
        """
        if results.get('img', None) is not None:
            h = results['img'].shape[0]
            w = results['img'].shape[1]
            rotation_matrix = cv2.getRotationMatrix2D(
                (w / 2, h / 2), results['rotated_angle'], 1)

            canvas_size = self._cal_canvas_size((h, w),
                                                results['rotated_angle'])
            center_shift = (int(
                (canvas_size[1] - w) / 2), int((canvas_size[0] - h) / 2))
            rotation_matrix[0, 2] += int((canvas_size[1] - w) / 2)
            rotation_matrix[1, 2] += int((canvas_size[0] - h) / 2)
            if self.pad_with_fixed_color:
                rotated_img = cv2.warpAffine(
                    results['img'],
                    rotation_matrix, (canvas_size[1], canvas_size[0]),
                    flags=cv2.INTER_NEAREST,
                    borderValue=self.pad_value)
            else:
                mask = np.zeros_like(results['img'])
                (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                                  np.random.randint(0, w * 7 // 8))
                img_cut = results['img'][h_ind:(h_ind + h // 9),
                                         w_ind:(w_ind + w // 9)]
                img_cut = mmcv.imresize(img_cut,
                                        (canvas_size[1], canvas_size[0]))
                mask = cv2.warpAffine(
                    mask,
                    rotation_matrix, (canvas_size[1], canvas_size[0]),
                    borderValue=[1, 1, 1])
                rotated_img = cv2.warpAffine(
                    results['img'],
                    rotation_matrix, (canvas_size[1], canvas_size[0]),
                    borderValue=[0, 0, 0])
                rotated_img = rotated_img + img_cut * mask

            results['img'] = rotated_img
        else:
            raise ValueError('`img` is not found in results')

        return center_shift

    def _rotate_bboxes(self, results: Dict, center_shift: Tuple[int,
                                                                int]) -> None:
        """Rotating the bounding boxes based on the given angle.

        Args:
            results (dict): Result dict containing the data to transform.
            center_shift (Tuple[int, int]): The shifting offset of the
                center point
        """
        if results.get('gt_bboxes', None) is not None:
            height, width = results['img_shape']
            box_list = []
            for box in results['gt_bboxes']:
                rotated_box = self._rotate_points((width / 2, height / 2),
                                                  bbox2poly(box),
                                                  results['rotated_angle'],
                                                  center_shift)
                rotated_box = poly2bbox(rotated_box)
                box_list.append(rotated_box)

            results['gt_bboxes'] = np.array(
                box_list, dtype=np.float32).reshape(-1, 4)

    def _rotate_polygons(self, results: Dict,
                         center_shift: Tuple[int, int]) -> None:
        """Rotating the polygons based on the given angle.

        Args:
            results (dict): Result dict containing the data to transform.
            center_shift (Tuple[int, int]): The shifting offset of the
                center point
        """
        if results.get('gt_polygons', None) is not None:
            height, width = results['img_shape']
            polygon_list = []
            for poly in results['gt_polygons']:
                rotated_poly = self._rotate_points(
                    (width / 2, height / 2), poly, results['rotated_angle'],
                    center_shift)
                polygon_list.append(rotated_poly)
            results['gt_polygons'] = polygon_list

    def transform(self, results: Dict) -> Dict:
        """Applying random rotate on results.

        Args:
            results (Dict): Result dict containing the data to transform.
            center_shift (Tuple[int, int]): The shifting offset of the
                center point

        Returns:
            dict: The transformed data
        """
        # TODO rotate char_quads & char_rects for SegOCR
        if self.use_canvas:
            results['rotated_angle'] = self._sample_angle(self.max_angle)
            # rotate image
            center_shift = self._rotate_img(results)
            # rotate gt_bboxes
            self._rotate_bboxes(results, center_shift)
            # rotate gt_polygons
            self._rotate_polygons(results, center_shift)

            results['img_shape'] = (results['img'].shape[0],
                                    results['img'].shape[1])
        else:
            args = [
                dict(
                    cls='Affine',
                    rotate=[-self.max_angle, self.max_angle],
                    backend='cv2',
                    order=0)  # order=0 -> cv2.INTER_NEAREST
            ]
            imgaug_transform = ImgAugWrapper(args)
            results = imgaug_transform(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_angle = {self.max_angle}'
        repr_str += f', pad_with_fixed_color = {self.pad_with_fixed_color}'
        repr_str += f', pad_value = {self.pad_value}'
        repr_str += f', use_canvas = {self.use_canvas})'
        return repr_str