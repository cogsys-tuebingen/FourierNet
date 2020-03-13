import cv2

from ..registry import PIPELINES
import numpy as np
import torch
from mmdet.utils import print_log


def get_polar_coordinates(c_x, c_y, pos_mask_contour, n=72):
    if len(pos_mask_contour.shape) == 2:
        ct = pos_mask_contour
    else:
        ct = pos_mask_contour[:, 0, :]
    x = ct[:, 0] - c_x
    y = ct[:, 1] - c_y
    angle = np.arctan2(x, y) * 180 / np.pi
    angle[angle < 0] += 360
    angle = angle.astype('int')
    dist = np.sqrt(x ** 2 + y ** 2)
    idx = np.argsort(angle)
    angle = angle[idx]
    dist = dist[idx]

    interval = 360 // n
    new_coordinate = {}
    for i in range(0, 360, interval):
        if i in angle:
            d = dist[angle == i].max()
            new_coordinate[i] = d
        elif i + 1 in angle:
            d = dist[angle == i + 1].max()
            new_coordinate[i] = d
        elif i - 1 in angle:
            d = dist[angle == i - 1].max()
            new_coordinate[i] = d
        elif i + 2 in angle:
            d = dist[angle == i + 2].max()
            new_coordinate[i] = d
        elif i - 2 in angle:
            d = dist[angle == i - 2].max()
            new_coordinate[i] = d
        elif i + 3 in angle:
            d = dist[angle == i + 3].max()
            new_coordinate[i] = d
        elif i - 3 in angle:
            d = dist[angle == i - 3].max()
            new_coordinate[i] = d

    distances = np.zeros(n)

    for a in range(0, 360, interval):
        if not a in new_coordinate.keys():
            new_coordinate[a] = 1e-6
            distances[a // interval] = 1e-6
        else:
            distances[a // interval] = new_coordinate[a]

    return distances, new_coordinate


def polar_centerness_target(pos_mask_targets, max_centerness=None):
    centerness_targets = np.sqrt(pos_mask_targets.min() / pos_mask_targets.max())
    if max_centerness:
        centerness_targets /= max_centerness
    return np.clip(centerness_targets, None, 1.0)


def get_centerpoint(lis):
    area = 0.0
    x, y = 0.0, 0.0
    a = len(lis)
    for i in range(a):
        lat = lis[i][0]
        lng = lis[i][1]
        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]
        else:
            lat1 = lis[i - 1][0]
            lng1 = lis[i - 1][1]
        fg = (lat * lng1 - lng * lat1) / 2.0
        area += fg
        x += fg * (lat + lat1) / 3.0
        y += fg * (lng + lng1) / 3.0
    x = x / area
    y = y / area

    return [int(x), int(y)]


@PIPELINES.register_module
class ConvertToContour(object):
    """Converts the mask from a binary grid representation into boundary contour points,
        and also returns the contour center and the centerness of the center if required.

        Args:
            use_max_only (bool): if true, only the largest contour will be returned for instances with multiple contours
            return_centerness (bool): if true, the centerness of the contour center point is returned
            contour_points (int, optional): Number of contour point used when calculating the centerness
        """

    def __init__(self,
                 use_max_only=True,
                 return_centerness=True,
                 contour_points=None
                 ):
        self.use_max_only = use_max_only
        self.return_centerness = return_centerness
        self.contour_points = contour_points

    def __call__(self, results):

        mask_centers = []
        mask_contours = []
        max_centernesses = []

        # Go through each mask instance in image and find it's center, countour and max centerness
        for mask in results['gt_masks']:
            contour_results = self.get_contour(mask)
            if contour_results is None:
                return None
            elif self.return_centerness:
                center, contour, centerness = contour_results
            else:
                center, contour = contour_results
            contour = contour[0]
            y, x = center
            mask_centers.append([x, y])  # save mask centers of all objects
            mask_contours.append(torch.tensor(contour[:, 0, :]))  # save contour points of all objects
            if self.return_centerness:
                max_centernesses.append(centerness)

        results['gt_centers'] = mask_centers
        results['gt_masks'] = mask_contours
        if self.return_centerness:
            results['gt_max_centerness'] = max_centernesses

        return results

    def get_contour(self, mask):
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if self.use_max_only:

            contour.sort(key=lambda cx: cv2.contourArea(cx), reverse=True)
            try:
                # only save the contour with the largest area
                count = contour[0][:, 0, :]
            except IndexError:
                # This Happens when a mask is empty
                return None
        else:
            count = np.concatenate(contour)[:, 0, :]

        # Calculate the center point
        try:
            center = get_centerpoint(count)
        except (ArithmeticError, ValueError):
            x, y = count.mean(axis=0)
            center = [int(x), int(y)]

        if self.return_centerness:
            points, _ = get_polar_coordinates(center[0], center[1], count, self.contour_points)
            centernes = polar_centerness_target(points)
            return center, contour, centernes
        else:
            return center, contour
