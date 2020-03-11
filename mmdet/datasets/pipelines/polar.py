import cv2

from ..registry import PIPELINES
import numpy as np


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
    # only calculate pos centerness targets, otherwise there may be nan
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
class PolarTarget(object):
    def __init__(self,
                 contour_points,
                 use_max_only=True,
                 return_max_centerness=True):
        self.contour_points = contour_points
        self.use_max_only = use_max_only
        self.return_max_centerness = return_max_centerness

    def __call__(self, results):

        mask_centers = []
        mask_contours = []
        max_centernesses = []

        # Go through each mask instance in image and find it's center, countour and max centerness
        for mask in results['gt_masks']:
            if self.return_max_centerness:
                center, contour, max_centerness = self.get_single_centerpoint(mask)
            else:
                center, contour = self.get_single_centerpoint(mask)
            contour = contour[0]
            y, x = center
            mask_centers.append([x, y])  # save mask centers of all objects
            mask_contours.append(contour)  # save contour points of all objects
            max_centernesses.append(max_centerness)

        results['gt_centers'] = mask_centers
        results['gt_poly'] = mask_contours
        results['gt_max_centerness'] = max_centernesses

        return results

    def get_single_centerpoint(self, mask):
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if self.use_max_only:
            contour.sort(key=lambda cx: cv2.contourArea(cx), reverse=True)  # only save the biggest one
            '''debug IndexError: list index out of range'''
            count = contour[0][:, 0, :]
        else:
            count = np.concatenate(contour)[:, 0, :]
        try:
            center = get_centerpoint(count)
        except:
            x, y = count.mean(axis=0)
            center = [int(x), int(y)]

        if self.return_max_centerness:
            points, _ = get_polar_coordinates(center[0], center[1], count, self.contour_points)
            max_centerness = polar_centerness_target(points)
            return center, contour, max_centerness
        else:
            return center, contour
