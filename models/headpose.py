import logging
import typing

import cv2
import numpy as np
from ml_serving.utils import helpers
from ml_serving.drivers import driver

from utils import images

LOG = logging.getLogger(__name__)

idx_tensor = np.arange(0, 66)
MARGIN_COEF = .4
HEAD_POSE_DRIVER_TYPE = 'openvino'
HEAD_POSE_PATH = (
    '/opt/intel/openvino/deployment_tools/intel_models/'
    'head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml'
)
# HEAD_POSE_THRESHOLDS = '37,35,25'
HEAD_POSE_THRESHOLDS = '25,25,20'


class HeadPoseFilter(object):
    def __init__(self, **kwargs):
        # super().__init__(filter_type='head pose', **kwargs)
        head_pose_driver = kwargs.get('head_pose_driver')
        self._driver_type = kwargs.get('head_pose_driver_type', HEAD_POSE_DRIVER_TYPE)
        if head_pose_driver is None:
            model_path = kwargs.get('head_pose_model_path', HEAD_POSE_PATH)
            if model_path == 'skip':
                head_pose_driver = None
            else:
                if self._driver_type == HEAD_POSE_DRIVER_TYPE:
                    head_pose_driver = driver.load_driver('auto')().load_model(model_path)
                elif self._driver_type == 'hopenet':
                    # project https://github.com/kibernetika-ai/deep-head-pose
                    head_pose_driver = driver.load_driver('auto')().load_model(
                        model_path,
                        model_class='detector_app.filters.headpose_hopenet:HopenetProxy',
                        map_location='cpu'
                    )
                else:
                    raise ValueError('unknown head pose driver type \'{}\''.format(self._driver_type))
        self._driver = head_pose_driver
        self._head_pose_axis_threshold = float(kwargs.get('head_pose_axis_threshold', 0.0))
        self._head_pose_thresholds = get_thresholds(
            kwargs.get('head_pose_thresholds', HEAD_POSE_THRESHOLDS)
        )
        # self._no_skip = helpers.boolean_string(kwargs.get('head_pose_no_skip'))

    def set_thresholds(self, head_pose_thresholds: str):
        self._head_pose_thresholds = get_thresholds(head_pose_thresholds)

    # def _filter(self,
    #             frame: np.ndarray,
    #             to_filter: typing.List[obj.Object],
    #             ) -> typing.List[obj.Object]:
    #     if self._driver is None:
    #         return to_filter
    #     filtered = []
    #     poses = self.head_poses(frame, [f.get_bbox() for f in to_filter])
    #     for i, ind in enumerate(poses):
    #         f = to_filter[i]
    #         set_head_pose(f, ind)
    #         if not self._no_skip:
    #             if wrong_pose(ind, self._head_pose_thresholds, self._head_pose_axis_threshold):
    #                 f.set_skipped(True)
    #         filtered.append(f)
    #     return filtered

    def _im_size(self):
        if self._driver_type == HEAD_POSE_DRIVER_TYPE:
            return 60
        if self._driver_type == 'hopenet':
            return 224
        raise ValueError('unknown driver type \'{}\''.format(self._driver_type))

    def head_poses(self, frame, boxes):
        if self._driver is None:
            return []
        if boxes is None or len(boxes) == 0:
            return []

        imgs = np.stack(images.get_images(
            frame, np.array(boxes), self._im_size(), 0, normalization=None, face_crop_margin_coef=MARGIN_COEF))

        return self.head_poses_for_images(imgs, False)

    def head_poses_for_images(self, imgs, resize: bool):

        if self._driver_type == HEAD_POSE_DRIVER_TYPE:
            # Convert to BGR.
            imgs = imgs[:, :, :, ::-1]

            if resize:
                im_size = self._im_size()
                imgs = [cv2.resize(img, (im_size, im_size), interpolation=cv2.INTER_AREA) for img in imgs]
            outputs = self._driver.predict({'data': np.array(imgs).transpose([0, 3, 1, 2])})

            yaw = - outputs["angle_y_fc"].reshape([-1])
            pitch = - outputs["angle_p_fc"].reshape([-1])
            roll = outputs["angle_r_fc"].reshape([-1])

            # Return shape [N, 3] as a result
            return np.array([yaw, pitch, roll]).transpose()

        if self._driver_type == 'hopenet':
            ret = []
            if resize:
                im_size = self._im_size()
                imgs = [cv2.resize(img, (im_size, im_size), interpolation=cv2.INTER_AREA) for img in imgs]
            imgs = [images.hopenet(img) for img in imgs]

            outputs = self._driver.predict({'0': np.stack(imgs)})
            for i, out in enumerate(outputs['0']):
                yaw = np.sum(softmax(outputs['0'][i]) * idx_tensor) * 3 - 99
                pitch = np.sum(softmax(outputs['1'][i]) * idx_tensor) * 3 - 99
                roll = np.sum(softmax(outputs['2'][i]) * idx_tensor) * 3 - 99
                ret.append([yaw, pitch, roll])

            return np.array(ret)

        raise ValueError('unknown driver type \'{}\''.format(self._driver_type))


def wrong_pose(pose: [float], head_pose_thresholds: [float], head_pose_axis_threshold: float):
    if len(pose) != 3:
        raise ValueError('expected head pose as 3 floats')
    if len(head_pose_thresholds) != 3:
        raise ValueError('expected head pose thresholds as 3 floats')
    # [yaw, pitch, roll]
    [y, p, r] = pose
    if head_pose_axis_threshold is not None:
        _, _, _, z_len = _head_pose_to_axis(pose)
        return z_len > head_pose_axis_threshold
    return (np.abs(y) > head_pose_thresholds[0]
            or np.abs(p) > head_pose_thresholds[1]
            or np.abs(r) > head_pose_thresholds[2])


# def set_head_pose(to: obj.Object, hp_ind: [float]):
#     to.add_filter_data('head_pose', hp_ind,
#                        "head pose: {:.2f} {:.2f} {:.2f}".format(hp_ind[0], hp_ind[1], hp_ind[2]))
#     axis = _head_pose_to_axis(hp_ind)
#     to.add_filter_data('head_pose_axis', axis,
#                        "head pose axis len: {:.2f}".format(axis[3]))
#     to.add_debug('filter_head_pose', '[{:.3f}, {:.3f}, {:.3f}]'.format(hp_ind[0], hp_ind[1], hp_ind[2]))


def _head_pose_to_axis(hp_ind: [float]):
    (yaw, pitch, roll) = hp_ind

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # X-Axis pointing to right
    x1 = np.cos(yaw) * np.cos(roll)
    y1 = np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)

    # Y-Axis pointing down
    x2 = -np.cos(yaw) * np.sin(roll)
    y2 = np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)

    # Z-Axis out of the screen
    x3 = np.sin(yaw)
    y3 = -np.cos(yaw) * np.sin(pitch)
    z_len = np.sqrt(x3 ** 2 + y3 ** 2)

    return (x1, y1), (x2, y2), (x3, y3), z_len


def get_thresholds(thresholds: str) -> [int]:
    if isinstance(thresholds, list):
        return thresholds

    head_pose_thresholds_spl = thresholds.split(",")
    if len(head_pose_thresholds_spl) != 3:
        raise ValueError('head_pose_thresholds must be three comma separated numbers')
    try:
        ret = [float(f.strip()) for f in head_pose_thresholds_spl]
        return ret
    except Exception:
        raise ValueError('head_pose_thresholds must be three comma separated float numbers')


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))
