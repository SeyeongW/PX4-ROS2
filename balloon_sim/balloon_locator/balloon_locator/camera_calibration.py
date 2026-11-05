"""Measured monocular ROS camera_calibration YAML; no nominal calibration values."""
from pathlib import Path

import cv2
import numpy as np
import yaml

from .geometry import calibrated_intrinsics


class CameraCalibration:
    def __init__(self, path):
        try:
            document = yaml.safe_load(Path(path).expanduser().read_text())
            self.width, self.height = document['image_width'], document['image_height']
            if (type(self.width) is not int or type(self.height) is not int
                    or min(self.width, self.height) <= 0):
                raise ValueError('invalid_calibration_dimensions')
            self.raw_k = self._matrix(document, 'camera_matrix', 3, 3)
            rectification = self._matrix(document, 'rectification_matrix', 3, 3)
            self.p = self._matrix(document, 'projection_matrix', 3, 4)
            self.k = self.p[:, :3].copy()
            model = document['distortion_model']
            columns = {'plumb_bob': (4, 5), 'rational_polynomial': (8,)}.get(model, ())
            count = document['distortion_coefficients']['cols']
            if count not in columns:
                raise ValueError('unsupported_distortion_model_or_coefficients')
            self.d = self._matrix(document, 'distortion_coefficients', 1, count).ravel()
            if not np.allclose(rectification, np.eye(3)) or not np.allclose(self.p[:, 3], 0.):
                raise ValueError('only_monocular_identity_rectification_supported')
            for matrix in (self.raw_k, self.k):
                calibrated_intrinsics(dict(k=matrix.ravel().tolist(), d=[], width=self.width,
                                           height=self.height, distortion_model='plumb_bob'))
        except (KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
            raise ValueError('invalid_ros_camera_calibration') from error
        self._maps = None

    @staticmethod
    def _matrix(document, name, rows, cols):
        value = document[name]
        if value['rows'] != rows or value['cols'] != cols:
            raise ValueError('invalid_calibration_matrix_shape')
        matrix = np.asarray(value['data'], dtype=float).reshape(rows, cols)
        if not np.isfinite(matrix).all():
            raise ValueError('nonfinite_calibration_matrix')
        return matrix

    def rectify(self, frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.shape != (self.height, self.width, 3)):
            raise ValueError('image_resolution_does_not_match_calibration')
        if self._maps is None:
            self._maps = cv2.initUndistortRectifyMap(self.raw_k, self.d, np.eye(3), self.k,
                                                  (self.width, self.height), cv2.CV_32FC1)
        return cv2.remap(frame, *self._maps, interpolation=cv2.INTER_LINEAR)

    def message(self, stamp, frame_id):
        from sensor_msgs.msg import CameraInfo
        # The emitted image is already rectified. Publish its effective pinhole
        # K and zero D so downstream nodes do not rectify it a second time.
        message = CameraInfo(width=self.width, height=self.height, distortion_model='plumb_bob',
            k=self.k.ravel().tolist(), d=[0.] * 5, r=np.eye(3).ravel().tolist(), p=self.p.ravel().tolist())
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        return message
