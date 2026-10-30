import numpy as np
from sensor_msgs.msg import Image


def image_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding not in ('bgr8', 'rgb8'):
        raise ValueError(f'지원하지 않는 인코딩: {msg.encoding}')
    if (msg.height <= 0 or msg.width <= 0 or msg.step < msg.width * 3
            or len(msg.data) != msg.height * msg.step):
        raise ValueError('invalid_image_layout')
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    arr = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return arr[:, :, ::-1].copy() if msg.encoding == 'rgb8' else arr


def bgr_to_image(frame: np.ndarray, stamp, frame_id: str) -> Image:
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0):
        raise ValueError('expected_nonempty_bgr8_frame')
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = 'bgr8'
    msg.step = msg.width * 3
    msg.data = frame.tobytes()
    return msg
