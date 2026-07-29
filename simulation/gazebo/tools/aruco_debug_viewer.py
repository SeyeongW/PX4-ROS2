#!/usr/bin/env python3
"""Small, deterministic viewer for the detector's compressed debug image.

The Humble rqt_image_view CLI can interpret a transport-specific topic as a
raw Image when its ROS graph discovery races simulator startup.  This viewer
subscribes to CompressedImage directly and lets Qt own the GUI event loop.
"""

from __future__ import annotations

import sys
import threading

import cv2
import numpy as np
import rclpy
from python_qt_binding.QtCore import Qt, QTimer, Signal
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtWidgets import QApplication, QLabel, QMainWindow
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


TOPIC = "/aruco/debug_image/compressed"
WINDOW_TITLE = "ArUco detection result"


class DebugImageNode(Node):
    def __init__(self, window: "DebugImageWindow") -> None:
        super().__init__("aruco_debug_viewer")
        self._window = window
        self._first_frame = True
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._subscription = self.create_subscription(
            CompressedImage, TOPIC, self._on_image, qos
        )
        self.get_logger().info(f"waiting for {TOPIC}")

    def _on_image(self, message: CompressedImage) -> None:
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().error("failed to decode detector JPEG")
            return

        self._window.frame_received.emit(bgr)
        if self._first_frame:
            self._first_frame = False
            self.get_logger().info(
                f"displaying {bgr.shape[1]}x{bgr.shape[0]} detector frames"
            )


class DebugImageWindow(QMainWindow):
    frame_received = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self._label = QLabel("Waiting for ArUco detector frames...")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background: #202020; color: white;")
        self.setCentralWidget(self._label)
        self.resize(960, 720)
        self.frame_received.connect(self.set_bgr_image)

    def set_bgr_image(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self._label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._label.setPixmap(pixmap)


def main() -> int:
    application = QApplication([sys.argv[0]])
    window = DebugImageWindow()
    rclpy.init(args=None)
    node = DebugImageNode(window)

    def spin_ros() -> None:
        try:
            rclpy.spin(node)
        except (ExternalShutdownException, KeyboardInterrupt):
            pass

    spin_thread = threading.Thread(
        target=spin_ros, name="aruco-debug-ros", daemon=True
    )
    spin_thread.start()

    shutdown_timer = QTimer()
    shutdown_timer.timeout.connect(
        lambda: None if rclpy.ok() else application.quit()
    )
    shutdown_timer.start(100)

    window.show()
    window.raise_()
    window.activateWindow()
    result = application.exec_()

    shutdown_timer.stop()
    if rclpy.ok():
        rclpy.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
