from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class VideoPanel(QWidget):
    """비디오를 표시하는 영역."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Video:")
        title.setStyleSheet("font-size: 11pt; font-weight: bold;")
        layout.addWidget(title)

        self.label = QLabel()
        self.label.setMinimumHeight(300)
        self.label.setMinimumWidth(400)
        self.label.setScaledContents(False)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            "background-color: black; "
            "border: 2px solid rgb(44, 49, 58); "
            "border-radius: 5px;"
        )
        layout.addWidget(self.label)

    def show_frame(self, frame_rgb: np.ndarray):
        """
        RGB(OpenCV) 프레임을 받아 QLabel에 출력.
        frame_rgb: shape (H, W, 3), dtype uint8
        """
        if frame_rgb is None:
            return

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.label.setPixmap(scaled)
