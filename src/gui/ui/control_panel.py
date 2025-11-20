from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QSlider


class ControlPanel(QWidget):
    """재생/스텝 이동/슬라이더/시간 라벨을 포함한 컨트롤 패널."""

    play_clicked = Signal()
    step_clicked = Signal(int)      # delta seconds (예: -3, +3)
    slider_changed = Signal(int)    # time index

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.play_button = QPushButton("▶ Play")
        self.play_button.setEnabled(False)
        self.play_button.setMinimumHeight(40)
        self.play_button.setMinimumWidth(100)
        self.play_button.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.play_button.clicked.connect(self.play_clicked)
        layout.addWidget(self.play_button)

        self.step_back_button = QPushButton("⏪ -3s")
        self.step_back_button.setEnabled(False)
        self.step_back_button.setMinimumHeight(40)
        self.step_back_button.setMinimumWidth(80)
        self.step_back_button.setStyleSheet("font-size: 11pt;")
        self.step_back_button.clicked.connect(lambda: self.step_clicked.emit(-3))
        layout.addWidget(self.step_back_button)

        self.step_forward_button = QPushButton("+3s ⏩")
        self.step_forward_button.setEnabled(False)
        self.step_forward_button.setMinimumHeight(40)
        self.step_forward_button.setMinimumWidth(80)
        self.step_forward_button.setStyleSheet("font-size: 11pt;")
        self.step_forward_button.clicked.connect(lambda: self.step_clicked.emit(3))
        layout.addWidget(self.step_forward_button)

        lbl = QLabel("Time Step:")
        lbl.setStyleSheet("font-size: 13pt; font-weight: bold;")
        layout.addWidget(lbl)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setEnabled(False)
        self.time_slider.setMinimumHeight(40)
        self.time_slider.valueChanged.connect(self.slider_changed)
        layout.addWidget(self.time_slider, stretch=1)

        self.time_label = QLabel("0.00 s / 0.00 s")
        self.time_label.setStyleSheet("font-size: 13pt; font-weight: bold; min-width: 140px;")
        layout.addWidget(self.time_label)

    # ------------------------------------------------------------------
    # 외부에서 사용하는 메서드
    # ------------------------------------------------------------------
    def setup_time_controls(self, num_steps: int):
        """Time 슬라이더/버튼을 run 길이에 맞게 설정."""
        has_data = num_steps > 0

        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(max(0, num_steps - 1))
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)

        self.time_slider.setEnabled(has_data)
        self.play_button.setEnabled(has_data)
        self.step_back_button.setEnabled(has_data)
        self.step_forward_button.setEnabled(has_data)

    def set_time_label(self, current_time: float, total_time: float):
        self.time_label.setText(f"{current_time:.2f} s / {total_time:.2f} s")

    def set_playing(self, is_playing: bool):
        self.play_button.setText("⏸ Pause" if is_playing else "▶ Play")

    @property
    def slider(self) -> QSlider:
        return self.time_slider
