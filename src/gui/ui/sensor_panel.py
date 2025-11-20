from __future__ import annotations

from typing import Dict, List, Optional

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QCheckBox,
)


class SensorPanel(QWidget):
    """
    센서 체크박스 + 플롯(IMU/Wheel/Steering/All)을 묶은 패널.
    실제 선 플로팅은 SensorPlotter 가 담당하고,
    여기서는 센서 선택/가시성만 관리한다.
    """

    sensors_changed = Signal(list)        # 활성 센서 목록
    visibility_changed = Signal(dict)     # {"imu": bool, "wheel": bool, ...}

    def __init__(self, sensor_groups, sensor_colors_list, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._sensor_groups = sensor_groups
        self._sensor_colors_list = list(sensor_colors_list)
        self._sensor_colors: Dict[str, str] = {}
        self._sensor_checkboxes: Dict[str, QCheckBox] = {}

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Sensor Data:")
        title.setStyleSheet("font-size: 11pt; font-weight: bold;")
        main_layout.addWidget(title)

        # ------------------------------
        # 센서 체크박스 + 색상 표기
        # ------------------------------
        sensor_controls_layout = QHBoxLayout()
        color_index = 0

        for group_name, sensors in self._sensor_groups:
            group_box = QGroupBox(group_name)
            group_layout = QVBoxLayout(group_box)

            for sensor_name in sensors:
                row = QHBoxLayout()

                color = self._sensor_colors_list[color_index % len(self._sensor_colors_list)]
                self._sensor_colors[sensor_name] = color

                color_label = QLabel("●")
                color_label.setStyleSheet(f"color: {color}; font-size: 16pt; font-weight: bold;")
                color_label.setFixedWidth(20)
                row.addWidget(color_label)

                cb = QCheckBox(sensor_name.replace("_", " "))
                cb.setChecked(True)
                cb.stateChanged.connect(self._on_sensor_checkbox_changed)
                self._sensor_checkboxes[sensor_name] = cb
                row.addWidget(cb)

                row.addStretch()
                group_layout.addLayout(row)

                color_index += 1

            sensor_controls_layout.addWidget(group_box)

        sensor_controls_layout.addStretch()
        main_layout.addLayout(sensor_controls_layout)

        # ------------------------------
        # 플롯 가시성 토글
        # ------------------------------
        toggle_layout = QHBoxLayout()
        lbl = QLabel("Visible Plots:")
        lbl.setStyleSheet("font-size: 10pt; font-weight: bold;")
        toggle_layout.addWidget(lbl)

        self._plot_toggle_imu = QCheckBox("IMU")
        self._plot_toggle_imu.setChecked(False)
        self._plot_toggle_imu.stateChanged.connect(self._on_visibility_changed)
        toggle_layout.addWidget(self._plot_toggle_imu)

        self._plot_toggle_wheel = QCheckBox("Wheel")
        self._plot_toggle_wheel.setChecked(False)
        self._plot_toggle_wheel.stateChanged.connect(self._on_visibility_changed)
        toggle_layout.addWidget(self._plot_toggle_wheel)

        self._plot_toggle_steering = QCheckBox("Steering")
        self._plot_toggle_steering.setChecked(False)
        self._plot_toggle_steering.stateChanged.connect(self._on_visibility_changed)
        toggle_layout.addWidget(self._plot_toggle_steering)

        self._plot_toggle_all = QCheckBox("All")
        self._plot_toggle_all.setChecked(True)
        self._plot_toggle_all.stateChanged.connect(self._on_visibility_changed)
        toggle_layout.addWidget(self._plot_toggle_all)

        toggle_layout.addStretch()
        main_layout.addLayout(toggle_layout)

        # ------------------------------
        # 플롯 위젯들
        # ------------------------------
        self._plot_imu = pg.PlotWidget()
        self._plot_imu.setMinimumHeight(200)
        self._plot_imu.setBackground((33, 37, 43))
        self._plot_imu.setLabel("left", "IMU")
        self._plot_imu.setLabel("bottom", "Time (s)")
        main_layout.addWidget(self._plot_imu)

        self._plot_wheel = pg.PlotWidget()
        self._plot_wheel.setMinimumHeight(200)
        self._plot_wheel.setBackground((33, 37, 43))
        self._plot_wheel.setLabel("left", "Wheel")
        self._plot_wheel.setLabel("bottom", "Time (s)")
        main_layout.addWidget(self._plot_wheel)

        self._plot_steering = pg.PlotWidget()
        self._plot_steering.setMinimumHeight(200)
        self._plot_steering.setBackground((33, 37, 43))
        self._plot_steering.setLabel("left", "Steering")
        self._plot_steering.setLabel("bottom", "Time (s)")
        main_layout.addWidget(self._plot_steering)

        self._plot_all_widget = pg.GraphicsLayoutWidget()
        self._plot_all_widget.setMinimumHeight(250)
        self._plot_all_widget.setBackground((33, 37, 43))
        main_layout.addWidget(self._plot_all_widget)

        # 초기 가시성 반영
        self._on_visibility_changed()
        # 초기 센서 목록 전달
        self._emit_active_sensors()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------
    def _emit_active_sensors(self):
        active = [name for name, cb in self._sensor_checkboxes.items() if cb.isChecked()]
        self.sensors_changed.emit(active)

    def _on_sensor_checkbox_changed(self, state: int):
        self._emit_active_sensors()

    def _on_visibility_changed(self, *args):
        vis = {
            "imu": self._plot_toggle_imu.isChecked(),
            "wheel": self._plot_toggle_wheel.isChecked(),
            "steering": self._plot_toggle_steering.isChecked(),
            "all": self._plot_toggle_all.isChecked(),
        }

        self._plot_imu.setVisible(vis["imu"])
        self._plot_wheel.setVisible(vis["wheel"])
        self._plot_steering.setVisible(vis["steering"])
        self._plot_all_widget.setVisible(vis["all"])

        self.visibility_changed.emit(vis)

    # ------------------------------------------------------------------
    # 외부에서 접근하는 프로퍼티/메서드
    # ------------------------------------------------------------------
    @property
    def main_plot_widget(self):
        return self._plot_all_widget

    @property
    def imu_plot(self):
        return self._plot_imu

    @property
    def wheel_plot(self):
        return self._plot_wheel

    @property
    def steering_plot(self):
        return self._plot_steering

    def get_active_sensors(self) -> List[str]:
        return [name for name, cb in self._sensor_checkboxes.items() if cb.isChecked()]

    def get_sensor_colors(self) -> Dict[str, str]:
        return dict(self._sensor_colors)
