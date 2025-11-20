from __future__ import annotations

from typing import Sequence, Mapping
import numpy as np
import pyqtgraph as pg


class SensorPlotter:
    """
    전체 센서 플롯 + 그룹별(IMU/Wheel/Steering) 플롯을 관리하는 헬퍼.
    """

    def __init__(
        self,
        main_plot_widget: pg.GraphicsLayoutWidget,
        imu_plot: pg.PlotWidget,
        wheel_plot: pg.PlotWidget,
        steering_plot: pg.PlotWidget,
        sensor_groups,
        sensor_colors: Mapping[str, str],
        y_padding_ratio: float = 0.1,
    ):
        self._main_plot_widget = main_plot_widget
        self._imu_plot = imu_plot
        self._wheel_plot = wheel_plot
        self._steering_plot = steering_plot
        self._sensor_groups = sensor_groups
        self._sensor_colors = dict(sensor_colors)
        self._y_padding_ratio = y_padding_ratio

        self._vlines: list[pg.InfiniteLine] = []
        self._main_plot = None

    def rebuild(
        self,
        sensors: Mapping[str, np.ndarray],
        time_data: np.ndarray,
        active_sensors: Sequence[str],
    ):
        """
        전체를 다시 그림.
        sensors: {'Time': ..., 'IMU_VerAccelVal': ..., ...}
        time_data: sensors['Time'].flatten()
        active_sensors: 체크된 센서 이름 리스트
        """
        if time_data is None or len(time_data) == 0:
            return

        # 기존 플롯 지우기
        if self._main_plot_widget is not None:
            self._main_plot_widget.clear()
        if self._imu_plot is not None:
            self._imu_plot.clear()
        if self._wheel_plot is not None:
            self._wheel_plot.clear()
        if self._steering_plot is not None:
            self._steering_plot.clear()

        self._vlines = []
        self._main_plot = None

        if not active_sensors:
            return

        # ==========================
        # 0. 전체 센서 플롯
        # ==========================
        main_plot = self._main_plot_widget.addPlot()
        self._main_plot = main_plot

        all_values = []

        for sensor_name in active_sensors:
            if sensor_name not in sensors:
                continue
            sensor_values = sensors[sensor_name].flatten()
            all_values.extend(sensor_values.tolist())

            color = self._sensor_colors.get(sensor_name, "#FFFFFF")
            pen = pg.mkPen(color=color, width=2)
            main_plot.plot(time_data, sensor_values, pen=pen, name=sensor_name)

        main_plot.setXRange(time_data[0], time_data[-1], padding=0)
        main_plot.enableAutoRange(axis="x", enable=False)
        main_plot.enableAutoRange(axis="y", enable=False)

        if all_values:
            y_min, y_max = float(np.min(all_values)), float(np.max(all_values))
            y_range = y_max - y_min
            padding = y_range * self._y_padding_ratio if y_range > 0 else 1.0
            main_plot.setYRange(y_min - padding, y_max + padding, padding=0)

        main_plot.setLabel("bottom", "Time (s)")
        main_plot.setLabel("left", "Value (All Sensors)")

        vline_main = pg.InfiniteLine(pos=0, angle=90, pen="w")
        main_plot.addItem(vline_main)
        self._vlines.append(vline_main)

        # ==========================
        # 1. 그룹별 플롯
        # ==========================
        group_plot_map = {
            "IMU": self._imu_plot,
            "Wheel": self._wheel_plot,
            "Steering": self._steering_plot,
        }

        for group_name, group_sensors in self._sensor_groups:
            plot_widget = group_plot_map.get(group_name)
            if plot_widget is None:
                continue

            group_values = []

            # 그룹 중 활성 센서만 그림
            for sensor_name in group_sensors:
                if sensor_name not in active_sensors:
                    continue
                if sensor_name not in sensors:
                    continue

                sensor_values = sensors[sensor_name].flatten()
                group_values.extend(sensor_values.tolist())

                color = self._sensor_colors.get(sensor_name, "#FFFFFF")
                pen = pg.mkPen(color=color, width=2)
                plot_widget.plot(time_data, sensor_values, pen=pen, name=sensor_name)

            plot_widget.setXRange(time_data[0], time_data[-1], padding=0)
            plot_widget.enableAutoRange(axis="x", enable=False)
            plot_widget.enableAutoRange(axis="y", enable=False)

            if group_values:
                y_min, y_max = float(np.min(group_values)), float(np.max(group_values))
                y_range = y_max - y_min
                padding = y_range * self._y_padding_ratio if y_range > 0 else 1.0
                plot_widget.setYRange(y_min - padding, y_max + padding, padding=0)
            else:
                plot_widget.clear()

            vline_group = pg.InfiniteLine(pos=0, angle=90, pen="w")
            plot_widget.addItem(vline_group)
            self._vlines.append(vline_group)

    def update_vlines(self, time_data: np.ndarray, step: int):
        """현재 step에 맞춰 vline 위치만 업데이트."""
        if not self._vlines:
            return
        if time_data is None or len(time_data) == 0:
            return
        if step < 0 or step >= len(time_data):
            return

        current_time = float(time_data[step])
        for vline in self._vlines:
            if vline is not None:
                vline.setPos(current_time)
