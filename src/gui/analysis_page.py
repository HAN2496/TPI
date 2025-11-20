# analysis_page.py

import os
import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
)


class AnalysisPage(QWidget):
    """
    메인 윈도우의 new_page 안에 들어가는 '데이터셋 분석' 전용 위젯.
    MainWindow에서는 set_run(...) 만 불러주면 되고,
    나머지 분석/표시 로직은 전부 이 안에서 처리한다.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # 현재 선택된 run 정보
        self.current_folder: str | None = None
        self.current_timestamp: str | None = None
        self.current_data: dict | None = None

        # UI 구성 요소 핸들
        self.info_label: QLabel | None = None
        self.analyze_button: QPushButton | None = None

        self.tabs: QTabWidget | None = None

        # Overview 탭
        self.overview_stats_table: QTableWidget | None = None
        self.overview_plot: pg.PlotWidget | None = None

        # Sensor Stats 탭
        self.sensor_stats_combo: QComboBox | None = None
        self.sensor_stats_text: QLabel | None = None
        self.sensor_stats_plot: pg.PlotWidget | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # 상단: 현재 대상 정보 + 분석 실행 버튼
        top_layout = QHBoxLayout()
        self.info_label = QLabel("No dataset selected.")
        self.info_label.setStyleSheet("font-size: 11pt;")
        top_layout.addWidget(self.info_label, stretch=1)

        self.analyze_button = QPushButton("Analyze Current Run")
        self.analyze_button.setMinimumHeight(32)
        self.analyze_button.setStyleSheet("font-size: 10pt; font-weight: bold;")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.analyze_button.setEnabled(False)
        top_layout.addWidget(self.analyze_button)

        main_layout.addLayout(top_layout)

        # 중앙: 탭 위젯
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        # --- Overview 탭 ---
        self._build_overview_tab()
        # --- Sensor Stats 탭 ---
        self._build_sensor_stats_tab()
        # --- Events / Correlation placeholder 탭 ---
        self._build_placeholder_tabs()

    def _build_overview_tab(self):
        """주행 전체에 대한 요약 통계 + 대표 타임라인 플롯."""
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)
        overview_layout.setContentsMargins(10, 10, 10, 10)
        overview_layout.setSpacing(10)

        # 요약 통계 테이블
        self.overview_stats_table = QTableWidget()
        self.overview_stats_table.setColumnCount(2)
        self.overview_stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.overview_stats_table.horizontalHeader().setStretchLastSection(True)
        self.overview_stats_table.verticalHeader().setVisible(False)
        self.overview_stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        overview_layout.addWidget(self.overview_stats_table)

        # 대표 타임라인 플롯
        self.overview_plot = pg.PlotWidget()
        self.overview_plot.setBackground((33, 37, 43))
        self.overview_plot.setMinimumHeight(220)
        self.overview_plot.setLabel("bottom", "Time (s)")
        self.overview_plot.setLabel("left", "Value")
        overview_layout.addWidget(self.overview_plot)

        self.tabs.addTab(overview_widget, "Overview")

    def _build_sensor_stats_tab(self):
        """센서 하나 선택해서 분포/기본 통계 보는 탭."""
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(10)

        # 왼쪽: 센서 선택 콤보박스
        left_layout = QVBoxLayout()
        label = QLabel("Select Sensor:")
        left_layout.addWidget(label)

        self.sensor_stats_combo = QComboBox()
        self.sensor_stats_combo.currentIndexChanged.connect(
            self.update_sensor_stats_view
        )
        left_layout.addWidget(self.sensor_stats_combo)

        left_layout.addStretch()
        stats_layout.addLayout(left_layout, 1)

        # 오른쪽: 통계 텍스트 + 히스토그램
        right_layout = QVBoxLayout()

        self.sensor_stats_text = QLabel("No sensor selected.")
        self.sensor_stats_text.setWordWrap(True)
        right_layout.addWidget(self.sensor_stats_text)

        self.sensor_stats_plot = pg.PlotWidget()
        self.sensor_stats_plot.setBackground((33, 37, 43))
        self.sensor_stats_plot.setMinimumHeight(220)
        self.sensor_stats_plot.setLabel("bottom", "Value")
        self.sensor_stats_plot.setLabel("left", "Count")
        right_layout.addWidget(self.sensor_stats_plot, stretch=1)

        stats_layout.addLayout(right_layout, 3)

        self.tabs.addTab(stats_widget, "Sensor Stats")

    def _build_placeholder_tabs(self):
        """Events / Correlation 탭은 일단 설명만 있는 placeholder로 생성."""
        # Events 탭
        events_widget = QWidget()
        events_layout = QVBoxLayout(events_widget)
        events_label = QLabel(
            "Events / Anomalies tab (TODO)\n\n"
            "예: 급가속/급제동/급조향 이벤트 검출 결과를 여기에 표시할 수 있습니다."
        )
        events_label.setWordWrap(True)
        events_layout.addWidget(events_label)
        self.tabs.addTab(events_widget, "Events")

        # Correlation 탭
        corr_widget = QWidget()
        corr_layout = QVBoxLayout(corr_widget)
        corr_label = QLabel(
            "Correlation tab (TODO)\n\n"
            "예: 센서 간 상관계수 행렬과 Scatter plot을 여기에 표시할 수 있습니다."
        )
        corr_label.setWordWrap(True)
        corr_layout.addWidget(corr_label)
        self.tabs.addTab(corr_widget, "Correlation")

    # ------------------------------------------------------------------
    # MainWindow 쪽에서 호출하는 API
    # ------------------------------------------------------------------
    def set_run(self, folder: str, timestamp: str, data: dict | None, label_text: str = ""):
        """
        MainWindow.load_data(...)에서 호출해서
        '현재 분석 대상 run' 을 알려주는 메서드.
        """
        self.current_folder = folder
        self.current_timestamp = timestamp
        self.current_data = data

        if data is None:
            self.info_label.setText("No dataset selected.")
            self.analyze_button.setEnabled(False)
            return

        info = data.get("info", {})
        folder_name = os.path.basename(folder) if folder else "N/A"

        text = f"Folder: {folder_name} | Timestamp: {timestamp}"
        if label_text:
            text += f" | Label: {label_text}"
        text += (
            f"\nDriver: {info.get('Driver', 'N/A')} | "
            f"Vehicle: {info.get('Vehicle', 'N/A')}"
        )

        self.info_label.setText(text)
        self.analyze_button.setEnabled(True)

    # ------------------------------------------------------------------
    # 분석 실행 / 탭 채우기
    # ------------------------------------------------------------------
    def run_analysis(self):
        """현재 run에 대해 간단한 분석을 수행하고 탭을 채운다."""
        if not self.current_data:
            self.info_label.setText(
                "No dataset loaded. Please select a run on the Home page."
            )
            return

        sensors = self.current_data.get("sensors", {})
        if "Time" not in sensors:
            self.info_label.setText(
                "Current data has no 'Time' sensor. Cannot analyze."
            )
            return

        time_data = sensors["Time"].flatten()

        # Overview 탭 업데이트
        self.fill_overview_tab(sensors, time_data)

        # Sensor Stats 탭 업데이트
        self.fill_sensor_stats_tab(sensors)

        # 기본 선택 센서
        if self.sensor_stats_combo is not None and self.sensor_stats_combo.count() > 0:
            self.sensor_stats_combo.setCurrentIndex(0)
            self.update_sensor_stats_view()

    def fill_overview_tab(self, sensors: dict, time_data: np.ndarray):
        """Overview 탭에 요약 통계 + 타임라인 플롯 그리기."""
        if self.overview_stats_table is None or self.overview_plot is None:
            return

        # ---- 메트릭 계산 ----
        duration = float(time_data[-1] - time_data[0])

        # Wheel speed 기반 vehicle speed 추정
        wheel_candidates = [
            "WHL_SpdFLVal",
            "WHL_SpdFRVal",
            "WHL_SpdRLVal",
            "WHL_SpdRRVal",
        ]
        speed_values = []
        for name in wheel_candidates:
            if name in sensors:
                speed_values.append(sensors[name].flatten())

        vehicle_speed = None
        if speed_values:
            vehicle_speed = np.mean(np.vstack(speed_values), axis=0)

        steering_angle = sensors.get("SAS_AnglVal", None)
        long_accel = sensors.get("IMU_LongAccelVal", None)
        lat_accel = sensors.get("IMU_LatAccelVal", None)

        metrics: list[tuple[str, str]] = []
        metrics.append(("Duration (s)", f"{duration:.2f}"))

        if vehicle_speed is not None:
            metrics.append(("Speed mean", f"{np.mean(vehicle_speed):.2f}"))
            metrics.append(("Speed max", f"{np.max(vehicle_speed):.2f}"))

        if steering_angle is not None:
            v = steering_angle.flatten()
            metrics.append(("Steering angle mean", f"{np.mean(v):.2f}"))
            metrics.append(("Steering angle max", f"{np.max(v):.2f}"))
            metrics.append(("Steering angle min", f"{np.min(v):.2f}"))

        if long_accel is not None:
            v = long_accel.flatten()
            metrics.append(("Long accel mean", f"{np.mean(v):.3f}"))
            metrics.append(("Long accel max", f"{np.max(v):.3f}"))
            metrics.append(("Long accel min", f"{np.min(v):.3f}"))

        if lat_accel is not None:
            v = lat_accel.flatten()
            metrics.append(("Lat accel mean", f"{np.mean(v):.3f}"))
            metrics.append(("Lat accel max", f"{np.max(v):.3f}"))
            metrics.append(("Lat accel min", f"{np.min(v):.3f}"))

        # ---- 테이블 채우기 ----
        self.overview_stats_table.setRowCount(len(metrics))
        for row, (name, val) in enumerate(metrics):
            item_name = QTableWidgetItem(name)
            item_val = QTableWidgetItem(val)
            self.overview_stats_table.setItem(row, 0, item_name)
            self.overview_stats_table.setItem(row, 1, item_val)

        # ---- 대표 타임라인 플롯 ----
        self.overview_plot.clear()

        plot_y = None
        ylabel = "Value"

        if vehicle_speed is not None:
            plot_y = vehicle_speed
            ylabel = "Vehicle speed"
        elif steering_angle is not None:
            plot_y = steering_angle.flatten()
            ylabel = "Steering angle"
        elif long_accel is not None:
            plot_y = long_accel.flatten()
            ylabel = "Long accel"

        if plot_y is not None:
            pen = pg.mkPen(width=2)
            self.overview_plot.plot(time_data, plot_y, pen=pen)
            self.overview_plot.setLabel("left", ylabel)
            self.overview_plot.setLabel("bottom", "Time (s)")

    def fill_sensor_stats_tab(self, sensors: dict):
        """Sensor Stats 탭의 센서 목록 콤보박스를 채운다."""
        if self.sensor_stats_combo is None:
            return

        self.sensor_stats_combo.blockSignals(True)
        self.sensor_stats_combo.clear()

        # Time 센서는 제외
        names = [k for k in sensors.keys() if k != "Time"]
        names.sort()
        for name in names:
            self.sensor_stats_combo.addItem(name)

        self.sensor_stats_combo.blockSignals(False)

        if self.sensor_stats_text is not None:
            if names:
                self.sensor_stats_text.setText(
                    "Select a sensor to see statistics."
                )
            else:
                self.sensor_stats_text.setText("No sensors available.")

        # 기존 히스토그램도 초기화
        if self.sensor_stats_plot is not None:
            self.sensor_stats_plot.clear()

    def update_sensor_stats_view(self):
        """Sensor Stats 탭에서 선택된 센서의 통계/히스토그램 갱신."""
        if (
            self.sensor_stats_combo is None
            or self.sensor_stats_plot is None
            or self.current_data is None
        ):
            return

        sensor_name = self.sensor_stats_combo.currentText()
        if not sensor_name:
            return

        sensors = self.current_data.get("sensors", {})
        if sensor_name not in sensors:
            return

        values = sensors[sensor_name].flatten().astype(float)
        if values.size == 0:
            if self.sensor_stats_text is not None:
                self.sensor_stats_text.setText(
                    f"Sensor: {sensor_name}\nNo data."
                )
            self.sensor_stats_plot.clear()
            return

        v_min = float(np.min(values))
        v_max = float(np.max(values))
        v_mean = float(np.mean(values))
        v_std = float(np.std(values))
        v_med = float(np.median(values))

        text = (
            f"Sensor: {sensor_name}\n"
            f"min: {v_min:.4f}\n"
            f"max: {v_max:.4f}\n"
            f"mean: {v_mean:.4f}\n"
            f"median: {v_med:.4f}\n"
            f"std: {v_std:.4f}"
        )
        if self.sensor_stats_text is not None:
            self.sensor_stats_text.setText(text)

        # 히스토그램 플롯
        self.sensor_stats_plot.clear()

        n = len(values)
        # 샘플 수에 따라 적당한 bin 수 선택
        bins = 20 if n > 200 else max(5, n // 10)
        hist, bin_edges = np.histogram(values, bins=bins)

        x = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        w = (bin_edges[1] - bin_edges[0]) * 0.9

        bg = pg.BarGraphItem(x=x, height=hist, width=w)
        self.sensor_stats_plot.addItem(bg)
        self.sensor_stats_plot.setLabel("bottom", "Value")
        self.sensor_stats_plot.setLabel("left", "Count")
