import os

import numpy as np
import cv2

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QSplitter,
    QScrollArea,
    QGridLayout,
    QWidget,
    QHeaderView,
)

# PyDracula
from .pydracula.modules import *
from .pydracula.widgets import *

# 환경 설정
os.environ["QT_FONT_DPI"] = "96"  # High DPI 이슈 해결

# 프로젝트 모듈
from ..utils.data_loader import DatasetManager
from .analysis_page import AnalysisPage

from ..configs.gui import (
    DATASETS_ROOT,
    DEFAULT_VIDEO_FPS,
    UPDATE_INTERVAL_MS,
    Y_AXIS_PADDING_RATIO,
    SENSOR_GROUPS,
    SENSOR_COLORS,
)
from .logic.state import RunState
from .logic.playback import PlaybackController
from .logic.sensor_plotter import SensorPlotter
from .logic.gps_controller import GpsController
from .ui.dataset_panel import DatasetPanel
from .ui.video_panel import VideoPanel
from .ui.gps_panel import GpsPanel
from .ui.sensor_panel import SensorPanel
from .ui.control_panel import ControlPanel

# PyDracula에서 사용하는 전역
widgets = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # PyDracula UI 세팅
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        global widgets
        widgets = self.ui

        # 상태
        self.state = RunState()

        # 컨트롤러들 (init_tpi_viewer에서 초기화)
        self.dataset_panel = None
        self.video_panel = None
        self.gps_panel = None
        self.sensor_panel = None
        self.control_panel = None

        self.playback = None
        self.sensor_plotter = None
        self.gps_controller = None

        # Analysis page
        self.analysis_page = None

        # 기본 설정
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        title = "TPI Dataset Viewer - PyDracula"
        description = "TPI Dataset Viewer - Interactive visualization tool for driving data"
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        self.resize(1664, 936)

        # 사이드 메뉴/설정 버튼 연결
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))
        UIFunctions.uiDefinitions(self)

        widgets.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)
        widgets.btn_save.clicked.connect(self.buttonClick)

        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)

        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)

        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)

        # TPI Viewer + 분석 페이지 초기화
        self.init_tpi_viewer()
        self.init_analysis_page()

        # 테마 (원하면 True 로 변경)
        useCustomTheme = False
        themeFile = r"themes\py_dracula_light.qss"
        if useCustomTheme:
            UIFunctions.theme(self, themeFile, True)
            AppFunctions.setThemeHack(self)

        # 기본 페이지
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))

        self.show()

    # ------------------------------------------------------------------
    # 버튼 클릭 (좌측 메뉴)
    # ------------------------------------------------------------------
    def buttonClick(self):
        btn = self.sender()
        btnName = btn.objectName()

        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.widgets)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        if btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.new_page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        if btnName == "btn_save":
            print("Save BTN clicked!")

        print(f'Button "{btnName}" pressed!')

    # ------------------------------------------------------------------
    # TPI Viewer 초기화 (home 페이지 내부 레이아웃 구성)
    # ------------------------------------------------------------------
    def init_tpi_viewer(self):
        widgets.home.setStyleSheet("background: transparent;")

        home_layout = QVBoxLayout(widgets.home)
        splitter = QSplitter(Qt.Horizontal)
        home_layout.addWidget(splitter)

        # 좌측: DatasetPanel
        self.dataset_panel = DatasetPanel(DATASETS_ROOT)
        self.dataset_panel.dataset_selected.connect(self.load_data)
        splitter.addWidget(self.dataset_panel)

        # 우측: 스크롤 영역
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        right_widget = QWidget()
        right_layout = QGridLayout(right_widget)
        right_layout.setContentsMargins(50, 0, 50, 0)

        scroll_area.setWidget(right_widget)
        splitter.addWidget(scroll_area)
        splitter.setSizes([250, 1000])

        # (0,0) Video
        self.video_panel = VideoPanel()
        right_layout.addWidget(self.video_panel, 0, 0)

        # (0,1) GPS
        self.gps_panel = GpsPanel()
        right_layout.addWidget(self.gps_panel, 0, 1)

        # (1,0-1) Sensors
        self.sensor_panel = SensorPanel(SENSOR_GROUPS, SENSOR_COLORS)
        right_layout.addWidget(self.sensor_panel, 1, 0, 1, 2)

        # (2,0-1) Controls
        self.control_panel = ControlPanel()
        right_layout.addWidget(self.control_panel, 2, 0, 1, 2)

        # 컨트롤러 생성
        self.playback = PlaybackController(self.control_panel.slider, UPDATE_INTERVAL_MS, parent=self)
        self.sensor_plotter = SensorPlotter(
            main_plot_widget=self.sensor_panel.main_plot_widget,
            imu_plot=self.sensor_panel.imu_plot,
            wheel_plot=self.sensor_panel.wheel_plot,
            steering_plot=self.sensor_panel.steering_plot,
            sensor_groups=SENSOR_GROUPS,
            sensor_colors=self.sensor_panel.get_sensor_colors(),
            y_padding_ratio=Y_AXIS_PADDING_RATIO,
        )
        self.gps_controller = GpsController(
            plot_widget_plain=self.gps_panel.plot_widget,
            web_view=self.gps_panel.web_view,
            stack_widget=self.gps_panel.stack_widget,
        )

        # 시그널 연결
        self.sensor_panel.sensors_changed.connect(self.on_sensors_changed)
        self.gps_panel.map_toggled.connect(self.on_map_toggled)
        self.gps_panel.web_load_finished.connect(self.on_gps_page_load_finished)

        self.control_panel.play_clicked.connect(self.on_play_clicked)
        self.control_panel.step_clicked.connect(self.on_step_clicked)
        self.control_panel.slider_changed.connect(self.on_slider_changed)

        self.playback.playback_started.connect(lambda: self.control_panel.set_playing(True))
        self.playback.playback_stopped.connect(lambda: self.control_panel.set_playing(False))

        # 첫 번째 데이터셋 자동 로딩
        first = self.dataset_panel.get_first_dataset()
        if first is not None:
            folder, timestamp = first
            self.load_data(folder, timestamp)

    # ------------------------------------------------------------------
    # 분석 페이지 초기화 (new_page 안에 AnalysisPage 삽입)
    # ------------------------------------------------------------------
    def init_analysis_page(self):
        self.analysis_page = AnalysisPage(self)
        container = widgets.new_page

        layout = container.layout()
        if layout is None:
            layout = QVBoxLayout(container)
        else:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.analysis_page)

    # ------------------------------------------------------------------
    # 데이터 로딩
    # ------------------------------------------------------------------
    def load_data(self, folder_path: str, timestamp: str):
        if self.playback is not None:
            self.playback.stop()

        # 기존 비디오 자원 해제
        if self.state.video_capture is not None:
            try:
                self.state.video_capture.release()
            except Exception:
                pass
            self.state.video_capture = None

        try:
            loader = DatasetManager(folder_path)
            data = loader.load_all(timestamp, folder_path)

            self.state.folder_path = folder_path
            self.state.timestamp = timestamp
            self.state.data = data

            # Info 패널 업데이트
            info = data.get("info", {})
            self.dataset_panel.set_info(info)

            sensors = data.get("sensors", {})
            time_data = None
            if "Time" in sensors:
                time_data = sensors["Time"].flatten()
                self.control_panel.setup_time_controls(len(time_data))
                self.playback.set_time_array(time_data)
            else:
                self.control_panel.setup_time_controls(0)
                self.playback.set_time_array(None)

            # 비디오 로딩
            video_capture = None
            video_fps = DEFAULT_VIDEO_FPS
            video_path = loader.get_video_path(timestamp, folder_path)
            if video_path and os.path.exists(video_path):
                video_capture = cv2.VideoCapture(video_path)
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0:
                    video_fps = fps

            self.state.video_capture = video_capture
            self.state.video_fps = video_fps

            # 센서 플롯 재구성
            if time_data is not None:
                active_sensors = self.sensor_panel.get_active_sensors()
                self.sensor_plotter.rebuild(sensors, time_data, active_sensors)
                self.sensor_plotter.update_vlines(time_data, 0)

            # GPS 초기 업데이트
            gps_coords = data.get("gps") or []
            if time_data is not None:
                self.gps_controller.send_full_track(gps_coords)
                self.gps_controller.update(gps_coords, time_data, 0)

            # 분석 페이지에 run 설정
            label = self.dataset_panel.get_label(folder_path, timestamp)
            if self.analysis_page is not None:
                self.analysis_page.set_run(folder_path, timestamp, data, label)

            # 시각화 초기 업데이트
            if time_data is not None and len(time_data) > 0:
                self.update_visualization(0)

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback

            traceback.print_exc()

    # ------------------------------------------------------------------
    # 센서 / GPS / 비디오 시각화
    # ------------------------------------------------------------------
    def update_visualization(self, step: int):
        if self.state.data is None:
            return

        sensors = self.state.data.get("sensors", {})
        if "Time" not in sensors:
            return

        time_data = sensors["Time"].flatten()
        if step < 0 or step >= len(time_data):
            return

        current_time = float(time_data[step])
        total_time = float(time_data[-1])

        # 시간 라벨
        self.control_panel.set_time_label(current_time, total_time)

        # 비디오 프레임 업데이트
        self._update_video_frame(current_time)

        # 센서 플롯 vline 업데이트
        self.sensor_plotter.update_vlines(time_data, step)

        # GPS 업데이트
        gps_coords = self.state.data.get("gps") or []
        self.gps_controller.update(gps_coords, time_data, step)

    def _update_video_frame(self, current_time: float):
        cap = self.state.video_capture
        if cap is None or not cap.isOpened():
            return

        frame_number = int(current_time * self.state.video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.video_panel.show_frame(frame_rgb)

    # ------------------------------------------------------------------
    # 시그널 핸들러
    # ------------------------------------------------------------------
    def on_sensors_changed(self, active_sensors: list[str]):
        if self.state.data is None:
            return
        sensors = self.state.data.get("sensors", {})
        if "Time" not in sensors:
            return

        time_data = sensors["Time"].flatten()
        self.sensor_plotter.rebuild(sensors, time_data, active_sensors)

        step = self.control_panel.slider.value()
        self.sensor_plotter.update_vlines(time_data, step)

    def on_map_toggled(self, use_map: bool):
        self.gps_controller.set_use_map(use_map)

        if self.state.data is None:
            return
        sensors = self.state.data.get("sensors", {})
        if "Time" not in sensors:
            return
        time_data = sensors["Time"].flatten()
        gps_coords = self.state.data.get("gps") or []

        self.gps_controller.send_full_track(gps_coords)
        step = self.control_panel.slider.value()
        self.gps_controller.update(gps_coords, time_data, step)

    def on_gps_page_load_finished(self, ok: bool):
        gps_coords = []
        time_data = None
        step = None

        if self.state.data is not None:
            gps_coords = self.state.data.get("gps") or []
            sensors = self.state.data.get("sensors", {})
            if "Time" in sensors:
                time_data = sensors["Time"].flatten()
                step = self.control_panel.slider.value()

        self.gps_controller.on_page_load_finished(ok, gps_coords, time_data, step)

    def on_play_clicked(self):
        if self.playback.time_array is None:
            return
        self.playback.toggle()

    def on_step_clicked(self, delta_seconds: int):
        if self.playback.time_array is None:
            return
        if self.state.data is None:
            return

        # 재생 중이면 멈추고
        self.playback.stop()

        time_data = self.playback.time_array
        slider = self.control_panel.slider

        current_step = slider.value()
        if current_step < 0 or current_step >= len(time_data):
            current_step = 0

        current_time = float(time_data[current_step])
        target_time = current_time + float(delta_seconds)

        target_step = int(np.searchsorted(time_data, target_time))
        target_step = max(0, min(len(time_data) - 1, target_step))
        slider.setValue(target_step)

    def on_slider_changed(self, step: int):
        self.update_visualization(step)

    # ------------------------------------------------------------------
    # Qt 이벤트 오버라이드
    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        UIFunctions.resize_grips(self)
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()
        if event.buttons() == Qt.LeftButton:
            print("Mouse click: LEFT CLICK")
        if event.buttons() == Qt.RightButton:
            print("Mouse click: RIGHT CLICK")
        super().mousePressEvent(event)

    def closeEvent(self, event):
        if self.state.video_capture is not None:
            try:
                self.state.video_capture.release()
            except Exception:
                pass
        event.accept()

