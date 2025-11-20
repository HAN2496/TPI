from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QStackedWidget,
)


class GpsPanel(QWidget):
    """GPS 궤적 플롯 + MAP 토글 + QWebEngineView 를 포함하는 패널."""

    map_toggled = Signal(bool)       # True: MAP on
    web_load_finished = Signal(bool) # map.html loadFinished

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 헤더
        header_layout = QHBoxLayout()
        title = QLabel("GPS Path:")
        title.setStyleSheet("font-size: 11pt; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.map_checkbox = QCheckBox("MAP")
        self.map_checkbox.setChecked(False)
        self.map_checkbox.stateChanged.connect(
            lambda state: self.map_toggled.emit(state != 0)
        )
        header_layout.addWidget(self.map_checkbox)

        layout.addLayout(header_layout)

        # pyqtgraph 플롯
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumHeight(300)
        self._plot_widget.setAspectLocked(True)
        self._plot_widget.setLabel("left", "Latitude")
        self._plot_widget.setLabel("bottom", "Longitude")
        self._plot_widget.setBackground((33, 37, 43))

        # 웹 뷰
        self._web_view = QWebEngineView()
        self._web_view.setMinimumHeight(300)
        self._web_view.loadFinished.connect(self.web_load_finished)

        # map.html 로드 (현재 작업 디렉터리 기준)
        try:
            html_path = Path("ui/map.html")
            if html_path.exists():
                with html_path.open("r", encoding="utf-8") as f:
                    html = f.read()
                self._web_view.setHtml(html)
        except Exception as e:
            print("[GPS] map.html 로드 실패:", e)

        # 스택
        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._plot_widget)  # index 0: plain
        self._stack.addWidget(self._web_view)     # index 1: map
        self._stack.setCurrentIndex(0)

        layout.addWidget(self._stack)

    # ------------------------------------------------------------------
    # 외부에서 접근하는 프로퍼티들
    # ------------------------------------------------------------------
    @property
    def plot_widget(self):
        return self._plot_widget

    @property
    def web_view(self):
        return self._web_view

    @property
    def stack_widget(self):
        return self._stack
