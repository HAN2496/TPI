from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QTimer, QTime, Signal
from PySide6.QtWidgets import QSlider


class PlaybackController(QObject):
    """센서 시간축 기준으로 재생/일시정지를 관리하는 컨트롤러."""

    playback_started = Signal()
    playback_stopped = Signal()

    def __init__(self, slider: QSlider, interval_ms: int = 33, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._slider = slider
        self._interval_ms = interval_ms

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timeout)

        self._time_array = None
        self._is_playing: bool = False
        self._play_start_time: Optional[QTime] = None
        self._play_start_step: int = 0

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def time_array(self):
        return self._time_array

    def set_time_array(self, time_array):
        """센서 Time 데이터(1D numpy array)를 설정."""
        self._time_array = time_array

    def toggle(self):
        if self._is_playing:
            self.stop()
        else:
            self.play()

    def play(self):
        if self._time_array is None or len(self._time_array) == 0:
            return
        if self._is_playing:
            return

        self._is_playing = True
        self._play_start_time = QTime.currentTime()
        self._play_start_step = self._slider.value()
        self._timer.start(self._interval_ms)
        self.playback_started.emit()

    def stop(self):
        if not self._is_playing:
            return
        self._is_playing = False
        self._timer.stop()
        self.playback_stopped.emit()

    def _on_timeout(self):
        if self._time_array is None or self._play_start_time is None:
            return
        if len(self._time_array) == 0:
            return

        elapsed_ms = self._play_start_time.msecsTo(QTime.currentTime())
        elapsed_sec = elapsed_ms / 1000.0

        start_sensor_time = self._time_array[self._play_start_step]
        target_sensor_time = start_sensor_time + elapsed_sec

        target_step = int(np.searchsorted(self._time_array, target_sensor_time))
        max_index = len(self._time_array) - 1

        if target_step >= max_index:
            # 마지막 프레임에서 정지
            target_step = max_index
            self._slider.setValue(target_step)
            self.stop()
        else:
            self._slider.setValue(target_step)