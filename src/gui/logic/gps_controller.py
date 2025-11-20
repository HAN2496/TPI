from __future__ import annotations

from typing import Sequence
import pyqtgraph as pg


class GpsController:
    """
    GPS 궤적을 pyqtgraph 플롯과 웹 지도(QWebEngineView)에 표시하는 헬퍼.
    """

    def __init__(self, plot_widget_plain, web_view, stack_widget):
        self._plot_widget_plain: pg.PlotWidget = plot_widget_plain
        self._web_view = web_view
        self._stack_widget = stack_widget

        self._use_map: bool = False
        self._js_ready: bool = False

    def set_use_map(self, use_map: bool):
        """MAP 모드 on/off."""
        self._use_map = use_map
        if self._stack_widget is not None:
            self._stack_widget.setCurrentIndex(1 if use_map else 0)

    def on_page_load_finished(
        self,
        ok: bool,
        gps_coords: Sequence[Sequence[float]] | None,
        time_data=None,
        step: int | None = None,
    ):
        """map.html 로드 완료 시 호출."""
        self._js_ready = ok
        if not ok:
            return

        if not self._use_map or not gps_coords:
            return

        self.send_full_track(gps_coords)

        if time_data is not None and step is not None:
            self.update(gps_coords, time_data, step)

    def update(self, gps_coords, time_data, step: int):
        """현재 시간에 맞게 GPS 표시."""
        if not gps_coords or time_data is None or len(time_data) == 0:
            return
        if step < 0 or step >= len(time_data):
            return

        # 센서 시간 기준으로 GPS 인덱스 매핑
        progress = step / max(1, (len(time_data) - 1))
        gps_index = min(int(progress * (len(gps_coords) - 1)), len(gps_coords) - 1)

        coord = gps_coords[gps_index]
        lon = float(coord[0])
        lat = float(coord[1])

        # 플레인 모드(pyqtgraph)
        if not self._use_map or self._web_view is None or not self._js_ready:
            if self._plot_widget_plain is None:
                return

            self._plot_widget_plain.clear()

            lons = [c[0] for c in gps_coords]
            lats = [c[1] for c in gps_coords]

            pen = pg.mkPen("#FF5555", width=2)
            self._plot_widget_plain.plot(lons, lats, pen=pen, symbol=None)

            self._plot_widget_plain.plot(
                [lon],
                [lat],
                pen=None,
                symbol="o",
                symbolSize=10,
                symbolBrush="g",
            )
            return

        # 웹 지도 모드
        if self._js_ready and self._web_view is not None:
            js_code = f"window.updateGps({lat:.8f}, {lon:.8f});"
            self._web_view.page().runJavaScript(js_code)

    def send_full_track(self, gps_coords):
        """전체 GPS 궤적을 웹 페이지로 전송."""
        if not self._use_map or not self._js_ready:
            return
        if not gps_coords or self._web_view is None:
            return

        lats = [coord[1] for coord in gps_coords]
        lons = [coord[0] for coord in gps_coords]

        js_lat_list = "[" + ",".join(f"{lat:.8f}" for lat in lats) + "]"
        js_lon_list = "[" + ",".join(f"{lon:.8f}" for lon in lons) + "]"

        js_code = f"window.setFullTrack({js_lat_list}, {js_lon_list});"
        self._web_view.page().runJavaScript(js_code)

    @property
    def use_map(self) -> bool:
        return self._use_map

    @property
    def js_ready(self) -> bool:
        return self._js_ready
