from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QGroupBox,
)

from src.utils.data_loader import DatasetManager


class DatasetPanel(QWidget):
    """좌측 데이터셋 트리 + Info 영역."""

    dataset_selected = Signal(str, str)  # folder_path, timestamp

    def __init__(self, datasets_root: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._datasets_root = datasets_root

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("📁 Datasets")
        title.setStyleSheet("font-size: 12pt; font-weight: bold; color: rgb(189, 147, 249);")
        layout.addWidget(title)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Dataset Files")
        self.tree.setIndentation(20)
        self.tree.setAnimated(True)
        self.tree.setExpandsOnDoubleClick(True)
        self.tree.setFocusPolicy(Qt.NoFocus)

        self.tree.setStyleSheet(
            """
            QTreeWidget {
                selection-background-color: transparent;
            }
            QTreeWidget::item {
                padding: 2px 4px;
                border: 0px;
                border-radius: 0px;
            }
            QTreeWidget::item:hover {
                background-color: rgba(80, 80, 80, 100);
            }
            QTreeWidget::item:selected {
                background-color: rgba(60, 60, 60, 180);
            }
            """
        )

        self.tree.header().setStretchLastSection(True)
        self.tree.itemClicked.connect(self._on_item_clicked)

        layout.addWidget(self.tree)

        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("No data loaded")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_group)

        self.load_datasets()

    # ------------------------------------------------------------------
    # 데이터셋 로딩 및 트리 구성
    # ------------------------------------------------------------------
    def load_datasets(self):
        self.tree.clear()
        root = self._datasets_root
        if not root.exists():
            return

        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue

            folder_item = QTreeWidgetItem(self.tree, [f"📂 {folder.name}"])
            folder_item.setForeground(0, QColor(189, 147, 249))

            loader = DatasetManager(str(folder))
            timestamps = loader.get_file_list(str(folder))

            for idx, timestamp in enumerate(timestamps):
                label = self._get_label_for_timestamp(folder, timestamp)
                label_emoji = "🟢" if str(label).lower() == "true" else "🔴"
                item_text = f"🎬 {timestamp}  |  {label_emoji}"

                child_item = QTreeWidgetItem(folder_item, [item_text])
                child_item.setData(0, Qt.UserRole, {"folder": str(folder), "timestamp": timestamp})

                if idx % 2 == 0:
                    child_item.setForeground(0, QColor(221, 221, 221))
                else:
                    child_item.setForeground(0, QColor(200, 200, 200))

            if len(timestamps) <= 5:
                folder_item.setExpanded(True)

    def _get_label_for_timestamp(self, folder: Path, timestamp: str) -> str:
        pattern = f"{timestamp}_*_*.txt"
        label_files = list(folder.glob(pattern))
        if label_files:
            return label_files[0].stem.split("_")[-1]
        return "Unknown"

    # ------------------------------------------------------------------
    # 외부에서 사용하는 헬퍼들
    # ------------------------------------------------------------------
    def get_first_dataset(self) -> Optional[tuple[str, str]]:
        """첫 번째 데이터셋 (folder, timestamp)을 반환하고 선택까지 해준다."""
        if self.tree.topLevelItemCount() == 0:
            return None

        folder_item = self.tree.topLevelItem(0)
        if folder_item.childCount() == 0:
            return None

        first_child = folder_item.child(0)
        data = first_child.data(0, Qt.UserRole)
        if not data:
            return None

        self.tree.setCurrentItem(first_child)
        return data["folder"], data["timestamp"]

    def set_info(self, info: dict):
        """우측 Info 패널 업데이트."""
        if not info:
            self.info_label.setText("No data loaded")
            return

        driver = info.get("Driver", "N/A")
        vehicle = info.get("Vehicle", "N/A")
        passengers = info.get("Passengers", "N/A")

        text = f"Driver: {driver}\nVehicle: {vehicle}\nPassengers: {passengers}"
        self.info_label.setText(text)

    def get_label(self, folder_path: str, timestamp: str) -> str:
        """label 파일에서 현재 run 의 label(true/false 등)을 가져온다."""
        folder = Path(folder_path)
        return self._get_label_for_timestamp(folder, timestamp)

    # ------------------------------------------------------------------
    # 내부 이벤트
    # ------------------------------------------------------------------
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        folder = data["folder"]
        timestamp = data["timestamp"]
        self.dataset_selected.emit(folder, timestamp)
