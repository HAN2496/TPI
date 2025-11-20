import sys

def run_gui():
    """Launch GUI application"""
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from src.gui.gui_main import MainWindow

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    MainWindow()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()