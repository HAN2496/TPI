from pathlib import Path

DATASETS_ROOT = Path("datasets")

DEFAULT_VIDEO_FPS = 30
UPDATE_INTERVAL_MS = 33
Y_AXIS_PADDING_RATIO = 0.1

SENSOR_GROUPS = [
    ("IMU", ["IMU_VerAccelVal", "IMU_LongAccelVal", "IMU_LatAccelVal"]),
    ("Wheel", ["WHL_SpdFLVal", "WHL_SpdFRVal", "WHL_SpdRLVal", "WHL_SpdRRVal"]),
    ("Steering", ["SAS_AnglVal", "SAS_SpdVal"]),
]

SENSOR_COLORS = [
    "#FF5555",  # IMU_VerAccelVal
    "#50FA7B",  # IMU_LongAccelVal
    "#8BE9FD",  # IMU_LatAccelVal
    "#BD93F9",  # WHL_SpdFLVal
    "#FFB86C",  # WHL_SpdFRVal
    "#FF79C6",  # WHL_SpdRLVal
    "#F1FA8C",  # WHL_SpdRRVal
    "#6272A4",  # SAS_AnglVal
    "#FFFFFF",  # SAS_SpdVal
]
