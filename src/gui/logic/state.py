from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class RunState:
    folder_path: str = ""
    timestamp: str = ""
    data: Optional[dict[str, Any]] = None
    video_capture: Any = None
    video_fps: float = 30.0
