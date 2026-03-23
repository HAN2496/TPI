from datetime import datetime
from pathlib import Path


def artifact_dir(method, timestamp=None) -> tuple[Path, bool]:
    """
    timestamp=None              -> 새 타임스탬프 폴더 생성, eval_only=False
    timestamp="test"            -> 고정 test 폴더 (디버그용), eval_only=False
    timestamp="20250101_120000" -> 기존 폴더 재사용, eval_only=True

    Returns: (path, eval_only)
    """
    eval_only = timestamp is not None and timestamp != "test"
    name = timestamp if timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("artifacts") / method / name
    d.mkdir(parents=True, exist_ok=True)
    return d, eval_only
