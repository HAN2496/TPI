import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .metrics import save_metrics_txt


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Run:
    """실험 1회 = 아티팩트 폴더 + 스펙 스냅샷 + 지표 수집.

    cfg.timestamp:
      None              -> 새 타임스탬프 폴더 (학습)
      "test"            -> 고정 디버그 폴더 (학습)
      "20250101_120000" -> 기존 폴더 재사용 (eval_only=True)
    """

    def __init__(self, name, cfg, root="outputs"):
        seed_all(cfg.seed)
        self.name = name
        self.cfg = cfg
        self.eval_only = cfg.timestamp is not None and cfg.timestamp != "test"
        ts = cfg.timestamp if cfg.timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / name / ts
        self.plots = self.dir / "plots"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        (self.dir / "cfg.json").write_text(
            json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def at(self, timestamp):
        """같은 method의 과거 run 폴더 (부분 로드용)."""
        return self.dir.parent / timestamp

    def finish(self):
        save_metrics_txt(self.metrics, self.dir / "metrics.txt")
        summary = {"method": self.name, "metrics": self.metrics,
                   "completed_at": datetime.now().isoformat()}
        (self.dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"  -> {self.dir}")
        return self.metrics
