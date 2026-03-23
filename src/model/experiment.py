from abc import ABC, abstractmethod
from pathlib import Path
import json
from datetime import datetime


class BaseExperiment(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def build(self) -> None:
        """데이터셋, 모델 아키텍처 초기화."""
        raise NotImplementedError

    @abstractmethod
    def train(self, out_dir: Path) -> dict:
        """학습 루프 실행. train metrics 반환."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, out_dir: Path) -> dict:
        """Test-time 평가. {split/driver_name: metrics_dict} 반환."""
        raise NotImplementedError

    @abstractmethod
    def save(self, out_dir: Path) -> None:
        """학습된 모델 저장 (trainer 내부에서 이미 저장하면 pass)."""
        raise NotImplementedError

    @abstractmethod
    def load(self, out_dir: Path) -> None:
        """저장된 모델 로드 (eval-only 모드)."""
        raise NotImplementedError

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        return {
            "method": self.method_name,
            "metrics": {**train_metrics, **eval_metrics},
            "completed_at": datetime.now().isoformat(),
        }

    def _log(self, *args, **kwargs):
        if self.cfg.verbose > 0:
            print(*args, **kwargs)

    def run(self, out_dir: Path, eval_only: bool = False) -> dict:
        self.build()
        if eval_only:
            self._log(f"[Eval-only] Loading from {out_dir}")
            self.load(out_dir)
            train_metrics = {}
        else:
            train_metrics = self.train(out_dir)
            self.save(out_dir)

        eval_metrics = self.evaluate(out_dir)
        summary = self.make_summary(train_metrics, eval_metrics)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return eval_metrics

    @property
    def method_name(self) -> str:
        return self.__class__.__name__.replace("Experiment", "").lower()
