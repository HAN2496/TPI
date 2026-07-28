import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path


class Cache:
    """드라이버 단위 raw CSV 디스크 캐시 (datasets/.cache/<driver>_<hash>.npz).

    키 = (episode id, CSV size, mtime) 해시 — CSV가 바뀌면 자동 무효화.
    한 드라이버의 에피소드를 처음 만질 때 통째로 로드/생성한다 (npz가 CSV 파싱보다 ~20배 빠름).
    """

    def __init__(self, root):
        self.dir = Path(root) / ".cache"
        self.eps = {}     # driver -> [Episode]
        self.frames = {}  # episode id -> DataFrame

    def register(self, episodes):
        for ep in episodes:
            self.eps.setdefault(ep.driver, []).append(ep)

    def frame(self, ep):
        if ep.id not in self.frames:
            self._load(ep.driver)
        return self.frames[ep.id]

    def _load(self, driver):
        eps = self.eps[driver]
        recs = [(e.id, e.state.stat().st_size, e.state.stat().st_mtime_ns) for e in eps]
        key = hashlib.md5(json.dumps(recs).encode()).hexdigest()[:16]
        path = self.dir / f"{driver}_{key}.npz"
        if path.exists():
            z = np.load(path)
            for e in eps:
                self.frames[e.id] = pd.DataFrame(z[f"d_{e.id}"], columns=z[f"c_{e.id}"].tolist())
            return
        arrays = {}
        for e in eps:
            df = pd.read_csv(e.state)
            self.frames[e.id] = df
            arrays[f"d_{e.id}"] = df.to_numpy(np.float32)
            arrays[f"c_{e.id}"] = np.asarray(df.columns, dtype=str)
        self.dir.mkdir(exist_ok=True)
        for old in self.dir.glob(f"{driver}_*.npz"):
            old.unlink()
        np.savez(path, **arrays)
