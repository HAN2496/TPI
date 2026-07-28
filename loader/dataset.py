import os
import json
import pandas as pd
from pathlib import Path

from .signals import Signals
from .cache import Cache

LABELS = {"True": True, "False": False, "None": None}
_SCAN = {}


def _by(files, tag, ext):
    return {f.split(tag)[0]: f for f in files
            if tag in f and f.endswith(ext) and not f.endswith("_smooth.csv")}


def _scan(root):
    root = str(root)
    if root in _SCAN:
        return _SCAN[root]
    records = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        state = _by(filenames, "_state_", ".csv")
        info = _by(filenames, "_info_", ".txt")
        scene = _by(filenames, "_scene_", ".mp4")
        gps = _by(filenames, "_gps_", ".kml")
        for ts in sorted(state.keys() & info.keys()):
            d = Path(dirpath)
            meta = json.loads((d / info[ts]).read_text(encoding="utf-8"))
            records.append({
                "id": ts,
                "driver": meta["Driver"],
                "label": LABELS[state[ts].rsplit("_", 1)[1][:-4]],
                "state": d / state[ts],
                "video": d / scene[ts] if ts in scene else None,
                "gps": d / gps[ts] if ts in gps else None,
                "meta": meta,
            })
    _SCAN[root] = records
    return records


class Episode:
    def __init__(self, rec, cache=None):
        self.id = rec["id"]
        self.driver = rec["driver"]
        self.label = rec["label"]
        self.state = rec["state"]
        self.video = rec["video"]
        self.gps = rec["gps"]
        self.meta = rec["meta"]
        self._cache = cache
        self._signals = None

    @property
    def signals(self):
        if self._signals is None:
            df = self._cache.frame(self) if self._cache else pd.read_csv(self.state)
            self._signals = Signals(df)
        return self._signals

    @property
    def event_time(self):
        return self.meta["Time_setup"]["A"]

    @property
    def post(self):
        return self.meta["Time_setup"]["C"]

    @property
    def thresholds(self):
        return self.meta["Threshold Value"]

    @property
    def vehicle(self):
        return self.meta["Vehicle"]

    @property
    def memo(self):
        return self.meta["Memo"]

    def info(self):
        print(f"Episode {self.id}")
        print(f"  driver    : {self.driver}")
        print(f"  label     : {self.label}")
        print(f"  vehicle   : {self.vehicle}")
        print(f"  event     : t={self.event_time}s  (pre {self.event_time}s / post {self.post}s)")
        print(f"  threshold : {self.thresholds}")
        print(f"  memo      : {self.memo}")

    def __repr__(self):
        return f"Episode({self.id}, {self.driver}, label={self.label})"


class EpisodeSet:
    def __init__(self, episodes):
        self.episodes = list(episodes)

    def filter(self, **kw):
        def ok(ep):
            return all(getattr(ep, k) in v if isinstance(v, (list, tuple, set)) else getattr(ep, k) == v
                       for k, v in kw.items())
        return EpisodeSet(ep for ep in self.episodes if ok(ep))

    def by_driver(self):
        groups = {}
        for ep in self.episodes:
            groups.setdefault(ep.driver, []).append(ep)
        return {name: EpisodeSet(eps) for name, eps in groups.items()}

    def summary(self):
        y = [ep.label for ep in self.episodes]
        drv = sorted({ep.driver for ep in self.episodes})
        print(f"{len(y)} episodes | True {y.count(True)} / False {y.count(False)} / None {y.count(None)}")
        print(f"  drivers ({len(drv)}): {drv}")

    def __len__(self):
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    def __getitem__(self, i):
        return self.episodes[i]

    def __repr__(self):
        return f"EpisodeSet({len(self.episodes)} episodes)"


class Driver:
    def __init__(self, name, episodes):
        self.name = name
        self.episodes = episodes

    def filter(self, **kw):
        return self.episodes.filter(**kw)

    def __len__(self):
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    def __getitem__(self, i):
        return self.episodes[i]

    def __repr__(self):
        y = [ep.label for ep in self.episodes]
        return f"Driver({self.name}, {len(y)} episodes, True {y.count(True)} / False {y.count(False)})"


class Dataset:
    def __init__(self, root="datasets", cache=True):
        self.root = Path(root)
        c = Cache(self.root) if cache else None
        self.episodes = EpisodeSet(Episode(r, c) for r in _scan(root))
        if c:
            c.register(self.episodes)
        self.drivers = {name: Driver(name, eps) for name, eps in self.episodes.by_driver().items()}

    @property
    def names(self):
        return list(self.drivers)

    def summary(self):
        self.episodes.summary()

    def __getitem__(self, name):
        return self.drivers[name]

    def __iter__(self):
        return iter(self.drivers.values())

    def __len__(self):
        return len(self.drivers)

    def __repr__(self):
        return f"Dataset({len(self.drivers)} drivers, {len(self.episodes)} episodes)"
