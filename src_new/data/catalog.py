import os
import json
from datetime import datetime
from pathlib import Path


_CATALOG_CACHE: dict[str, "DatasetCatalog"] = {}


def get_catalog(root: str | Path = "datasets") -> "DatasetCatalog":
    key = str(Path(root).resolve())
    if key not in _CATALOG_CACHE:
        _CATALOG_CACHE[key] = DatasetCatalog.load(root)
    return _CATALOG_CACHE[key]


class DatasetCatalog:
    CATALOG_FILE = "catalog.json"

    def __init__(self, episodes: list[dict], root: Path):
        self.episodes = episodes
        self.root = root

    @classmethod
    def build(cls, root: str | Path) -> "DatasetCatalog":
        root = Path(root)
        episodes = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            filenames.sort()

            csv_files = [f for f in filenames if f.lower().endswith(".csv") and "_state_" in f
                         and not f.lower().endswith("_smooth.csv")]
            txt_files = [f for f in filenames if f.lower().endswith(".txt") and "_info_" in f]

            csv_by_ts = {f.split("_state_")[0]: f for f in csv_files}
            txt_by_ts = {f.split("_info_")[0]: f for f in txt_files}
            common_ts = set(csv_by_ts.keys()) & set(txt_by_ts.keys())

            for ts in sorted(common_ts):
                info_path = os.path.join(dirpath, txt_by_ts[ts])
                csv_path = os.path.join(dirpath, csv_by_ts[ts])

                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)

                driver = info.get("Driver")
                if not driver:
                    continue

                # Label from filename
                base = os.path.splitext(csv_by_ts[ts])[0]
                label_raw = base.split("_")[-1]
                if label_raw == "True":
                    label = True
                elif label_raw == "False":
                    label = False
                else:
                    label = None

                # Count timesteps
                import pandas as pd
                try:
                    states = pd.read_csv(csv_path)
                    n_timesteps = len(states)
                except Exception:
                    n_timesteps = None

                # Source: vmc if folder contains "vmc", else "real"
                source = "vmc" if "vmc" in str(dirpath).lower() else "real"

                episodes.append({
                    "id": ts,
                    "driver": driver,
                    "label": label,
                    "dirpath": str(Path(dirpath).relative_to(root)),
                    "csv": csv_by_ts[ts],
                    "info": txt_by_ts[ts],
                    "source": source,
                    "n_timesteps": n_timesteps,
                })

        catalog_data = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "episodes": episodes,
        }

        catalog_path = root / cls.CATALOG_FILE
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog_data, f, indent=2, ensure_ascii=False)

        print(f"[Catalog] Built {len(episodes)} episodes → {catalog_path}")
        return cls(episodes, root)

    @classmethod
    def load(cls, root: str | Path) -> "DatasetCatalog":
        root = Path(root)
        catalog_path = root / cls.CATALOG_FILE

        if not catalog_path.exists():
            print(f"[Catalog] {catalog_path} not found, building...")
            return cls.build(root)

        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(data["episodes"], root)

    def query(self, drivers: list[str] | None = None,
              source: str | None = None,
              label: bool | None = None) -> list[dict]:
        result = self.episodes

        if drivers is not None:
            result = [e for e in result if e["driver"] in drivers]
        if source is not None:
            result = [e for e in result if e["source"] == source]
        if label is not None:
            result = [e for e in result if e["label"] == label]

        return result

    def invalidate(self):
        catalog_path = self.root / self.CATALOG_FILE
        if catalog_path.exists():
            catalog_path.unlink()
        _CATALOG_CACHE.pop(str(self.root.resolve()), None)
        print(f"[Catalog] Invalidated {catalog_path}")
