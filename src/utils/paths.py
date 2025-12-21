from pathlib import Path

class ExperimentPaths:
    def __init__(self, driver_name, model_name, feature_version, time_range, tag=None, tag_as_subdir=False):
        run_id = f"{feature_version}_t{time_range[0]}-{time_range[1]}"
        self.run_dir = Path("artifacts") / driver_name / model_name / run_id

        if tag:
            if tag_as_subdir:
                self.run_dir = self.run_dir / tag
            else:
                self.run_dir = Path("artifacts") / driver_name / model_name / f"{run_id}_{tag}"

    def get(self, filename, create=False):
        if create:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir / filename

    @property
    def best_model(self):
        return str(self.get("best_model.pt"))
    @property
    def config(self):
        return str(self.get("config.yaml"))
    @property
    def history(self):
        return str(self.get("history.json"))