from pathlib import Path

class ExperimentPaths:
    def __init__(self, driver_name, model_name, feature_version, time_range, tag=None):
        run_id = f"{feature_version}_t{time_range[0]}-{time_range[1]}"
        if tag:
            run_id += f"_{tag}"

        self.run_dir = Path("artifacts") / driver_name / model_name / run_id

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