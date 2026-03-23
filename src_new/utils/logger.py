import sys
from pathlib import Path
from datetime import datetime


class Tee:
    def __init__(self, file_path, mode='a'):
        self.file = open(file_path, mode, encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


class ExperimentLogger:
    def __init__(self, log_dir, experiment_name, add_timestamp=True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if add_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.txt"
        else:
            self.log_file = self.log_dir / f"{experiment_name}.txt"

        self.original_stdout = None
        self.tee = None

    def start(self):
        self.original_stdout = sys.stdout
        self.tee = Tee(self.log_file, mode='w')
        sys.stdout = self.tee
        return self

    def stop(self):
        if self.tee:
            self.tee.close()
        if self.original_stdout:
            sys.stdout = self.original_stdout

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def log(self, message):
        print(message)

    def log_dict(self, data, title=None):
        if title:
            self.log(f"\n=== {title} ===")
        for key, value in data.items():
            self.log(f"  {key}: {value}")

    def log_section(self, title, width=50):
        self.log(f"\n{'='*width}")
        self.log(f"{title}")
        self.log(f"{'='*width}")

    def log_metrics(self, metrics, step=None):
        if step is not None:
            self.log(f"\n[Step {step}]")
        else:
            self.log("\nMetrics:")
        for key, val in metrics.items():
            if isinstance(val, float):
                self.log(f"  {key}: {val:.6f}")
            else:
                self.log(f"  {key}: {val}")
