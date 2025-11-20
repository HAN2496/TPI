"""Experiment logging utilities"""
import sys
from pathlib import Path
from datetime import datetime


class Tee:
    """Write to both file and console (like Unix tee command)"""
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
    """Logger that captures stdout to both console and file

    Usage:
        # Method 1: Context manager (recommended)
        with ExperimentLogger('artifacts/강신길', 'analysis'):
            print('=== Analysis Results ===')
            print(f'AUROC: 0.85')

        # Method 2: Manual start/stop
        logger = ExperimentLogger('artifacts/강신길', 'training')
        logger.start()
        print('Training started...')
        logger.stop()

        # Method 3: With custom methods
        logger = ExperimentLogger('artifacts/강신길', 'experiment')
        logger.start()
        logger.log('Starting experiment...')
        logger.log_section('Model Training')
        logger.log_dict({'auroc': 0.85, 'n_layers': 4}, title='Results')
        logger.stop()
    """

    def __init__(self, log_dir, experiment_name, add_timestamp=True):
        """Initialize logger

        Args:
            log_dir: Directory to save log file
            experiment_name: Name of the experiment
            add_timestamp: Whether to add timestamp to filename
        """
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
        """Start capturing stdout"""
        self.original_stdout = sys.stdout
        self.tee = Tee(self.log_file, mode='w')
        sys.stdout = self.tee
        return self

    def stop(self):
        """Stop capturing stdout and restore original stdout"""
        if self.tee:
            self.tee.close()
        if self.original_stdout:
            sys.stdout = self.original_stdout

    def __enter__(self):
        """Context manager entry"""
        return self.start()

    def __exit__(self, *args):
        """Context manager exit"""
        self.stop()

    def log(self, message):
        """Log a message (works whether capturing or not)"""
        print(message)

    def log_dict(self, data, title=None):
        """Log dictionary in readable format

        Args:
            data: Dictionary to log
            title: Optional title for the section
        """
        if title:
            self.log(f"\n=== {title} ===")
        for key, value in data.items():
            self.log(f"  {key}: {value}")

    def log_section(self, title, width=50):
        """Log a section separator

        Args:
            title: Section title
            width: Width of separator line
        """
        self.log(f"\n{'='*width}")
        self.log(f"{title}")
        self.log(f"{'='*width}")
