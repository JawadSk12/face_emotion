"""CSV event log writer."""

import csv
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

LOG_PATH = Path(CONFIG["paths"]["logs_csv"])
if not LOG_PATH.is_absolute():
    LOG_PATH = PROJECT_ROOT / LOG_PATH


class EventLogger:
    """Append timestamped person/emotion observations to the configured CSV."""

    def __init__(self, log_path=LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(["timestamp", "person", "emotion"])

    def log(self, person, emotion):
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with self.log_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([timestamp, person, emotion])
