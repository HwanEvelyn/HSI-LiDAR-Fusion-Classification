from __future__ import annotations

from pathlib import Path


class SimpleLogger:
    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        print(message)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")
