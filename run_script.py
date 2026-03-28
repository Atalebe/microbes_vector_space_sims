#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a pipeline script with a YAML config.")
    parser.add_argument("script", help="Path to script, e.g. src/01_audit_dataset.py")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed through")
    args = parser.parse_args()

    script = Path(args.script)
    config = Path(args.config)
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    cmd = [sys.executable, str(script), str(config)] + list(args.extra)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
