# random_video_generator/main.py
from __future__ import annotations
import argparse
import logging
import sys
from .utils import setup_logging, VERSION
from .config import load_config
from .video_composer import assemble_and_write

def main():
    parser = argparse.ArgumentParser(description="Random vertical video generator (modular)")
    parser.add_argument("--config", "-c", required=True, help="Path to config.json")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--log", default="info", help="Set log level: debug, info, warning, error")
    args = parser.parse_args()

    if args.version:
        print(f"random_video_generator version {VERSION}")
        return

    setup_logging(args.log)

    try:
        cfg = load_config(args.config)
        assemble_and_write(cfg)
    except Exception as e:
        logging.error("ERROR: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
