from __future__ import annotations
import argparse
import asyncio
import logging
import sys
from .config import load_config
from .runner import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="detstream")
    parser.add_argument("--config", required=True, help="path to a YAML/JSON feed config")
    args = parser.parse_args()

    # Force line buffering so log lines appear as they happen
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = load_config(args.config)
    try:
        asyncio.run(run(app))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
