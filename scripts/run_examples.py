#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs examples, failing if they have errors using jupyter-nbconvert
"""

import os
import sys
from subprocess import check_call, DEVNULL
import logging

log = logging.getLogger(__name__)


def run_notebook(path: str) -> None:
    log.info("Running notebook: %s", path)
    check_call(["jupyter-nbconvert", "--to", "notebook", "--execute", "--stdout", path], stdout=DEVNULL)


def run_examples() -> None:
    log.info("Running example notebooks")
    notebooks = ["examples/Bias_metrics_usage.ipynb"]
    for notebook in notebooks:
        run_notebook(notebook)


def script_name() -> str:
    """:returns: script name with leading paths removed"""
    return os.path.split(sys.argv[0])[1]


def config_logging():
    import time

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format="{}: %(asctime)sZ %(levelname)s %(message)s".format(script_name()))
    logging.Formatter.converter = time.gmtime


def main():
    config_logging()
    run_examples()
    return 0


if __name__ == "__main__":
    sys.exit(main())
