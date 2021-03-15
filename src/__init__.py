#!/usr/bin/env python3

import logging
import logging.config
from pathlib import Path

LOG_CONFIG = Path(__file__).parent.parent.joinpath("configs", "logging.cfg")
logging.config.fileConfig(LOG_CONFIG.as_posix())
