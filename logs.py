#!/usr/bin/env python3
"""
Logging configuration for stormsurge logs
Defines structured logging for:
    * error messages     -- qualname error
    * Status messages   -- qualname status
"""

import logging
import structlog
from structlog.processors import JSONRenderer


COMMON_PROCESSORS = [
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="ISO"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    JSONRenderer(sort_keys=True),
]


def get_wrapped_logger(logger_name: str = "root", **kwargs):
    """ Returns a struct log equivalent for the named logger """
    return structlog.wrap_logger(
        logging.getLogger(logger_name), COMMON_PROCESSORS, **kwargs
    )


class FormatJSONL(logging.Formatter):
    """Prevents printing of the stack trace to enable JSON lines output."""

    def formatException(self, ei):
        return


ERROR_LOGGER = get_wrapped_logger("error")
STATUS_LOGGER = get_wrapped_logger("status")
