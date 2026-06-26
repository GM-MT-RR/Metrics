"""Rich-formatted logger with an attachable per-experiment file handler.

Same shape as ``Standard/standard/core/logger.py`` (a separate logger name so
the two packages don't fight over handlers).
"""
import logging

try:
    from rich.logging import RichHandler
    from rich.console import Console
    _handler = RichHandler(
        level="INFO",
        console=Console(),
        show_time=False,
        show_path=False,
        markup=True,
    )
    _handler.setFormatter(logging.Formatter("%(message)s"))
except ImportError:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))


class Logger:
    _logger = logging.getLogger("de_eval")
    _logger.handlers.clear()
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False

    @staticmethod
    def info(msg: object) -> None:
        Logger._logger.info(msg)

    @staticmethod
    def warn(msg: object) -> None:
        Logger._logger.warning(msg)

    @staticmethod
    def error(msg: object) -> None:
        Logger._logger.error(msg)

    @staticmethod
    def attach_file(path: str) -> None:
        for h in list(Logger._logger.handlers):
            if isinstance(h, logging.FileHandler):
                Logger._logger.removeHandler(h)
                h.close()
        fh = logging.FileHandler(path, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        Logger._logger.addHandler(fh)
