"""CLI subcommand registration."""
from . import run as _run
from . import batch as _batch
from . import list_components as _list


def register_parsers(subparser) -> None:
    _run.add_parser(subparser)
    _batch.add_parser(subparser)
    _list.add_parser(subparser)


__all__ = ["register_parsers"]
