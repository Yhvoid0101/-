"""Compatibility namespace for the flattened unified runtime tree.

The original runtime was imported as ``sandbox_trading.<module>`` from its
parent Hermes directory. The unified root keeps one physical copy of each
module and exposes that namespace without duplicating source files.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]
