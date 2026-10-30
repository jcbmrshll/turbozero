"""Imports an example's modules for its tests. Examples' scripts import their sibling modules as
top-level ones, so two examples' `game.py` would otherwise be one module in a process."""

import importlib
import sys
from pathlib import Path
from types import ModuleType

EXAMPLES = Path(__file__).parent.parent / "examples"


def load_example(example: str, *names: str) -> tuple[ModuleType, ...]:
    """The named modules of `examples/<example>/`, importing each other as its scripts do. Leaves
    sys.path and sys.modules as they were, so collection order doesn't matter."""
    directory = EXAMPLES / example
    siblings = {path.stem for path in directory.glob("*.py")}
    saved = {name: sys.modules.pop(name) for name in siblings if name in sys.modules}
    sys.path.insert(0, str(directory))
    try:
        return tuple(importlib.import_module(name) for name in names)
    finally:
        sys.path.remove(str(directory))
        for name in siblings:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
