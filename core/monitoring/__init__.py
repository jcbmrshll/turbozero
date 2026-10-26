"""Logging training runs to a seaglass server (https://github.com/jcbmrshll/seaglass), drawn
with turbozero's views.

    monitor = Monitor(project="othello")  # the server is $SEAGLASS_URL, else localhost:8008
    Trainer(..., monitor=monitor)

`Monitor` is seaglass's, with turbozero's bundle (`VIEWS`), which the server keeps with each
run: its page plays the run's test games (see `episodes.pgx_two_player_episode`) and explores
the searches behind their moves. Start a server with `seaglass serve --dir runs`.
"""

import os
from pathlib import Path
from typing import Any

import seaglass
from seaglass.client import DEFAULT_URL as SEAGLASS_URL

# the bundle runs are drawn with: layout.json and the modules it names
VIEWS = Path(__file__).with_name("views")
# the server runs log to when given none
DEFAULT_URL = os.environ.get("SEAGLASS_URL") or SEAGLASS_URL


class Monitor(seaglass.Monitor):
    """A `seaglass.Monitor` whose runs are drawn with turbozero's views (`VIEWS`)."""

    def __init__(
        self,
        url: str | None = None,
        project: str = "default",
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(url, project=project, name=name, views=VIEWS, **kwargs)


__all__ = ["DEFAULT_URL", "VIEWS", "Monitor"]
