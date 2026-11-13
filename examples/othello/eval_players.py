"""The players of evaluation games (`evaluate.py`), each from its `kind:key=value,...`:

    net:net=8x128-128s-8p-w1125-274m,visits=1000,engine=1,backend=native
    net:net=4x64-...,device=gpu-batched,visits=1000
    net:checkpoint=runs/othello/199.eqx,spec=resnet:8x128,clock=10
    edax:level=24
    egaroucid:clock=10,threads=8

Every player has a budget: a network `visits` a move or a `clock`, an engine a `level` or a
`clock`. A clock is a contest's, seconds for the whole game ("10"), or GGS's ("01:00//02:00").

A network is one of the league store's, by `net`, or any checkpoint, by `checkpoint` and its
`spec` (`net` then just names it). It plays on a `device`:

- `cpu` (the default): `cpu_search.py`'s search, its leaves evaluated by `backend` (bf16, f32, ort,
  native, int8-VARIANT), with `engine=1` by a thread a core (see `clock_match.net_player`)
- `gpu`: a tree on the host, leaves evaluated on the GPU a batch at a time (`gpu_play.py`)
- `gpu-batched`: core's MCTS, many games at once on the GPU, as the Elo league plays: fixed
  visits only, against engines at a level or other such networks

Everything that decides how a player plays is in its config, defaults filled in, and its id is the
config's digest (`core.evals.store.player_id`): a network's checkpoint by its sha256, and a
quantized model by its file's; an engine's build by its files' (`engines.build_digest`). Where
the files are isn't part of it, nor what it runs on: that's each game's host.
"""

import math
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engines import (
    CACHE_DIR,
    CLOCK_ONLY,
    ENGINES,
    RUNTIME,
    build_digest,
    default_dir,
    file_sha256,
)
from ggs import Clock

from core.evals.store import player_id

LEAGUE_STORE = CACHE_DIR / "league" / "league.db"
QUANTIZED_DIR = CACHE_DIR / "quantized"
DEVICES = ("cpu", "gpu", "gpu-batched")

# a network's options on each device: (type, default); None for a default of the machine's cores.
# They're all in its config, so a changed default here is a new player, never the same one
# playing differently
SEARCH = {
    "virtual_loss": (str, "loss"),
    "weight": (float, 1.0),
    "c_init": (float, 1.25),
    "max_collisions": (int, None),
    "random_symmetry": (int, 0),
    "reuse": (int, 1),
    "taper": (int, 0),
}
OPTIONS: dict[str, dict[str, tuple[type, Any]]] = {
    "cpu": {
        "engine": (int, 0),
        "backend": (str, "bf16"),
        "batch": (int, 1),
        **SEARCH,
        "threads": (int, None),
    },
    "gpu": {
        "batch": (int, 256),
        "inflight": (int, 2),
        "max_call": (int, 1024),
        "calls": (int, 2),
        "dtype": (str, "bfloat16"),
        **SEARCH,
        "ponder": (int, 0),
        "threads": (int, None),
    },
    # a tree of 4 x visits nodes, no root noise, temperature 0 (`elo_league.Othello`)
    "gpu-batched": {"tree_factor": (int, 4), "dtype": (str, "bfloat16")},
}
# a network's endgame solve, with `solve=N` (see `cpu_match.Solving`)
SOLVE = {"solve_exact": (int, 0), "solve_share": (float, 0.25)}
# a CPU network's expected seconds for a solve at 24 empties on 8 threads: Edax's (`cpu_match.Solving`)
SOLVE_COST = 0.19
# what names a network and where it is: not options of its search
NETWORK_KEYS = ("net", "checkpoint", "spec", "device")


def parse_clock(text: str) -> Clock:
    """A contest's clock of so many seconds ("10"), or GGS's ("01:00//02:00")."""
    return Clock.parse(text) if "/" in text else Clock.contest(float(text))


def clock_config(clock: Clock) -> dict[str, float | None]:
    """A clock as JSON: `ext` None for a contest's (played out after a timeout)."""
    return {
        "ini": clock.ini,
        "inc": clock.inc,
        "ext": None if math.isinf(clock.ext) else clock.ext,
    }


def clock_of(config: dict[str, Any]) -> Clock:
    ext = config["ext"]
    return Clock(config["ini"], config["inc"], math.inf if ext is None else ext)


def clock_label(clock: dict[str, float | None]) -> str:
    if clock["ext"] is None:
        return f"{clock['ini']:g}s"
    return f"{clock['ini']:g}/{clock['inc']:g}/{clock['ext']:g}s"


@dataclass(frozen=True)
class PlayerSpec:
    """A player resolved: who it is and what builds it.

    Attributes:
        config: everything that decides how it plays; its id is this's digest
        runtime: what builds it on a host (`maker`): its options as the players take them, and
            where its network is, from `CACHE_DIR`
        label: a name for people
        files: what a host needs of `CACHE_DIR` to play it
    """

    config: dict[str, Any]
    runtime: dict[str, Any]
    label: str
    files: tuple[str, ...]

    @property
    def id(self) -> str:
        return player_id(self.config)

    @property
    def device(self) -> str | None:
        """A network's device; None for an engine."""
        return self.config.get("device")

    @property
    def clock(self) -> Clock | None:
        """Its clock, if it plays on one; None at a fixed budget."""
        budget = self.config["budget"]
        return clock_of(budget["clock"]) if "clock" in budget else None


def _typed(name: str, value: Any, kind: type) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and kind is not str:
        value = value.strip()
    try:
        return kind(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}={value!r}: not a {kind.__name__}") from error


def _budget(options: dict[str, Any], fixed_key: str) -> dict[str, Any]:
    fixed, clock = options.pop(fixed_key, None), options.pop("clock", None)
    if (fixed is None) == (clock is None):
        raise ValueError(f"give one budget: {fixed_key}=N or clock=SECONDS")
    if fixed is not None:
        return {fixed_key: _typed(fixed_key, fixed, int)}
    return {"clock": clock_config(parse_clock(str(clock)))}


def _relative(path: Path) -> str | None:
    """`path` from `CACHE_DIR`, if it's under it: only those go to cloud hosts."""
    try:
        return str(path.resolve().relative_to(CACHE_DIR.resolve()))
    except ValueError:
        return None


def league_network(name: str, store: Path = LEAGUE_STORE) -> tuple[Path, str, str]:
    """The league store's network `name`: (checkpoint, net spec, sha256)."""
    if not store.exists():
        raise ValueError(f"no league store at {store}, to find {name}: use checkpoint=")
    db = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        row = db.execute(
            "SELECT path, net, sha256 FROM networks WHERE name = ?", (name,)
        ).fetchone()
    finally:
        db.close()
    if row is None:
        raise ValueError(f"no network {name} in the league's store")
    return Path(row[0]), row[1], row[2]


def model_files(name: str, backend: str) -> list[Path]:
    """The quantized model files a `backend` plays the network `name` with (see `quantize.py`,
    `cpu_net.py`): none for one made from the checkpoint as it plays."""
    if backend == "native":
        return [QUANTIZED_DIR / f"{name}-native.npz"]
    if backend.startswith("int8-"):
        variant = backend.split("-")[1]
        xml = QUANTIZED_DIR / f"{name}-{variant}.xml"
        return [xml, xml.with_suffix(".bin")]
    return []


def resolve_net(
    options: dict[str, Any], cores: int, store: Path = LEAGUE_STORE
) -> PlayerSpec:
    options = dict(options)
    device = options.pop("device", "cpu")
    if device not in DEVICES:
        raise ValueError(f"device={device}: one of {', '.join(DEVICES)}")
    checkpoint_option = options.pop("checkpoint", None)
    spec_option = options.pop("spec", None)
    name = options.pop("net", None)
    if checkpoint_option is not None:
        if spec_option is None:
            raise ValueError("checkpoint= needs its network's spec=")
        checkpoint = Path(checkpoint_option).expanduser()
        net, sha = spec_option, file_sha256(checkpoint)
        name = name or f"{checkpoint.parent.name}-{checkpoint.stem}"
    elif name is not None:
        checkpoint, net, sha = league_network(name, store)
    else:
        raise ValueError("a network needs net=NAME (the league's) or checkpoint=PATH")
    if device == "gpu-batched":
        budget = {"visits": _typed("visits", options.pop("visits", None), int)}
        if budget["visits"] is None or "clock" in options:
            raise ValueError("device=gpu-batched plays at fixed visits only")
    else:
        budget = _budget(options, "visits")
    known = OPTIONS[device]
    search: dict[str, Any] = {}
    for key, (kind, default) in known.items():
        value = options.pop(key, default)
        if value is None and key == "threads":
            value = cores
        search[key] = _typed(key, value, kind)
    if device != "gpu-batched" and "solve" in options:
        search["solve"] = _typed("solve", options.pop("solve"), int)
        for key, (kind, default) in SOLVE.items():
            search[key] = _typed(key, options.pop(key, default), kind)
        if device == "cpu":
            # what it plans its solve on a clock by (see `cpu_match.Solving`)
            search["solve_cost"] = _typed(
                "solve_cost", options.pop("solve_cost", SOLVE_COST), float
            )
    if options:
        raise ValueError(f"unknown options for a {device} network: {sorted(options)}")
    config: dict[str, Any] = {
        "kind": "net",
        "network": {"sha256": sha, "net": net},
        "device": device,
        "budget": budget,
        "search": search,
    }
    files = [checkpoint]
    if device == "cpu":
        models = model_files(name, search["backend"])
        for model in models:
            if not model.exists():
                raise ValueError(f"no {model}: build it (see quantize.py, cpu_net.py)")
        if models:
            config["model_sha256"] = {m.name: file_sha256(m) for m in models}
        files += models
    budget_label = (
        f"{budget['visits']}v" if "visits" in budget else clock_label(budget["clock"])
    )
    where = device + (f"/{search['backend']}" if device == "cpu" else "")
    solve = f" solve{search['solve']}" if "solve" in search else ""
    relative = _relative(checkpoint)
    runtime = {
        "player": "net",
        "network": {
            "name": name,
            "checkpoint": relative if relative is not None else str(checkpoint),
            "net": net,
        },
        "options": {
            **{k: v for k, v in search.items() if v is not None},
            "device": device,
            **({"visits": budget["visits"]} if "visits" in budget else {}),
        },
    }
    return PlayerSpec(
        config,
        runtime,
        f"{name} {budget_label} {where}{solve}",
        tuple(sorted({_relative(f) or str(f) for f in files})),
    )


def resolve_engine(name: str, options: dict[str, Any], cores: int) -> PlayerSpec:
    options = dict(options)
    engine_dir = Path(options.pop("dir")).expanduser() if "dir" in options else None
    if name in CLOCK_ONLY and "level" in options:
        raise ValueError(f"{name} plays only on a clock")
    budget = _budget(options, "level")
    threads = _typed("threads", options.pop("threads", cores), int)
    args = options.pop("args", [])
    args = args.split() if isinstance(args, str) else [str(a) for a in args]
    if options:
        raise ValueError(f"unknown options for {name}: {sorted(options)}")
    root = engine_dir if engine_dir is not None else default_dir(name)
    config = {
        "kind": "engine",
        "engine": name,
        "build": build_digest(name, root),
        "threads": threads,
        "args": args,
        "budget": budget,
    }
    budget_label = (
        f"L{budget['level']}" if "level" in budget else clock_label(budget["clock"])
    )
    relative = _relative(root)
    runtime = {
        "player": name,
        "dir": relative if relative is not None else str(root),
        "options": {
            "threads": threads,
            "args": " ".join(args),
            **({"level": budget["level"]} if "level" in budget else {}),
        },
    }
    extra = f" {' '.join(args)}" if args else ""
    files = tuple(
        f"{relative}/{part}" if relative is not None else str(root / part)
        for part in RUNTIME[name]
    )
    return PlayerSpec(
        config, runtime, f"{name} {budget_label} t{threads}{extra}", files
    )


def parse(spec: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """A player's kind and options, from `kind:key=value,...` or a table of a config file with
    its `kind`."""
    if isinstance(spec, dict):
        options = dict(spec)
        kind = options.pop("kind", None)
        if kind is None:
            raise ValueError(f"a player's table needs its kind: {spec}")
        return str(kind), options
    kind, _, rest = spec.partition(":")
    options = {}
    for item in rest.split(","):
        if item:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"{item!r} in {spec!r}: options are key=value")
            options[key.strip()] = value
    return kind.strip(), options


def resolve(
    spec: str | dict[str, Any],
    cores: int,
    batched: bool = False,
    store: Path = LEAGUE_STORE,
) -> PlayerSpec:
    """The player of `spec`, its options defaulting to `cores` threads (an engine's to one in a
    `batched` match, a process a game)."""
    kind, options = parse(spec)
    if kind == "net":
        return resolve_net(options, cores, store)
    if kind not in ENGINES and kind not in CLOCK_ONLY:
        raise ValueError(f"unknown player {kind!r}: net, or one of {sorted(ENGINES)}")
    return resolve_engine(kind, options, 1 if batched else cores)


def register_network(runtime: dict[str, Any]) -> None:
    """Makes a runtime's network known by name to the players, which find networks by name, on
    a host with no league store."""
    from bench_cpu_backends import NETWORKS

    network = runtime["network"]
    path = Path(network["checkpoint"])
    path = path if path.is_absolute() else CACHE_DIR / path
    NETWORKS[network["name"]] = (str(path), network["net"])


def maker(runtime: dict[str, Any], clock: Clock) -> Callable[[], Any]:
    """What builds a player (`clock_match.Player`) from its `runtime` for each game; engines
    plan for `clock`."""
    from clock_match import player_maker

    options = {k: str(v) for k, v in runtime["options"].items()}
    if runtime["player"] == "net":
        register_network(runtime)
        options["net"] = runtime["network"]["name"]
        return player_maker("net", options, clock)
    path = Path(runtime["dir"])
    options["dir"] = str(path if path.is_absolute() else CACHE_DIR / path)
    if not options["args"]:
        del options["args"]
    return player_maker(runtime["player"], options, clock)


def available_cores() -> int:
    return len(os.sched_getaffinity(0))
