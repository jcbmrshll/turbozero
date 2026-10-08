"""Episodes the monitor server renders, so a training run never draws anything.

A training run packs an episode's raw arrays into an `Episode` (see the packers here,
passed to a tester as its `episode_fn`); the server saves them and renders them on a
background thread with the renderer named by the episode, from RENDERERS.

Renderers run in the server process only, so their imports (jax, pgx, cairosvg) stay
inside them; the server pins jax to the CPU before any of them run.
"""

import io
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from operator import itemgetter
from typing import Any, cast

import numpy as np

from core.monitor.client import Episode
from core.monitor.media import Video

SVG = "http://www.w3.org/2000/svg"
# pgx's svgs use the default namespace; keep it that way when they're rewritten
ET.register_namespace("", SVG)


def pgx_two_player_episode(
    p1_label: str = "Black", p2_label: str = "White", frame_ms: int = 900
) -> Callable[[Any, Any], Episode]:
    """An `episode_fn` for testers of two-player pgx games: packs the first test
    episode for the monitor to draw as a gif (see `render_pgx_two_player`).

    Args:
        p1_label: what to call the player that moves first (e.g. its colour)
        p2_label: what to call the player that moves second
        frame_ms: how long each move is shown for

    Returns:
        Callable: `episode_fn(frames, p_ids) -> Episode`, see `BaseTester`
    """
    import jax

    def episode_fn(frames, p_ids) -> Episode:
        # the env state's arrays, named by their path in the pgx State, e.g. "state._x.board"
        state = {
            "state" + jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree_util.tree_leaves_with_path(frames.env_state)
        }
        return Episode(
            "pgx_two_player",
            env_id=frames.env_state.env_id,
            labels=[p1_label, p2_label],
            frame_ms=frame_ms,
            players=p_ids,
            completed=frames.completed,
            outcomes=frames.outcomes,
            values=np.stack(
                [frames.p1_value_estimate, frames.p2_value_estimate], axis=-1
            ),
            **state,
        )

    return episode_fn


def _pgx_states(data: dict[str, np.ndarray]) -> Any:
    """The pgx State pytree (with a leading time axis) the arrays were packed from."""
    import jax
    import pgx

    env_id = str(data["env_id"])
    template = pgx.make(env_id).init(jax.random.PRNGKey(0))  # type: ignore[arg-type]
    paths, treedef = jax.tree_util.tree_flatten_with_path(template)
    leaves = [data["state" + jax.tree_util.keystr(path)] for path, _ in paths]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _caption(svg: bytes, lines) -> bytes:
    """The svg with a strip below it holding a line of text per player."""
    root = ET.fromstring(svg)
    view_box = root.attrib.get("viewBox")
    if view_box:
        # draw in the viewBox's units, and grow it to fit the strip
        x, y, width, height = (float(v) for v in view_box.split())
        root.attrib["viewBox"] = f"{x} {y} {width} {height * 1.2}"
        if "height" in root.attrib:
            root.attrib["height"] = str(float(root.attrib["height"]) * 1.2)
    else:
        # pgx's own svgs only set a width and height
        width, height = float(root.attrib["width"]), float(root.attrib["height"])
        root.attrib["height"] = str(height * 1.2)
    strip = height * 0.2
    ET.SubElement(
        root,
        f"{{{SVG}}}rect",
        fill="#1e1e1e",
        x="0",
        y=str(height),
        width=str(width),
        height=str(strip),
    )
    for i, line in enumerate(lines):
        text = ET.SubElement(
            root,
            f"{{{SVG}}}text",
            x=str(0.01 * width),
            y=str(height * (1.05 + 0.1 * i)),
            fill="white",
            style="font-family: Arial;",
        )
        text.text = line
    return ET.tostring(root)


def render_pgx_two_player(data: dict[str, np.ndarray]) -> Video:
    """A two-player pgx game as a gif: the board after each move, captioned with the
    trained agent's and its opponent's value estimates, and the result at the end."""
    import cairosvg
    import jax
    from PIL import Image

    states = _pgx_states(data)
    agent, opponent = (int(p) for p in data["players"])
    p1_label, p2_label = (str(label) for label in data["labels"])
    completed = data["completed"].astype(bool)
    # frames after the game ended are padding
    num_frames = int(np.argmax(completed)) + 1 if completed.any() else len(completed)

    agent_label = p1_label if int(states.current_player[0]) == agent else p2_label
    opponent_label = p2_label if agent_label == p1_label else p1_label
    outcomes = data["outcomes"][num_frames - 1]
    agent_tag = opponent_tag = ""
    if completed.any():
        if outcomes[agent] > outcomes[opponent]:
            agent_tag, opponent_tag = "[W] ", "[L] "
        elif outcomes[agent] < outcomes[opponent]:
            agent_tag, opponent_tag = "[L] ", "[W] "
        else:
            agent_tag = opponent_tag = "[D] "

    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(num_frames):
            state = jax.tree.map(itemgetter(i), states)
            state.save_svg(f"{tmp}/frame.svg", color_theme="dark")
            with open(f"{tmp}/frame.svg", "rb") as f:
                svg = f.read()
            # the result is shown once the game is over
            done = bool(completed[i])
            agent_value, opponent_value = data["values"][i]
            svg = _caption(
                svg,
                [
                    f"{agent_tag if done else ''}Trained Agent ({agent_label}): {agent_value:+.4f}",
                    f"{opponent_tag if done else ''}Opponent ({opponent_label}): {opponent_value:+.4f}",
                ],
            )
            # returns the png's bytes, since no write_to is given
            png = cast(bytes, cairosvg.svg2png(bytestring=svg))
            images.append(Image.open(io.BytesIO(png)).convert("RGB"))
    # hold the final position
    return Video(images + [images[-1]] * 2, fps=1000 / int(data["frame_ms"]))


# renderer name (Episode.env) -> function from the episode's arrays to media
RENDERERS: dict[str, Callable[[dict[str, np.ndarray]], Any]] = {
    "pgx_two_player": render_pgx_two_player,
}
