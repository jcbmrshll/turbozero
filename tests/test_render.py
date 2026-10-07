"""Tests for rendering pgx games to .gifs."""
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import pytest
from PIL import Image, ImageSequence

try:
    # cairosvg loads the cairo system library on import (OSError when it is missing)
    import cairosvg
    from core.testing.utils import render_pgx_2p
except (ImportError, OSError) as e:
    pytest.skip(f"cairo is unavailable: {e}", allow_module_level=True)

from core.common import two_player_game
from core.testing.two_player_tester import TwoPlayerTester, TwoPlayerTestState

MAX_STEPS = 12
DURATION = 900


def frame_list(frames):
    """Splits stacked frames into a list of frames, as `BaseTester.run` does."""
    return [jax.device_get(jax.tree.map(lambda x: x[i], frames)) for i in range(MAX_STEPS)]


def num_rendered(frames):
    """Number of frames rendered: up to and including the first completed frame."""
    completed = [bool(f.completed) for f in frames]
    return completed.index(True) + 1 if any(completed) else len(frames)


def test_render_writes_only_the_gif(ttt, scripted, tmp_path):
    # unrelated files the render must not touch
    user_files = {"photo.png": b"not really a png", "diagram.svg": b"<svg/>", "000.png": b"x", "999.svg": b"y"}
    for name, content in user_files.items():
        (tmp_path / name).write_bytes(content)

    _, frames, p_ids = two_player_game(jax.random.PRNGKey(3), scripted.first_legal, scripted.first_legal, None, None,
                                       ttt.step_fn, ttt.init_fn, max_steps=MAX_STEPS)
    frames = frame_list(frames)
    gif_path = render_pgx_2p(frames, p_ids, "ttt", str(tmp_path), duration=DURATION)

    assert gif_path == str(tmp_path / "ttt.gif")
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([*user_files, "ttt.gif"])
    for name, content in user_files.items():
        assert (tmp_path / name).read_bytes() == content

    # the final frame is repeated twice more, which pillow merges into one longer frame
    n = num_rendered(frames)
    assert 1 < n < MAX_STEPS
    with Image.open(gif_path) as gif:
        assert len(list(ImageSequence.Iterator(gif))) == n
        gif.seek(n - 1)
        assert gif.info["duration"] == 3 * DURATION


def test_tester_render_creates_missing_render_dir(ttt, scripted, tmp_path):
    render_dir = tmp_path / "does" / "not exist"
    tester = TwoPlayerTester(num_episodes=2, render_fn=render_pgx_2p, render_dir=str(render_dir), name="ttt")
    params = jax.tree.map(lambda x: jnp.stack([x] * 2), {"w": jnp.zeros(3)})

    _, _, gif_path = tester.run(key=jax.random.PRNGKey(0), epoch_num=0, max_steps=MAX_STEPS, num_devices=2,
                                env_step_fn=ttt.step_fn, env_init_fn=ttt.init_fn, evaluator=scripted.first_legal,
                                state=TwoPlayerTestState(best_params=params), params=params)

    assert gif_path == str(render_dir / "ttt_0.gif")
    assert [p.name for p in render_dir.iterdir()] == ["ttt_0.gif"]


class ViewBoxState:
    """Stand-in pgx state whose SVG has a viewBox and no width/height attributes."""
    current_player = 0

    def to_svg(self, color_theme=None):  # pylint: disable=unused-argument
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
                '<rect width="200" height="100" fill="green"/></svg>')


def test_render_svg_with_viewbox(tmp_path, monkeypatch):
    frame = jax.device_get(dict(p1_value_estimate=jnp.array(0.5), p2_value_estimate=jnp.array(-0.5),
                                completed=jnp.array(True), outcomes=jnp.array([1.0, -1.0])))
    frame = type("Frame", (), dict(env_state=ViewBoxState(), **frame))()

    # capture the SVG that gets rasterized
    rendered = []
    svg2png = cairosvg.svg2png
    monkeypatch.setattr(cairosvg, "svg2png",
                        lambda bytestring, **kwargs: rendered.append(bytestring) or svg2png(bytestring=bytestring, **kwargs))

    gif_path = render_pgx_2p([frame], [0, 1], "viewbox", str(tmp_path))

    root = ET.fromstring(rendered[0])
    assert root.attrib["viewBox"] == "0.0 0.0 200.0 120.0"
    rect = root.findall("{http://www.w3.org/2000/svg}rect")[-1]
    assert (rect.attrib["width"], rect.attrib["y"]) == ("200.0", "100.0")
    texts = root.findall("{http://www.w3.org/2000/svg}text")
    assert [t.attrib["x"] for t in texts] == ["2.0", "2.0"]
    assert texts[0].text is not None and texts[0].text.startswith("[W] Trained Agent (Black): +0.5000")
    with Image.open(gif_path) as gif:
        assert len(list(ImageSequence.Iterator(gif))) == 1
