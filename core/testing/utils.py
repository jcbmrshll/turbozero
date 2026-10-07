
import io
import os

import xml.etree.ElementTree as ET
import cairosvg
from PIL import Image

def render_pgx_2p(frames, p_ids, title, frame_dir, p1_label='Black', p2_label='White', duration=900):
    """Really messy render function for rendering frames from a 2-player game from a PGX environment to a .gif.

    Intermediate frames are rendered in memory: the only file written is `{frame_dir}/{title}.gif`.
    `frame_dir` is created if it does not exist.

    Args:
        frames: `GameFrame`s of one episode; rendering stops after the first completed frame
        p_ids: player ids of the trained agent and its opponent
        title: name of the .gif (without extension)
        frame_dir: directory to save the .gif to
        p1_label: label of the player who moves first
        p2_label: label of the player who moves second
        duration: duration of each frame in milliseconds

    Returns:
        str: path to the .gif
    """
    trained_agent_color = p1_label if frames[0].env_state.current_player == p_ids[0] else p2_label
    opponent_color = p2_label if trained_agent_color == p1_label else p1_label
    agent_win = False
    opp_win = False
    draw = False
    images = []
    for frame in frames:
        env_state = frame.env_state
        if frame.completed.item():
            agent_win = frame.outcomes[p_ids[0]] > frame.outcomes[p_ids[1]]
            opp_win = frame.outcomes[p_ids[1]] > frame.outcomes[p_ids[0]]
            draw = frame.outcomes[0] == frame.outcomes[1]

        root = ET.fromstring(env_state.to_svg(color_theme='dark'))

        viewBox = root.attrib.get('viewBox', None)
        if viewBox:
            viewBox = viewBox.split()
            viewBox = [float(v) for v in viewBox]
            original_width = viewBox[2]
            original_height = viewBox[3]
        else:
            original_width = float(root.attrib.get('width', 0))
            original_height = float(root.attrib.get('height', 0))

        new_height = original_height * 1.2
        # Update the viewBox and height attributes
        if viewBox:
    
            viewBox[3] = new_height
            root.attrib['viewBox'] = ' '.join(map(str, viewBox))
        root.attrib['height'] = str(new_height)

        # Create a new text element
        p1_text = ET.Element('ns0:text', x=str(0.01 * original_width), y=str(original_height * 1.05), fill='white', style='font-family: Arial;')
        p2_text = ET.Element('ns0:text', x=str(0.01 * original_width), y=str(original_height * 1.15), fill='white', style='font-family: Arial;')
        emoji = "[W]" if agent_win else "[L]" if opp_win else "[D]" if draw else ""
        agent_text = f"{emoji} Trained Agent ({trained_agent_color}): {'+' if frame.p1_value_estimate > 0 else ''}{frame.p1_value_estimate:.4f}"
        emoji = "[W]" if opp_win else "[L]" if agent_win else "[D]" if draw else ""
        opp_text = f"{emoji} Opponent ({opponent_color}): {'+' if frame.p2_value_estimate > 0 else ''}{frame.p2_value_estimate:.4f}"
        p1_text.text = agent_text
        p2_text.text = opp_text

        new_area = ET.Element('ns0:rect', fill='#1e1e1e', height=str(new_height - original_height), width=str(original_width), x='0', y=str(original_height))
        root.append(new_area)

        root.append(p1_text)
        root.append(p2_text)

        png = io.BytesIO()
        cairosvg.svg2png(bytestring=ET.tostring(root, encoding='utf-8'), write_to=png)
        png.seek(0)
        images.append(Image.open(png))
        if frame.completed.item():
            break

    os.makedirs(frame_dir, exist_ok=True)
    gif_path = os.path.join(frame_dir, f"{title}.gif")
    images[0].save(gif_path, save_all=True, append_images=images[1:] + ([images[-1]] * 2), duration=duration, loop=0)
    return gif_path
