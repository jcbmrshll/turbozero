from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from core.common import two_player_game
from core.evaluators.random_evaluator import RandomEvaluator

MAX_STEPS = 10


def test_random_evaluator_plays_legal_moves_uniformly(ttt):
    evaluator = RandomEvaluator()
    state, meta = ttt.play([0, 4])
    eval_state = evaluator.init()
    keys = jax.random.split(jax.random.PRNGKey(0), 7000)

    actions = jax.jit(
        jax.vmap(
            lambda k: (
                evaluator.evaluate(k, eval_state, state, root_metadata=meta).action
            )
        )
    )(keys)

    counts = np.bincount(np.asarray(actions), minlength=ttt.num_actions)
    assert counts[0] == counts[4] == 0
    # 7 legal squares, 1000 picks each expected
    np.testing.assert_allclose(counts[np.asarray(meta.action_mask)], 1000, rtol=0.1)


def test_two_player_game_records_each_players_tree_of_each_position(ttt, make_search):
    # two searches of different sizes, so their records differ in shape too
    big = make_search().evaluator
    small = make_search(num_iterations=16, max_nodes=40).evaluator
    game = partial(
        two_player_game,
        evaluator_1=big,
        evaluator_2=small,
        params_1=None,
        params_2=None,
        env_step_fn=ttt.step_fn,
        env_init_fn=ttt.init_fn,
        max_steps=MAX_STEPS,
        p1_first=True,
    )
    _, frames, _ = jax.device_get(
        jax.jit(partial(game, record_search=True))(jax.random.PRNGKey(0))
    )
    # (tracing is enough to see that a game without records keeps none)
    _, unrecorded, _ = jax.eval_shape(game, jax.random.PRNGKey(0))
    assert unrecorded.p1_search is None and unrecorded.p2_search is None

    num_moves = int(np.argmax(frames.completed))
    board = frames.env_state._x.board
    # the cell each move filled
    played = [
        int(np.flatnonzero(board[t + 1] != board[t])[0]) for t in range(num_moves)
    ]
    records = (frames.p1_search, frames.p2_search)
    at = lambda records, t: jax.tree.map(lambda x: x[t], records)
    for t in range(num_moves):
        mover, other = (0, 1) if t % 2 == 0 else (1, 0)
        searched = at(records[mover], t + 1)
        # the mover's record is its search of the position, at least a search's worth
        assert searched.n[0] >= (big if mover == 0 else small).num_iterations
        carried = at(records[other], t + 1)
        if t == 0:
            # the other player hasn't searched yet
            assert carried.num_nodes == 0
            continue
        # the other's record is the subtree it kept from its search of the position before,
        # under the move it played
        before = at(records[other], t)
        child = np.flatnonzero(
            (before.parents == 0) & (before.actions == played[t - 1])
        )[0]
        assert carried.n[0] == before.n[child]
        assert carried.q[0] == before.q[child]


def test_two_player_game_has_every_frame_up_to_the_final_state(
    fixed_length_env, scripted
):
    max_steps = 4
    env = fixed_length_env(length=max_steps)

    _, frames, _ = jax.jit(
        partial(
            two_player_game,
            evaluator_1=scripted.first_legal,
            evaluator_2=scripted.first_legal,
            params_1={"w": jnp.ones(3)},
            params_2={"w": jnp.ones(3)},
            env_step_fn=env.step_fn,
            env_init_fn=env.init_fn,
            max_steps=max_steps,
        )
    )(jax.random.PRNGKey(0))

    assert frames.env_state.step.tolist() == list(range(max_steps + 1))
    assert frames.completed[-1]
