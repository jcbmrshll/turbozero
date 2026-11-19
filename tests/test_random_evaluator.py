import jax
import jax.numpy as jnp
import numpy as np

from turbozero.evaluators.random_evaluator import RandomEvaluator
from turbozero.types import StepMetadata


def test_plays_legal_actions_uniformly():
    evaluator = RandomEvaluator()
    mask = jnp.array([False, True, False, True, True])
    metadata = StepMetadata(
        rewards=jnp.zeros(2),
        action_mask=mask,
        terminated=jnp.array(False),
        cur_player_id=jnp.array(0),
        step=jnp.array(0),
    )

    def evaluate(key):
        return evaluator.evaluate(
            key, evaluator.init(None), None, metadata, None, lambda s, a: (s, metadata)
        )

    keys = jax.random.split(jax.random.PRNGKey(0), 3000)
    output = jax.jit(jax.vmap(evaluate))(keys)

    np.testing.assert_allclose(output.policy_weights[0], [0, 1 / 3, 0, 1 / 3, 1 / 3])
    counts = np.bincount(output.action, minlength=5)
    assert counts[[0, 2]].sum() == 0
    # about 1000 each: 5 standard deviations is about 130
    assert np.all(np.abs(counts[[1, 3, 4]] - 1000) < 130)
