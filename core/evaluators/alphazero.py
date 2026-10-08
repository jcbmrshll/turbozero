from typing import Any

import jax
import jax.numpy as jnp

from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.state import MCTSTree
from core.types import StepMetadata


class _AlphaZero(MCTS):
    """AlphaZero-specific logic for MCTS.

    Extends MCTS using the `AlphaZero` class, this class serves as a mixin to add AlphaZero-specific logic.
    """

    def __init__(
        self, dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25, **kwargs
    ):
        """Initializes an AlphaZero evaluator.

        Args:
            dirichlet_alpha: magnitude of Dirichlet noise.
            dirichlet_epsilon: proportion of root policy composed of Dirichlet noise.
            **kwargs: see `MCTS` class for additional configuration
        """
        super().__init__(**kwargs)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def get_config(self) -> dict:
        """Returns the configuration of the AlphaZero evaluator. Used for logging."""
        return {
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            **super().get_config(),
        }

    def update_root(
        self,
        key: jax.Array,
        tree: MCTSTree,
        root_embedding: Any,
        params: Any,
        root_metadata: StepMetadata,
        **kwargs,
    ) -> MCTSTree:  # pylint: disable=unused-argument
        """Populates the root node of the search tree. Adds Dirichlet noise to the root policy.

        Args:
            key: rng
            tree: The search tree.
            root_embedding: root environment state.
            params: nn parameters.
            root_metadata: metadata of the root environment state

        Returns:
            MCTSTree: The updated search tree.
        """
        # evaluate the root state
        root_key, dir_key = jax.random.split(key, 2)
        root_policy_logits, root_value = self.eval_fn(root_embedding, params, root_key)
        mask = root_metadata.action_mask
        min_logit = jnp.finfo(root_policy_logits.dtype).min
        root_policy = jax.nn.softmax(jnp.where(mask, root_policy_logits, min_logit))

        # Dirichlet noise over the legal actions only, as in AlphaZero (its root has a child per legal move,
        # and each gets a share of the noise): the softmax of independent log-Gamma(alpha) draws is a
        # Dirichlet(alpha) sample, computed in log space so small alphas don't underflow
        log_gamma = jax.random.loggamma(dir_key, self.dirichlet_alpha, mask.shape)
        dirichlet_noise = jax.nn.softmax(jnp.where(mask, log_gamma, min_logit))
        # both are distributions over the legal actions, so their mix is too
        noisy_policy = ((1 - self.dirichlet_epsilon) * root_policy) + (
            self.dirichlet_epsilon * dirichlet_noise
        )

        # update the root node
        root_node = tree.data_at(tree.ROOT_INDEX)
        root_node = self.update_root_node(
            root_node, noisy_policy, root_value, root_embedding
        )
        return tree.set_root(root_node)


class AlphaZero(MCTS):
    """AlphaZero: Monte Carlo Tree Search + Neural Network Leaf Evaluation.

    https://arxiv.org/abs/1712.01815

    Most of the work is actually done in the `MCTS` class, which AlphaZero extends.
    This class can take an arbitrary MCTS backend, which is why we use a separate class `_AlphaZero`
    to handle the AlphaZero-specific logic, then combine them here.
    """

    def __new__(cls, base_type: type = MCTS):
        """Creates a new AlphaZero class that extends the given MCTS class."""
        assert issubclass(base_type, MCTS)
        cls_type = type("AlphaZero", (_AlphaZero, base_type), {})
        cls_type.__name__ = f"AlphaZero({base_type.__name__})"
        return cls_type
