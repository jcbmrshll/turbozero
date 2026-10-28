import jax
import jax.numpy as jnp

from core.evaluators.mcts.mcts import MCTS


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

    def root_noise(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """Independent log-Gamma(alpha) draws, one per action, for `root_prior`'s Dirichlet noise.

        Args:
            key: rng
            shape: (F,)

        Returns:
            jax.Array: the draws
        """
        return jax.random.loggamma(key, self.dirichlet_alpha, shape)

    def root_prior(
        self, noise: jax.Array, policy: jax.Array, action_mask: jax.Array
    ) -> jax.Array:
        """The root's prior: the network's policy, with Dirichlet noise mixed in.

        Args:
            noise: the search's `root_noise`
            policy: the network's policy, 0 for illegal actions
            action_mask: the legal actions

        Returns:
            jax.Array: the prior
        """
        min_logit = jnp.finfo(policy.dtype).min
        # Dirichlet noise over the legal actions only, as in AlphaZero (its root has a child per legal move,
        # and each gets a share of the noise): the softmax of independent log-Gamma(alpha) draws is a
        # Dirichlet(alpha) sample, computed in log space so small alphas don't underflow
        dirichlet_noise = jax.nn.softmax(jnp.where(action_mask, noise, min_logit))
        # both are distributions over the legal actions, so their mix is too
        return ((1 - self.dirichlet_epsilon) * policy) + (
            self.dirichlet_epsilon * dirichlet_noise
        )


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
