
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from operator import itemgetter
from typing import Any

import jax

from core.common import partition
from core.evaluators.evaluator import Evaluator
from core.types import EnvInitFn, EnvStepFn


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TestState:
    """Base class for TestState."""


class BaseTester:
    """Base class for Testers.

    A Tester is used to evaluate the performance of an agent in an environment,
    in some cases against one or more opponents.

    A Tester may maintain its own internal state.
    """
    def __init__(self, num_keys: int, epochs_per_test: int = 1, render_fn: Callable | None = None, 
                 render_dir: str = '/tmp/turbozero/', name: str | None = None):
        """Initializes a Tester.

        Args:
            num_keys: number of keys to use for tester
                - often equal to number of episodes
                - provided on initialization to ensure reproducibility, even when a different number of devices is used
                - perhaps there is a better way to enforce this
            epochs_per_test: number of epochs between each test
            render_fn: (optional) function to render frames from a test episode to a .gif
            render_dir: directory to save .gifs
            name: (optional) name of the tester (used for logging and differentiating between testers)
                - defaults to the class name
        """
        self.num_keys = num_keys
        self.epochs_per_test = epochs_per_test
        self.render_fn = render_fn
        self.render_dir = render_dir
        if name is None:
            name = self.__class__.__name__
        self.name = name


    def init(self, *args, **kwargs) -> TestState: #pylint: disable=unused-argument
        """Initializes the internal state of the Tester."""
        return TestState()


    def check_size_compatibilities(self, num_devices: int) -> None: #pylint: disable=unused-argument
        """Checks if tester configuration is compatible with number of devices being utilized."""
        return


    def split_keys(self, key: jax.Array, num_devices: int) -> jax.Array:
        """Splits keys across devices.

        Args:
            key: rng
            num_devices: number of devices

        Returns:
            jax.Array: keys split across devices
        """
        # partition keys across devices (do this here so its reproducible no matter the number of devices used)
        keys = jax.random.split(key, self.num_keys)
        keys = partition(keys, num_devices)
        return keys


    def run(self, key: jax.Array, epoch_num: int, max_steps: int, num_devices: int, #pylint: disable=unused-argument 
        env_step_fn: EnvStepFn, env_init_fn: EnvInitFn, evaluator: Evaluator, state: TestState,
        params: Any, *args) -> tuple[TestState, dict, str | None]:
        """Runs the test, if the current epoch is an epoch that should be tested on (i.e. `epoch_num % epochs_per_test == 0`).

        If a render function is provided, saves a .gif of the first episode of the test.

        Args:
            key: rng
            epoch_num: current epoch number
            max_steps: maximum number of steps per episode
            num_devices: number of devices
            env_step_fn: environment step function
            env_init_fn: environment initialization function
            evaluator: evaluator used by agent
            state: internal state of the tester
            params: nn parameters used by agent

        Returns:
            Tuple[TestState, Dict, str | None]:
                - updated internal state of the tester
                - metrics from the test
                - path to .gif of the first episode of the test (if render function provided, otherwise None)
                - on epochs that are not tested, returns `state` unchanged, empty metrics, and None
        """
        # split keys across devices
        keys = self.split_keys(key, num_devices)

        if epoch_num % self.epochs_per_test == 0:
            # run test
            state, metrics, frames, p_ids = self.test(max_steps, env_step_fn, \
                                                      env_init_fn, evaluator, keys, state, params)
            
            if self.render_fn is not None:
                # render first episode to .gif
                # get frames from first episode
                frames = jax.tree.map(lambda x: x[0], frames)
                # get player ids from first episode
                p_ids = p_ids[0]
                # get list of frames: the initial state, then one per step
                frame_list = [jax.device_get(jax.tree.map(itemgetter(i), frames)) for i in range(max_steps + 1)]
                # render frames to .gif
                path_to_rendering = self.render_fn(frame_list, p_ids, f"{self.name}_{epoch_num}", self.render_dir)
            else:
                path_to_rendering = None
            return state, metrics, path_to_rendering
        return state, {}, None

    
    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(self, max_steps: int, env_step_fn: EnvStepFn, env_init_fn: EnvInitFn, evaluator: Evaluator,
        keys: jax.Array, state: TestState, params: Any) -> tuple[TestState, dict, Any, jax.Array]:
        """Run the test implemented by the Tester. Parallelized across devices.

        Implemented by subclasses.

        Args:
            max_steps: maximum number of steps per episode
            env_step_fn: environment step function
            env_init_fn: environment initialization function
            evaluator: evaluator used by agent
            keys: rng
            state: internal state of the tester
            params: nn parameters used by agent

        Returns:
            Tuple[TestState, Dict, Any, jax.Array]:
                - updated internal state of the tester
                - metrics from the test
                - frames from the test (used to produce renderings)
                - player ids from the test (used to produce renderings)
        """
        raise NotImplementedError()
