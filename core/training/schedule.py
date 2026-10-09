from collections.abc import Sequence
from itertools import pairwise

from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.mcts import MCTS


def check_epochs(epochs: Sequence[int]) -> None:
    """Checks a schedule's first epochs start at 0 and increase."""
    if not epochs or epochs[0] != 0:
        raise ValueError(
            f"the schedule must start at epoch 0, got epochs {list(epochs)}"
        )
    if any(a >= b for a, b in pairwise(epochs)):
        raise ValueError(f"the schedule's epochs must increase, got {list(epochs)}")


class EvaluatorSchedule:
    """Self-play evaluators that take over from one another at given epochs, e.g. to search
    more MCTS iterations a move as training goes on:

        EvaluatorSchedule([(0, search_32), (50, search_64), (150, search_128)])

    (OLIVAW's self-play searched 100 iterations a move, then 200 from about generation 4
    and 400 from about generation 11, of 20: arXiv 2103.17228.)

    Pass one to `Trainer` as its `evaluator`. Each evaluator plays every epoch from its own
    up to the next one's. Self-play is compiled for each evaluator the first time it plays,
    so a schedule should change at a few epochs, not every one. At each change the
    evaluator's states (e.g. MCTS trees) are initialized again, since the new evaluator's
    may differ in shape; games in progress carry on, searching from an empty tree.
    """

    def __init__(self, stages: Sequence[tuple[int, Evaluator]]):
        """Initializes an EvaluatorSchedule.

        Args:
            stages: (first epoch, evaluator) pairs, in increasing order of epoch, starting
                at epoch 0
        """
        check_epochs([epoch for epoch, _ in stages])
        self.stages = tuple(stages)

    def at(self, epoch: int) -> Evaluator:
        """The evaluator that plays `epoch`."""
        return [evaluator for start, evaluator in self.stages if start <= epoch][-1]

    def get_config(self) -> list[dict]:
        """Returns each stage's first epoch and evaluator config. Used for logging."""
        return [
            {
                "epoch": epoch,
                "evaluator": evaluator.__class__.__name__,
                "config": evaluator.get_config(),
            }
            for epoch, evaluator in self.stages
        ]


def describe(evaluator: Evaluator) -> str:
    """A short description of a self-play evaluator, for the console and the monitor."""
    name = evaluator.__class__.__name__
    if isinstance(evaluator, MCTS):
        return f"{name}, {evaluator.num_iterations} iterations a move"
    return name


def parse_schedule(text: str) -> list[tuple[int, int]]:
    """Parses a schedule written `epoch:value,epoch:value,...`, e.g. `0:32,50:64,150:128`
    for 32 MCTS iterations a move from epoch 0, 64 from epoch 50 and 128 from epoch 150.

    Args:
        text: the schedule

    Returns:
        list[tuple[int, int]]: (first epoch, value) pairs, checked to start at epoch 0 and
            to increase in epoch
    """
    try:
        stages = [
            (int(epoch), int(value))
            for epoch, value in (stage.split(":") for stage in text.split(","))
        ]
    except ValueError:
        raise ValueError(
            f"expected epoch:value pairs separated by commas, e.g. 0:32,50:64, got {text!r}"
        ) from None
    check_epochs([epoch for epoch, _ in stages])
    return stages
