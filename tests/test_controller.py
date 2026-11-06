"""The Othello run controller (examples/othello/controller.py): its search and compute
arithmetic, and whole periods with fakes for the queue, train.py and the league."""

import json
import math
import shlex
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from example_modules import load_example

controller = load_example("othello", "controller")[0]
Config, Search, Controller = controller.Config, controller.Search, controller.Controller

SMALL, BIG = "resnet:2x32", "resnet:4x64"
BENCH = {
    SMALL: {
        "net": SMALL,
        "forward_ms": 1.0,
        "train_step_ms": 10.0,
        "forward_batch": 1024,
    },
    BIG: {
        "net": BIG,
        "forward_ms": 2.0,
        "train_step_ms": 30.0,
        "forward_batch": 1024,
    },
}
COSTS = controller.Costs(BENCH)


def config(tmp_path, **kwargs):
    return Config(
        **{
            "name": "ctl",
            "nets": [SMALL, BIG],
            "search": Search(63),
            "worktree": str(tmp_path / "frozen"),
            "queue_cwd": str(tmp_path / "task"),
            "train_args": ["--seed", "1"],
            # 2 epochs of 128 moves
            "buffer": 256,
            "trunk_turnovers": 1.0,
            "branch_turnovers": 1.0,
            "max_periods": 2,
            "ladder": None,
            "average_epochs": None,
            **kwargs,
        }
    )


# searches and compute


def test_a_search_doubles_every_budget_and_an_epochs_cycles():
    pcr = Search(127, 31, 0.25, 8320)
    assert pcr.doubled() == Search(255, 63, 0.25, 16640)
    # an epoch plays and stores the same moves at twice the compute
    assert pcr.epoch_cycles() == 8320 and pcr.cycles_per_move() == 56
    assert pcr.stored_per_epoch() == pytest.approx(8320 / 56 * 0.25)
    assert pcr.doubled().stored_per_epoch() == pytest.approx(pcr.stored_per_epoch())
    assert pcr.train_args() == [
        "--sims",
        "127",
        "--playout-cap",
        "0.25:31",
        "--selfplay-cycles",
        "8320",
        "--advance-every",
        "32",
    ]
    assert pcr.label() == "pcr25x127-31s"
    uniform = Search(63)
    assert uniform.doubled() == Search(127)
    assert uniform.epoch_cycles() == 128 * 64 and uniform.stored_per_epoch() == 128
    assert uniform.train_args() == ["--sims", "63"] and uniform.label() == "63s"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"full": 0},
        {"full": 127, "fast": 31},
        {"full": 127, "full_probability": 0.5},
        {"full": 127, "fast": 127, "full_probability": 0.25, "cycles": 8320},
        {"full": 127, "fast": 31, "full_probability": 0.25, "cycles": 8321},
    ],
)
def test_bad_searches_are_refused(kwargs):
    with pytest.raises(ValueError):
        Search(**kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"train_args": ["--sims", "64"]}, "sets these itself"),
        ({"train_args": ["--epochs=5"]}, "sets these itself"),
        ({"train_args": ["--lr", "1e-3"]}, "sets these itself"),
        ({"train_args": ["--lr-plateau"]}, "sets these itself"),
        ({"lr": 0}, "positive"),
        ({"grow_margin": 1.0}, "fraction"),
        ({"train_args": ["--grow-margin", "0"]}, "sets these itself"),
        ({"games": 401}, "even"),
        ({"nets": [SMALL, "resnet:2x32,heads=pooled"]}, "must differ"),
        ({"trunk_turnovers": 0}, "positive"),
        ({"margin_se": -1.0}, "negative"),
        ({"average_epochs": 0.5}, "at least 1"),
        ({"selfplay_average": True}, "needs average_epochs"),
    ],
)
def test_bad_configs_are_refused(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        config(tmp_path, **kwargs)


def test_the_waiting_network_trains_the_steps_its_own_play_would_afford(tmp_path):
    c = config(tmp_path)
    # 4x64 playing alone (the last): 128 moves x 64 cycles at 2 ms, and 256 steps at 30 ms
    last = 16.384 + 0.256 * 30
    assert controller.regime_seconds(c, Search(63), 1, COSTS) == pytest.approx(last)
    # 2x32 playing, 4x64 waiting at the share s = E / last of the steps: E = own + s T
    own, t = 8.192 + 0.256 * 10, 0.256 * 30
    e = own / (1 - t / last)
    assert controller.regime_seconds(c, Search(63), 0, COSTS) == pytest.approx(e)
    assert e == pytest.approx(own + e / last * t)
    assert (
        controller.waiting_steps(c, Search(63), 0, COSTS)
        == round(256 * e / last)
        == 168
    )
    assert controller.waiting_steps(c, Search(63), 1, COSTS) is None
    # a deeper search makes the bigger network's self-play cost more: fewer steps
    assert controller.waiting_steps(c, Search(127), 0, COSTS) == 148


def test_an_epochs_compute_is_its_networks_evaluations_and_steps(tmp_path):
    c = config(tmp_path)
    compute = controller.epoch_compute(c, Search(63), 0, COSTS)
    # 2x32 evaluates 128 moves x 64 cycles at 1 ms a cycle and takes 256 steps; 4x64, waiting,
    # 168
    assert compute["seconds"] == pytest.approx(8.192 + 0.256 * 10 + 0.168 * 30)
    assert compute["selfplay_evals"] == 8192 * 1024
    assert compute["train_samples"] == {"2x32": 256 * 4096, "4x64": 168 * 4096}
    # the last network plays and trains alone
    alone = controller.epoch_compute(c, Search(63), 1, COSTS)
    assert alone["seconds"] == pytest.approx(16.384 + 0.256 * 30)
    assert alone["train_samples"] == {"4x64": 256 * 4096}
    total = controller.added(controller.scaled(compute, 2), alone)
    assert total["train_samples"] == {
        "2x32": 2 * 256 * 4096,
        "4x64": (2 * 168 + 256) * 4096,
    }


def test_branches_match_on_benchmarked_time(tmp_path):
    c = config(tmp_path)
    stay, double = controller.branch_epochs(c, Search(63), 0, COSTS, c.buffer)
    # double fills its buffer once (2 epochs); stay takes the epochs of the same time
    assert double == 2

    def cost(search):
        return controller.epoch_compute(c, search, 0, COSTS)["seconds"]

    assert stay == round(2 * cost(Search(127)) / cost(Search(63))) == 3


def test_the_window_rule_grows_sublinearly_past_c():
    window = controller.Window()
    assert window.target(100) == 100 and window.target(256) == 256
    # an 8x128 run's 274 minutes, about 30M positions over 1024 environments
    assert window.target(30e6 / 1024) == pytest.approx(1125, rel=0.01)
    # resized once the target is 25% past the capacity, rounded up to a multiple of 32
    assert window.capacity(1000, 256) == 256
    n = 256 * (1 + 0.25 * 0.75 / 0.075) ** (1 / 0.75)
    assert window.capacity(n - 1, 256) == 256 and window.capacity(n + 1, 256) == 352
    assert window.capacity(30e6 / 1024, 1000) == 1000
    assert window.capacity(30e6 / 1024, 800) == 1152


def test_elo_of_a_series():
    assert controller.elo_of(10, 0, 10) == (
        0.0,
        pytest.approx(0.0 + 400 / math.log(10) / 0.25 * 0.5 / math.sqrt(20)),
    )
    elo, se = controller.elo_of(150, 0, 50)
    assert elo == pytest.approx(400 * math.log10(3))
    assert 0 < se < 30
    # a clean sweep is clipped, not infinite
    assert math.isfinite(controller.elo_of(400, 0, 0)[0])


def meta(playing, trained):
    return {"growth": {"playing": playing, "trained": trained, "values": []}}


def test_players_tells_who_played_when_from_the_networks_training():
    # none took over: the one playing throughout
    assert controller.players(meta(0, [5, 5, 0]), meta(0, [8, 8, 0]), 5, 8) == [
        [0, 5, 8]
    ]
    # from scratch, 1 took over after 2 epochs: 0 trained epochs 0-1
    assert controller.players(None, meta(1, [2, 6, 4]), 0, 6) == [[0, 0, 2], [1, 2, 6]]
    # 1 then 2 took over in one step from epoch 10: 1 trained from 10 until 2 played at 17, and
    # 2 joined at 12, when 1 took over from 0
    assert controller.players(meta(0, [10, 10, 0]), meta(2, [12, 17, 8]), 10, 20) == [
        [0, 10, 12],
        [1, 12, 17],
        [2, 17, 20],
    ]
    # without growth, one network
    assert controller.players(None, {"growth": None}, 0, 4) == [[0, 0, 4]]


# whole periods, with fakes


class FakeQueue:
    def __init__(self):
        self.submitted = []

    def submit(self, about, estimate_minutes, command):
        self.submitted.append(
            SimpleNamespace(about=about, minutes=estimate_minutes, command=command)
        )

    def names(self):
        return [shlex.split(s.command)[-1] for s in self.submitted]


class FakeRunner:
    """Plays train.py and bench_nets.py: a step saves the state and checkpoints it would. No
    network takes over during a step (the controller runs every step --grow-manual); with
    --take-over the next plays from its start."""

    def __init__(self, overhead_ms=0.0):
        # the search's own time per cycle, besides the playing network's forward pass
        self.overhead_ms = overhead_ms
        self.calls = []

    def bench(self, nets, args, out):
        return {n: BENCH[n] for n in nets}

    def train(self, args, log):
        self.calls.append(args)
        assert "--grow-manual" in args

        def value(flag):
            return args[args.index(flag) + 1]

        directory, last = Path(value("--ckpt-dir")), int(value("--epochs"))
        nets = [
            value("--net"),
            *(args[i + 1] for i, a in enumerate(args) if a == "--grow"),
        ]
        epoch, playing, trained = 0, 0, [0] * len(nets)
        if "--branch-from" in args:
            start = controller.read_meta(value("--branch-from"))
            epoch, playing = start["epoch"], start["growth"]["playing"]
            trained = list(start["growth"]["trained"])
            if "--take-over" in args:
                playing += 1
        for _ in range(epoch, last):
            for i in (playing, playing + 1):
                if i < len(nets):
                    trained[i] += 1
        directory.mkdir(parents=True)
        averages = (
            [f"avg{float(value('--average-epochs')):g}"]
            if "--average-epochs" in args
            else []
        )
        for net in nets[playing : playing + 2]:
            for kept in ("", *averages):
                (directory / controller.net_name(net) / kept).mkdir(exist_ok=True)
                (
                    directory / controller.net_name(net) / kept / f"{last - 1}.eqx"
                ).touch()
        # each epoch's self-play time, as train.py logs it
        cycle_ms = BENCH[nets[playing]]["forward_ms"] + self.overhead_ms
        seconds = Search(int(value("--sims"))).epoch_cycles() * cycle_ms / 1e3
        with open(log, "a") as f:
            if "--branch-from" in args:
                f.write("branching from ...\n")
            f.writelines(
                f"Epoch {e}: {{'selfplay_seconds': '{seconds:.4f}'}}\n"
                for e in range(epoch, last)
            )
        meta = {
            "epoch": last,
            "growth": {"playing": playing, "trained": trained, "values": []},
        }
        np.savez(directory / f"state-{last}.npz", meta=np.array(json.dumps(meta)))


class FakeLeague:
    """Records what's registered and asked for; a series' result is `elo(a, b)`, as 200 games
    (in wins and losses). `play` gives way the first `give_way` times, and the first
    `give_way_ladder` times it's asked for a series against Edax."""

    def __init__(self, elo, give_way=0, give_way_ladder=0):
        self.elo = elo
        self.give_way = give_way
        self.give_way_ladder = give_way_ladder
        self.registered, self.matches, self.retired, self.resumed = {}, [], [], []

    def register(self, name, checkpoint, net, run, sims):
        self.registered[name] = SimpleNamespace(
            checkpoint=checkpoint, net=net, run=run, sims=list(sims)
        )

    def match(self, a, b, games):
        self.matches.append((a, b, games))
        return len(self.matches)

    def play(self, requests, seconds, log):
        if self.give_way:
            self.give_way -= 1
            return False
        edax = any(self.matches[r - 1][1].startswith("edax") for r in requests)
        if edax and self.give_way_ladder:
            self.give_way_ladder -= 1
            return False
        return True

    def record(self, a, b):
        score = 1 / (1 + 10 ** (-self.elo(a, b) / 400))
        wins = round(200 * score)
        return wins, 0, 200 - wins

    def retire(self, name):
        self.retired.append(name)

    def resume(self, name):
        self.resumed.append(name)


def drain(run_dir, queue, runner, league, limit=40, start=0):
    """Runs the queued steps in order, as the queue would, from the `start`-th, until none is
    left."""
    done = start
    while done < len(queue.submitted):
        assert done < limit
        Controller(run_dir, queue, runner, league).run(queue.names()[done])
        done += 1
    return queue.names()


def decisions(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl").read_text().splitlines()
    ]


def test_a_period_branches_on_the_search_and_keeps_the_winner(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner = FakeQueue(), FakeRunner()
    # double wins by 60 in period 0, loses by 10 in period 1; 4x64 never pays its way
    league = FakeLeague(lambda a, b: 60.0 if "p000-double" in a else -10.0)
    ctl = Controller.init(
        run_dir, config(tmp_path), queue=queue, runner=runner, league=league
    )
    ctl.start()

    steps = drain(run_dir, queue, runner, league)

    assert steps == [
        "bench",
        "p000-trunk",
        "p000-stay",
        "p000-double",
        "p000-test",
        "p001-trunk",
        "p001-stay",
        "p001-double",
        "p001-test",
    ]
    # each step queued from the frozen worktree, with an estimate
    assert all(
        s.command.startswith(
            f"cd {tmp_path / 'frozen'} && taskset -c 0-7 .venv/bin/python "
            "examples/othello/controller.py step "
        )
        and s.minutes >= 5
        for s in queue.submitted
    )
    args = {Path(a[a.index("--ckpt-dir") + 1]).name: a for a in runner.calls}
    # the trunk starts from scratch, 1 turnover (2 epochs) at 63 sims
    trunk = args["p000-trunk"]
    assert "--branch-from" not in trunk and trunk[trunk.index("--sims") + 1] == "63"
    assert trunk[trunk.index("--epochs") + 1] == "2"
    assert ["--grow", BIG] == trunk[2:4]
    # the next's training loss is logged, against the engine set's value error; no network
    # takes over by itself
    assert trunk[trunk.index("--grow-metric") + 1] == "loss"
    assert trunk[trunk.index("--grow-margin") + 1] == "0"
    assert trunk[trunk.index("--grow-guard") + 1] == "engine_value_mse"
    assert "--grow-manual" in trunk
    # 4x64 trains the steps its own play would afford beside 2x32's, at the step's search
    assert trunk[trunk.index("--grow-train-steps") + 1] == "168"
    assert trunk[-2:] == ["--seed", "1"]
    # a constant learning rate, 5e-4 unless the config says otherwise
    lr = trunk.index("--lr")
    assert trunk[lr : lr + 4] == ["--lr", "0.0005", "--lr-final", "0.0005"]
    # both branches go on from its state, at equal benchmarked time
    stay, double = args["p000-stay"], args["p000-double"]
    trunk_state = str(run_dir / "p000-trunk" / "state-2.npz")
    for branch in (stay, double):
        assert branch[branch.index("--branch-from") + 1] == trunk_state
        assert "--grow-manual" in branch
    assert double[double.index("--sims") + 1] == "127"
    assert double[double.index("--grow-train-steps") + 1] == "148"
    assert double[double.index("--epochs") + 1] == "4"
    assert stay[stay.index("--epochs") + 1] == str(2 + 3)
    # double won: the next trunk goes on from its state, at its search
    nxt = args["p001-trunk"]
    assert nxt[nxt.index("--branch-from") + 1] == str(
        run_dir / "p000-double" / "state-4.npz"
    )
    assert nxt[nxt.index("--sims") + 1] == "127"
    # the series: stay's next network against its playing one (size), double's playing network
    # against stay's (sims), and double's next one against stay's
    assert [m[:2] for m in league.matches[:3]] == [
        ("4x64-ctl-p000-stay-e5@128", "2x32-ctl-p000-stay-e5@128"),
        ("2x32-ctl-p000-double-e4@128", "2x32-ctl-p000-stay-e5@128"),
        ("4x64-ctl-p000-double-e4@128", "4x64-ctl-p000-stay-e5@128"),
    ]
    assert league.matches[0][2] == 400
    registered = league.registered["2x32-ctl-p000-double-e4"]
    assert registered.checkpoint == str(run_dir / "p000-double" / "2x32" / "3.eqx")
    assert (registered.net, registered.run, registered.sims) == (
        SMALL,
        "ctl-p000-double",
        [128],
    )
    # each player retired once
    assert sorted(league.retired) == sorted(
        {name for m in league.matches for name in m[:2]}
    )
    # the decisions, and the states no branch can start from any more are gone
    made = decisions(run_dir)
    assert [(d["period"], d["question"], d["decision"]) for d in made] == [
        (0, "test", "double"),
        (1, "test", "stay"),
    ]
    assert [s["question"] for s in made[0]["series"]] == ["size", "sims", "next sims"]
    assert made[0]["series"][1]["elo"] == pytest.approx(60, abs=5)
    assert not (run_dir / "p000-trunk" / "state-2.npz").exists()
    assert not (run_dir / "p000-stay" / "state-5.npz").exists()
    # spent counts every step, kept only the line of states the run went on from
    data = json.loads((run_dir / "controller.json").read_text())
    c = config(tmp_path)

    def cost(search):
        return controller.epoch_compute(c, search, 0, COSTS)["seconds"]

    # period 0's trunk at 63 sims, then its double and period 1's trunk and stay at 127
    assert data["kept"]["seconds"] == pytest.approx(
        2 * cost(Search(63)) + (2 + 2 + 3) * cost(Search(127))
    )
    assert data["spent"]["seconds"] > data["kept"]["seconds"]
    assert data["done"].startswith("after 2 periods")
    status = Controller(run_dir, queue, runner, league).status()
    assert "period 2" in status and "+60" in status


def test_the_next_network_takes_over_once_it_beats_the_playing_one_in_stay(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner = FakeQueue(), FakeRunner()
    # in period 0, stay's 4x64 beats its 2x32 by 60, and double beats stay too: growing first
    league = FakeLeague(lambda a, b: 60.0 if "p000" in a else -10.0)
    Controller.init(
        run_dir, config(tmp_path), queue=queue, runner=runner, league=league
    ).start()

    steps = drain(run_dir, queue, runner, league)

    assert steps[1:] == [
        "p000-trunk",
        "p000-stay",
        "p000-double",
        "p000-test",
        "p001-trunk",
        "p001-stay",
        "p001-double",
        "p001-test",
    ]
    made = decisions(run_dir)
    assert [(d["period"], d["decision"], d["playing"]) for d in made] == [
        (0, "grow", "4x64"),
        (1, "stay", "4x64"),
    ]
    assert made[0]["series"][0]["question"] == "size"
    assert made[0]["series"][0]["elo"] == pytest.approx(60, abs=5)
    # 4x64 takes over from stay's state at the next trunk, at stay's search
    args = {Path(a[a.index("--ckpt-dir") + 1]).name: a for a in runner.calls}
    trunk = args["p001-trunk"]
    assert trunk[trunk.index("--branch-from") + 1] == str(
        run_dir / "p000-stay" / "state-5.npz"
    )
    assert "--take-over" in trunk and trunk[trunk.index("--sims") + 1] == "63"
    assert all(
        "--take-over" not in a for name, a in args.items() if name != "p001-trunk"
    )
    ctl = Controller(run_dir, queue, runner, league)
    assert ctl.step("p001-trunk")["players"] == [[0, 5, 5], [1, 5, 7]]
    # its trunk costs 4x64's epochs, playing alone
    assert ctl.step("p001-trunk")["compute"]["seconds"] == pytest.approx(
        2 * (16.384 + 0.256 * 30)
    )
    # 4x64, the last, plays period 1: only its sims are tested
    assert [m[:2] for m in league.matches[3:]] == [
        ("4x64-ctl-p001-double-e9@128", "4x64-ctl-p001-stay-e10@128")
    ]
    assert ctl.head.playing == 1 and not ctl.head.take_over
    assert not (run_dir / "p000-double" / "state-4.npz").exists()


def test_sims_only_go_up_while_the_next_network_does_not_pay_its_way(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner = FakeQueue(), FakeRunner()
    # 4x64 stays 10 behind; double wins by 60 in period 0, by 20 in period 1 (under the margin)
    league = FakeLeague(
        lambda a, b: (
            -10.0
            if a.startswith("4x64-ctl-p00") and "stay" in a
            else 60.0
            if "p000-double" in a
            else 20.0
        )
    )
    Controller.init(
        run_dir, config(tmp_path), queue=queue, runner=runner, league=league
    ).start()

    drain(run_dir, queue, runner, league)

    assert [d["decision"] for d in decisions(run_dir)] == ["double", "stay"]
    ctl = Controller(run_dir, queue, runner, league)
    assert ctl.head.playing == 0 and ctl.head.search == Search(127)


def test_a_rating_step_that_gives_way_goes_back_in_the_queue(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner = FakeQueue(), FakeRunner()
    league = FakeLeague(lambda a, b: 0.0, give_way=1)
    Controller.init(
        run_dir,
        config(tmp_path, max_periods=1),
        queue=queue,
        runner=runner,
        league=league,
    ).start()

    steps = drain(run_dir, queue, runner, league)

    assert steps == [
        "bench",
        "p000-trunk",
        "p000-stay",
        "p000-double",
        "p000-test",
        "p000-test",
    ]
    # its series asked for once, its players registered once
    assert len(league.matches) == 3


def test_a_failed_step_is_retried_from_where_it_stopped(tmp_path):
    run_dir = tmp_path / "run"
    queue, league = FakeQueue(), FakeLeague(lambda a, b: 0.0)

    class Crashing(FakeRunner):
        crashed = False

        def train(self, args, log):
            if "p000-stay" in " ".join(args) and not self.crashed:
                self.crashed = True
                # it saved a state on the way, then stopped
                directory = Path(args[args.index("--ckpt-dir") + 1])
                directory.mkdir(parents=True)
                np.savez(directory / "state-3.npz", meta=np.array("{}"))
                raise RuntimeError("out of memory")
            super().train(args, log)

    runner = Crashing()
    Controller.init(
        run_dir,
        config(tmp_path, max_periods=1),
        queue=queue,
        runner=runner,
        league=league,
    ).start()
    with pytest.raises(RuntimeError, match="out of memory"):
        drain(run_dir, queue, runner, league)
    ctl = Controller(run_dir, queue, runner, league)
    assert ctl.step("p000-stay")["status"] == "failed"
    assert "out of memory" in ctl.step("p000-stay")["error"]

    ctl.retry("p000-stay")
    # it goes on from its saved state
    assert "--resume" in ctl.train_args(ctl.step("p000-stay"))
    assert "--branch-from" not in ctl.train_args(ctl.step("p000-stay"))


def test_stop_queues_nothing_more(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner, league = FakeQueue(), FakeRunner(), FakeLeague(lambda a, b: 0.0)
    Controller.init(
        run_dir, config(tmp_path), queue=queue, runner=runner, league=league
    ).start()
    Controller(run_dir, queue, runner, league).stop()

    assert drain(run_dir, queue, runner, league) == ["bench"]


def test_the_window_grows_with_the_positions_the_kept_line_stored(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner = FakeQueue(), FakeRunner()
    league = FakeLeague(lambda a, b: 60.0 if "p000-double" in a else -10.0)
    # KataGo's slope: 128 positions per environment (1 epoch) then about +36% at 2 epochs
    window = controller.Window(c=128, beta=0.4)
    Controller.init(
        run_dir,
        config(tmp_path, buffer=128, window=window),
        queue=queue,
        runner=runner,
        league=league,
    ).start()

    drain(run_dir, queue, runner, league)

    args = {Path(a[a.index("--ckpt-dir") + 1]).name: a for a in runner.calls}

    def buffer(step):
        return args[step][args[step].index("--buffer") + 1]

    # period 0: 1 epoch's 128 positions, then the branches from 128
    assert [buffer(s) for s in ("p000-trunk", "p000-stay", "p000-double")] == [
        "128"
    ] * 3
    # double kept, with 2 epochs more: 256 positions, a target of 174.5, 25% past 128
    assert buffer("p001-trunk") == "192"
    # its trunk: 1 turnover of the bigger buffer, 1.5 epochs at 128 a epoch, from epoch 2
    trunk = args["p001-trunk"]
    assert trunk[trunk.index("--epochs") + 1] == str(2 + 2)
    # then 512 positions, a target of 252.8, 25% past 192: the branches resize to 256
    assert buffer("p001-stay") == buffer("p001-double") == "256"
    data = json.loads((run_dir / "controller.json").read_text())
    # stay kept: 3 epochs on from epoch 4, every epoch storing 128 positions
    assert (data["head"]["epoch"], data["head"]["capacity"]) == (7, 256)
    assert data["head"]["stored"] == pytest.approx(128 * 7)


LADDER = ["edax-L1", "edax-L2", "edax-L4"]


def ladder_elo(a, b):
    """stay always kept; period 0's network beats edax-L1 and loses to L2, period 1's beats L2
    and L4, the top"""
    if not b.startswith("edax"):
        return -10.0
    if "p000" in a:
        return {"edax-L1": 100.0, "edax-L2": -20.0}[b]
    return {"edax-L2": 60.0, "edax-L4": 40.0}[b]


def test_each_period_climbs_the_ladder_from_where_the_last_stopped(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner, league = FakeQueue(), FakeRunner(), FakeLeague(ladder_elo)
    Controller.init(
        run_dir,
        config(tmp_path, ladder=LADDER),
        queue=queue,
        runner=runner,
        league=league,
    ).start()

    steps = drain(run_dir, queue, runner, league)

    # after each period's decisions, before the next period's trunk
    assert steps == [
        "bench",
        "p000-trunk",
        "p000-stay",
        "p000-double",
        "p000-test",
        "p000-ladder",
        "p001-trunk",
        "p001-stay",
        "p001-double",
        "p001-test",
        "p001-ladder",
    ]
    ctl = Controller(run_dir, queue, runner, league)
    first, second = ctl.step("p000-ladder"), ctl.step("p001-ladder")
    # the kept network: stay's playing one at its last epoch, a player of the test's series
    player = "2x32-ctl-p000-stay-e5@128"
    assert first["player"]["name"] == player
    assert first["player"]["checkpoint"] == str(
        run_dir / "p000-stay" / "2x32" / "4.eqx"
    )
    assert [x["b"]["name"] for x in first["series"]] == ["edax-L1", "edax-L2"]
    # 200 games, whole wins
    assert [r["elo"] for r in first["results"]] == [
        pytest.approx(100, abs=3),
        pytest.approx(-20, abs=3),
    ]
    assert (first["start_rung"], first["beaten"], first["next_rung"]) == (
        "edax-L1",
        "edax-L1",
        "edax-L2",
    )
    # the next period starts at the rung it lost to; beating the top stays there
    assert [x["b"]["name"] for x in second["series"]] == ["edax-L2", "edax-L4"]
    assert (second["beaten"], second["next_rung"]) == ("edax-L4", "edax-L4")
    assert ctl.head.rung == 2
    assert (player, "edax-L1", 400) in league.matches
    # retired after the test (once, though it played two series), and after its ladder
    assert player in league.resumed and league.retired.count(player) == 2
    ladders = [d for d in decisions(run_dir) if d["question"] == "ladder"]
    assert [(d["period"], d["beaten"], d["next"]) for d in ladders] == [
        (0, "edax-L1", "edax-L2"),
        (1, "edax-L4", "edax-L4"),
    ]
    assert [r["b"] for r in ladders[0]["rungs"]] == ["edax-L1", "edax-L2"]
    assert ctl.data["period"] == 2 and ctl.data["done"].startswith("after 2 periods")


def test_a_run_from_before_the_ladder_climbs_it_in_period_order(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner, league = FakeQueue(), FakeRunner(), FakeLeague(ladder_elo)
    Controller.init(
        run_dir, config(tmp_path), queue=queue, runner=runner, league=league
    ).start()
    drain(run_dir, queue, runner, league)
    # as if from before the ladder: no ladder steps
    data = json.loads((run_dir / "controller.json").read_text())
    data["config"]["ladder"] = LADDER
    (run_dir / "controller.json").write_text(json.dumps(data))
    done = len(queue.submitted)

    Controller(run_dir, queue, runner, league).advance()

    assert queue.names()[done:] == ["p000-ladder"]
    assert drain(run_dir, queue, runner, league, start=done)[done:] == [
        "p000-ladder",
        "p001-ladder",
    ]
    ctl = Controller(run_dir, queue, runner, league)
    assert ctl.step("p001-ladder")["start_rung"] == "edax-L2"
    # nothing more: the run was done
    ctl.advance()
    assert len(queue.submitted) == done + 2


def test_a_ladder_that_gives_way_goes_on_where_it_was(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner = FakeQueue(), FakeRunner()
    league = FakeLeague(ladder_elo, give_way_ladder=2)
    Controller.init(
        run_dir,
        config(tmp_path, ladder=LADDER, max_periods=1),
        queue=queue,
        runner=runner,
        league=league,
    ).start()

    steps = drain(run_dir, queue, runner, league)

    # it gave way during each rung's series, and went on with the same series each time
    assert steps[-3:] == ["p000-ladder"] * 3
    edax = [m for m in league.matches if m[1].startswith("edax")]
    assert [m[1] for m in edax] == ["edax-L1", "edax-L2"]
    ladder = Controller(run_dir, queue, runner, league).step("p000-ladder")
    assert ladder["requests"] == [4, 5] and len(ladder["results"]) == 2


def test_the_benchmark_times_training_as_train_py_takes_it(tmp_path):
    run_dir = tmp_path / "run"
    queue, league = FakeQueue(), FakeLeague(lambda a, b: 0.0)

    class Recording(FakeRunner):
        def bench(self, nets, args, out):
            self.bench_args = args
            return super().bench(nets, args, out)

    runner = Recording()
    Controller.init(
        run_dir,
        config(tmp_path, train_args=["--grad-accum", "4", "--remat"]),
        queue=queue,
        runner=runner,
        league=league,
    ).start()
    Controller(run_dir, queue, runner, league).run("bench")
    assert runner.bench_args == [
        "--train-batch",
        "4096",
        "--remat",
        "--grad-accum",
        "4",
    ]


def test_steps_share_a_compilation_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **kw: calls.append(kw))
    runner = controller.LocalRunner(config(tmp_path), tmp_path / "run" / "compiled")
    runner.train(["--epochs", "1"], tmp_path / "log")
    assert calls[0]["env"]["JAX_COMPILATION_CACHE_DIR"] == str(
        tmp_path / "run" / "compiled"
    )
    assert "JAX_PLATFORMS" not in calls[0]["env"]


# the cost model's measured part, the margins, averaged weights


def test_costs_add_the_search_overhead_measured_at_the_nearest_budget():
    assert COSTS.cycle_ms(SMALL, Search(63)) == 1.0
    costs = controller.Costs(BENCH, {"63s": 0.5, "pcr25x255-63s": 0.75})
    assert costs.cycle_ms(SMALL, Search(63)) == 1.5
    # not measured: the nearest budget's (by ratio), the bigger one on a tie
    assert costs.overhead(Search(100)) == 0.5
    assert costs.overhead(Search(511)) == 0.75
    tie = controller.Costs(BENCH, {"64s": 0.5, "256s": 0.75})
    assert tie.overhead(Search(128)) == 0.75
    assert costs.train_ms(BIG) == 30.0


def test_a_steps_self_play_overhead_comes_from_its_log_without_compiling_epochs(
    tmp_path,
):
    log = tmp_path / "step.log"
    # 8192 cycles of 2x32 (1 ms each): each process's first epoch compiles; a retry's last try
    # counts
    log.write_text(
        "branching from x\n"
        "Epoch 2: {'selfplay_seconds': '60.0000', 'train_seconds': '1.0'}\n"
        "Epoch 3: {'selfplay_seconds': '12.288', 'train_seconds': '1.0'}\n"
        "Epoch 4: {'selfplay_seconds': '99.0'}\n"
        "resuming from y\n"
        "Epoch 4: {'selfplay_seconds': '70.0'}\n"
        "Epoch 5: {'selfplay_seconds': '12.288'}\n"
        "Epoch 6: {'selfplay_seconds': '16.384'}\n"
    )
    measured = controller.selfplay_overhead(log, Search(63), SMALL, COSTS)
    assert measured is not None
    # epochs 3, 5 and 6: median 12.288 s, 1.5 ms a cycle, 0.5 beyond the forward pass
    assert measured["epochs"] == 3 and measured["selfplay_seconds"] == 12.288
    assert measured["overhead_ms"] == pytest.approx(0.5)
    assert (
        controller.selfplay_overhead(tmp_path / "none.log", Search(63), SMALL, COSTS)
        is None
    )


def test_the_run_prices_its_steps_with_the_overhead_they_measure(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner = FakeQueue(), FakeRunner(overhead_ms=0.5)
    league = FakeLeague(lambda a, b: -10.0)
    Controller.init(
        run_dir,
        config(tmp_path, max_periods=1),
        queue=queue,
        runner=runner,
        league=league,
    ).start()

    drain(run_dir, queue, runner, league)

    ctl = Controller(run_dir, queue, runner, league)
    trunk = ctl.step("p000-trunk")
    assert trunk["measured"]["overhead_ms"] == pytest.approx(0.5)
    # the trunk counts what it measured; the branches were matched with it, double at 127
    # sims taking 63's overhead until it measured its own
    c = config(tmp_path, max_periods=1)
    measured = controller.Costs(BENCH, {"63s": 0.5})
    assert trunk["compute"]["seconds"] == pytest.approx(
        2 * controller.epoch_compute(c, Search(63), 0, measured)["seconds"]
    )
    stay, double = controller.branch_epochs(c, Search(63), 0, measured, 256)
    assert (
        ctl.step("p000-stay")["last_epoch"],
        ctl.step("p000-double")["last_epoch"],
    ) == (
        2 + stay,
        2 + double,
    )
    assert ctl.step("p000-stay")["waiting_steps"] == controller.waiting_steps(
        c, Search(63), 0, measured
    )
    assert ctl.costs.overhead_ms == pytest.approx({"63s": 0.5, "127s": 0.5})


@pytest.mark.parametrize(
    "elo, margin_se, decision",
    [(60.0, 2.0, "double"), (40.0, 2.0, "stay"), (40.0, 1.0, "double")],
)
def test_a_challenger_must_win_by_its_series_standard_errors(
    tmp_path, elo, margin_se, decision
):
    run_dir = tmp_path / "run"
    queue, runner = FakeQueue(), FakeRunner()
    # 200 games: a standard error of about 25 Elo
    league = FakeLeague(lambda a, b: elo if "double" in a else -10.0)
    Controller.init(
        run_dir,
        config(tmp_path, max_periods=1, margin_se=margin_se),
        queue=queue,
        runner=runner,
        league=league,
    ).start()

    drain(run_dir, queue, runner, league)

    (made,) = decisions(run_dir)
    assert made["decision"] == decision and made["margin_se"] == margin_se
    sims = made["series"][1]
    assert sims["question"] == "sims" and 20 < sims["se"] < 30


def test_the_league_rates_the_networks_weights_averaged(tmp_path):
    run_dir = tmp_path / "run"
    queue, runner = FakeQueue(), FakeRunner()
    league = FakeLeague(lambda a, b: -10.0)
    Controller.init(
        run_dir,
        config(tmp_path, max_periods=1, average_epochs=4.0, selfplay_average=True),
        queue=queue,
        runner=runner,
        league=league,
    ).start()

    drain(run_dir, queue, runner, league)

    args = {Path(a[a.index("--ckpt-dir") + 1]).name: a for a in runner.calls}
    for step_args in args.values():
        assert step_args[step_args.index("--average-epochs") + 1] == "4"
        assert step_args[step_args.index("--selfplay-average") + 1] == "4"
    assert league.matches[0][:2] == (
        "4x64-ctl-p000-stay-e5-avg4@128",
        "2x32-ctl-p000-stay-e5-avg4@128",
    )
    registered = league.registered["2x32-ctl-p000-stay-e5-avg4"]
    assert registered.checkpoint == str(
        run_dir / "p000-stay" / "2x32" / "avg4" / "4.eqx"
    )
