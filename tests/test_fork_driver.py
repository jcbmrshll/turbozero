"""fork_driver.py's decisions, with a fake Modal and logs written as train.py writes them."""

import shlex
from pathlib import Path

import pytest
from example_modules import load_example

fork_driver = load_example("othello", "fork_driver")[0]

NETS = [("r11", "resnet:11x64"), ("r15", "resnet:15x64")]
RULES = fork_driver.Rules(
    fork_from_epoch=20,
    window_epochs=5,
    margin=0.003,
    windows_ahead=2,
    min_epochs=10,
    after_lr_step=5,
    max_fork_epochs=40,
    retry_after=20,
    plateau_epochs=10,
    plateau_per_100=0.001,
    lr_min_epochs=20,
)


class FakeModal:
    """Records what the driver asks of Modal; jobs keep running unless a test ends them."""

    def __init__(self):
        self.jobs: dict[str, list[str]] = {}
        self.stopped: list[str] = []
        self.resumed: list[str] = []
        self.states: dict[str, str] = {}
        self.in_bucket: set[str] = set()

    def launch(self, job, steps):
        assert job not in self.jobs
        self.jobs[job] = list(steps)

    def extend(self, job, steps):
        self.jobs[job] += steps

    def stop(self, job):
        self.stopped.append(job)

    def state(self, job):
        return self.states.get(job, "running")

    def resume(self, job):
        self.resumed.append(job)

    def has_state(self, path):
        return path in self.in_bucket


def config(**more):
    return fork_driver.Config(
        **{
            "name": "t",
            "nets": NETS,
            "train_args": ["--seed", "0"],
            "tags": "slick:x",
            "commit": "abc",
            "segment_epochs": 10,
            "lead_epochs": 2,
            "rules": RULES,
            **more,
        }
    )


@pytest.fixture
def run(tmp_path):
    modal = FakeModal()
    driver = fork_driver.Driver(tmp_path, config(), modal, log=lambda _: None)
    return driver, modal


def argv(step: str) -> list[str]:
    return shlex.split(step.split(" > ")[0])


def flag(step: str, name: str) -> list[str]:
    words = argv(step)
    return [words[i + 1] for i, w in enumerate(words) if w == name]


def log_epochs(driver, line_name, values, nets=("r11", "r15")):
    """Writes train.py's metric lines for epochs `values` maps to the playing network's metric,
    every network logging the same (those of `nets`), into each epoch's segment log."""
    line = driver.state.lines[line_name]
    for epoch, value in values.items():
        segment = next(s for s in line.segments if s.start <= epoch < s.end)
        metrics = {"avg4/engine_value_mse": f"{value:.5f}"} | {
            f"{n}/avg4/engine_value_mse": f"{value:.5f}" for n in nets
        }
        Path(segment.log).parent.mkdir(parents=True, exist_ok=True)
        with open(segment.log, "a") as f:
            f.write(f"Epoch {epoch}: {{'moves': '1.0'}}\n")
            f.write(f"Epoch {epoch}: {metrics}\n")


def advance(driver, line_name, upto, value=lambda _: 0.7):
    """The line logs every epoch up to `upto` (exclusive) not yet logged, ticking as a running
    driver would at every epoch."""
    for epoch in range(driver.logged_to(driver.state.lines[line_name]), upto):
        line = driver.state.lines[line_name]
        while epoch >= line.end:
            driver.tick()
        log_epochs(driver, line_name, {epoch: value(epoch)})
        driver.tick()


def test_the_mainline_starts_from_scratch_with_the_first_settings(run):
    driver, modal = run
    driver.tick()
    (step,) = modal.jobs["t-main"]
    assert flag(step, "--net") == ["r11=resnet:11x64"]
    assert flag(step, "--grow") == ["r15=resnet:15x64"]
    assert flag(step, "--lr") == flag(step, "--lr-final") == ["0.001"]
    assert flag(step, "--weight-decay") == ["0.5"]
    assert flag(step, "--lr-of") == ["r15=0.001"]
    assert flag(step, "--weight-decay-of") == ["r15=0.5"]
    assert flag(step, "--buffer") == ["9000"] and flag(step, "--buffer-window") == [
        "1125"
    ]
    assert flag(step, "--sims") == ["128"] and flag(step, "--playout-cap") == [
        "0.25:32"
    ]
    assert flag(step, "--epochs") == flag(step, "--save-state-at") == ["10"]
    assert not flag(step, "--branch-from") and "--seed" in argv(step)


def test_each_segment_branches_from_the_last_ones_end(run):
    driver, modal = run
    driver.tick()
    advance(driver, "main", 8)
    assert len(modal.jobs["t-main"]) == 2
    second = modal.jobs["t-main"][1]
    assert flag(second, "--branch-from") == [
        f"{driver.dir}/lines/main/seg0/state-10.npz"
    ]
    assert flag(second, "--epochs") == ["20"]
    assert flag(second, "--ckpt-dir") == [f"{driver.dir}/lines/main/seg10"]


def test_a_network_whose_metric_plateaus_steps_down_its_own_learning_rate(run):
    driver, modal = run
    driver.tick()
    # r11 flat; r15 still improving
    for epoch in range(28):
        line = driver.state.lines["main"]
        while epoch >= line.end:
            driver.tick()
        segment = next(s for s in line.segments if s.start <= epoch < s.end)
        Path(segment.log).parent.mkdir(parents=True, exist_ok=True)
        with open(segment.log, "a") as f:
            f.write(
                f"Epoch {epoch}: {{'avg4/engine_value_mse': '0.7', "
                f"'r11/avg4/engine_value_mse': '0.7', "
                f"'r15/avg4/engine_value_mse': '{0.7 - 0.01 * epoch:.4f}'}}\n"
            )
        driver.tick()
    assert driver.state.lr_steps == [(30, "r11", 1)]
    step = modal.jobs["t-main"][-1]
    assert flag(step, "--lr") == ["0.0005"] and flag(step, "--weight-decay") == ["0.25"]
    assert flag(step, "--lr-of") == ["r15=0.001"]


def test_forks_start_from_a_segment_end_in_the_bucket_and_alternate(run):
    driver, modal = run
    driver.tick()
    advance(driver, "main", 21)
    # the state at 20 isn't in the bucket yet: no fork
    assert list(modal.jobs) == ["t-main"]
    modal.in_bucket.add(f"{driver.dir}/lines/main/seg10/state-20.npz")
    driver.tick()
    (step,) = modal.jobs["t-window0"]
    assert flag(step, "--branch-from") == [
        f"{driver.dir}/lines/main/seg10/state-20.npz"
    ]
    assert flag(step, "--buffer-window") == ["2250"] and flag(step, "--sims") == ["128"]
    assert driver.state.lines["window0"].start == 20


def test_a_sims_fork_well_ahead_takes_over_the_mainline(run):
    driver, modal = run
    driver.tick()
    advance(driver, "main", 21)
    modal.in_bucket.add(f"{driver.dir}/lines/main/seg10/state-20.npz")
    driver.state.started = ["window"]  # so the next is a sims fork
    driver.tick()
    assert "t-sims1" in modal.jobs
    advance(driver, "main", 45)
    advance(driver, "sims1", 29, value=lambda _: 0.69)
    assert driver.state.main == "main"  # too early: min_epochs
    advance(driver, "sims1", 30, value=lambda _: 0.69)
    assert driver.state.main == "sims1"
    assert "t-main" in modal.stopped
    assert driver.state.lines["main"].status == "replaced"
    # the new mainline decides its own learning rate steps from here
    assert driver.state.lines["sims1"].kind == "main"
    # and the next fork starts only from a segment end the new mainline has decided
    modal.in_bucket.update(
        f"{driver.dir}/lines/sims1/seg{e}/state-{e + 10}.npz" for e in (20, 30, 40)
    )
    advance(driver, "sims1", 52, value=lambda _: 0.69)
    assert any(l.kind == "window" for l in driver.state.lines.values())
    assert_lines_follow_the_mainlines_learning_rates(driver)


def assert_lines_follow_the_mainlines_learning_rates(driver):
    """Every segment of every line still running took the mainline's steps at its start."""
    for line in driver.state.lines.values():
        if line.status != "running":
            continue
        for segment in line.segments:
            assert segment.lrs == driver.lrs_at(segment.start), (
                line.name,
                segment.start,
            )


def test_a_fork_that_plateaus_level_with_the_mainline_is_killed(run):
    driver, modal = run
    driver.tick()
    advance(driver, "main", 21)
    modal.in_bucket.add(f"{driver.dir}/lines/main/seg10/state-20.npz")
    driver.state.started = ["window"]
    driver.tick()
    advance(driver, "main", 45)
    advance(driver, "sims1", 31)
    fork = driver.state.lines["sims1"]
    assert fork.status == "killed" and "t-sims1" in modal.stopped
    assert (
        driver.state.retry_at["sims"]
        == driver.logged_to(driver.state.lines["main"]) + 20
    )
    assert driver.state.main == "main"


def test_a_window_fork_is_judged_only_after_a_learning_rate_step(run):
    driver, modal = run
    driver.tick()
    # the mainline improves slowly: no plateau, no learning rate step
    advance(driver, "main", 21, value=lambda e: 0.8 - 0.002 * e)
    modal.in_bucket.add(f"{driver.dir}/lines/main/seg10/state-20.npz")
    driver.tick()
    assert "t-window0" in modal.jobs
    advance(driver, "main", 58, value=lambda e: 0.8 - 0.002 * e)
    # well ahead, and long enough for a sims fork: still running
    advance(driver, "window0", 45, value=lambda e: 0.79 - 0.002 * e)
    assert driver.state.lines["window0"].status == "running"
    assert driver.state.main == "main"
    # a learning rate step at 40, then after_lr_step epochs: it takes over
    driver.state.lr_steps.append((40, "r11", 1))
    advance(driver, "window0", 46, value=lambda e: 0.79 - 0.002 * e)
    assert driver.state.main == "window0"


def test_a_fork_waits_for_the_mainlines_learning_rates_at_its_next_segment(run):
    driver, modal = run
    driver.tick()
    advance(driver, "main", 21)
    modal.in_bucket.add(f"{driver.dir}/lines/main/seg10/state-20.npz")
    driver.tick()
    # the fork reaches its segment end (30) before the mainline has decided there
    log_epochs(driver, "window0", {e: 0.7 for e in range(20, 29)})
    driver.tick()
    assert len(modal.jobs["t-window0"]) == 1
    advance(driver, "main", 29)
    driver.tick()
    assert len(modal.jobs["t-window0"]) == 2


def test_a_job_that_ended_short_is_resumed(run):
    driver, modal = run
    driver.tick()
    advance(driver, "main", 5)
    modal.states["t-main"] = "timeout"
    driver.tick()
    assert modal.resumed == ["t-main"]


def test_a_resumed_attempts_repeated_epochs_take_its_later_lines(tmp_path):
    log = tmp_path / "x.log"
    log.write_text(
        "Epoch 3: {'moves': '1.0'}\n"
        "Epoch 3: {'avg4/engine_value_mse': '0.5'}\n"
        "modal-jobs: step 0 of j, attempt 2, resuming\n"
        "Epoch 3: {'moves': '1.0'}\n"
        "Epoch 3: {'avg4/engine_value_mse': '0.4', 'x_game': 'not a number'}\n"
    )
    assert fork_driver.read_log(log) == {
        3: {"moves": 1.0, "avg4/engine_value_mse": 0.4}
    }


def test_the_driver_sets_its_own_flags():
    with pytest.raises(ValueError, match="the driver sets --lr"):
        config(train_args=["--lr", "1e-3"]).check()
    with pytest.raises(ValueError, match="windows go up to the buffer"):
        config(buffer=2000).check()


def test_the_template_loads(tmp_path):
    (tmp_path / "driver.toml").write_text(
        fork_driver.TEMPLATE.format(
            name="r",
            tags="t",
            train_args=", ".join(f'"{a}"' for a in fork_driver.RECIPE),
        )
    )
    loaded = fork_driver.Config.load(tmp_path / "driver.toml")
    assert [n for n, _ in loaded.nets] == ["r11", "r15", "r19"]
    assert loaded.sims[1] == (256, 64) and loaded.rules.margin == 0.003


def test_on_modal_lines_are_jobs_that_follow_and_keep_states_in_the_bucket(
    tmp_path, monkeypatch
):
    # patched where fork_driver looks them up
    modal_jobs = fork_driver.modal_jobs
    (modal_store,) = load_example("othello", "modal_store")
    Job, job_key, get_json = fork_driver.Job, fork_driver.job_key, fork_driver.get_json
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    store = modal_store.DirStore(tmp_path / "bucket")
    store.put_bytes(
        "files/.cache/turbozero/othello/engine-sets/nr26-evals-29k.npz", b"set"
    )
    spawned = []
    monkeypatch.setattr(modal_jobs, "open_store", lambda: (store, {}))
    monkeypatch.setattr(modal_jobs, "spawn", lambda job, store: spawned.append(job))
    monkeypatch.setattr(modal_jobs, "ensure_sync", lambda: None)
    monkeypatch.setattr(modal_jobs, "resolve_commit", lambda c: "f" * 40)
    run_dir = home / ".cache" / "turbozero" / "runs" / "d"
    run_dir.mkdir(parents=True)
    backend = fork_driver.ModalBackend(config(hours_per_line=10))
    driver = fork_driver.Driver(run_dir, backend.config, backend, log=lambda _: None)

    driver.tick()
    job = Job.from_dict(get_json(store, job_key("t-main", "job.json")))
    assert job.states_in_bucket and job.follow_minutes == 30 and job.hours == 10
    assert spawned == [job]
    (step,) = job.steps
    assert step.ckpt_dir == f"{run_dir}/lines/main/seg0"
    assert step.log == f"{run_dir}/logs/main-seg0.log"

    advance(driver, "main", 8)
    job = Job.from_dict(get_json(store, job_key("t-main", "job.json")))
    assert [s.ckpt_dir.rsplit("/", 1)[1] for s in job.steps] == ["seg0", "seg10"]
    assert not backend.has_state(f"{run_dir}/lines/main/seg0/state-10.npz")
    store.put_bytes(job.key_of(f"{run_dir}/lines/main/seg0/state-10.npz"), b"s")
    assert backend.has_state(f"{run_dir}/lines/main/seg0/state-10.npz")

    # a job that ran out of time goes on, with more
    fork_driver.put_json(
        store,
        job_key("t-main", "status.json"),
        {"state": "timeout", "attempts": [{"seconds": 36000}]},
    )
    driver.tick()
    resumed = get_json(store, job_key("t-main", "job.json"))
    assert resumed["hours"] == pytest.approx(20) and len(spawned) == 2


def test_a_fork_starts_only_from_a_segment_end_the_mainline_has_decided(run):
    # as after an adoption: the new mainline has logged its segment's end, but hasn't yet
    # decided its learning rates there (nor added the next segment)
    driver, modal = run
    driver.tick()
    advance(driver, "main", 20)
    main = driver.state.lines["main"]
    main.segments = main.segments[:2]
    log_epochs(driver, "main", {19: 0.7})
    modal.in_bucket.add(f"{driver.dir}/lines/main/seg10/state-20.npz")
    driver.maybe_fork(main)
    assert list(modal.jobs) == ["t-main"]
    driver.tick()  # decides at 20, adds the next segment, then forks
    assert "t-window0" in modal.jobs
    assert_lines_follow_the_mainlines_learning_rates(driver)
