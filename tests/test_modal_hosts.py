"""Modal containers as cloud hosts, with stand-ins for Modal's calls: a container on another CPU
model than the one asked for is swapped for a new one without costing its slot an attempt."""

from typing import Any

from example_modules import load_example

(modal_hosts,) = load_example("othello", "modal_hosts")


class Call:
    """A spawned container: running until `said` is set, then what it returned."""

    def __init__(self) -> None:
        self.said: Any = None
        self.done = False

    def get(self, timeout: float) -> Any:
        if not self.done:
            raise TimeoutError
        return self.said


class Hosts(modal_hosts.ModalHosts):
    def __init__(self, cpu_id: str | None = "GenuineIntel-6-173"):
        super().__init__("CPU", 8, 16, cpu_id)
        self.started: list[Call] = []

    def create(self, name, zone, labels, metadata, hours) -> str | None:
        call = Call()
        self.started.append(call)
        self.calls[name] = (labels["tz-cpu-slot"], labels["tz-cpu-job"], call)
        self.made[name] = (zone, labels, metadata, hours)
        return None

    def _stop_app(self, name: str) -> None:
        pass


def start(hosts: Hosts) -> None:
    hosts.create("h-0", "modal", {"tz-cpu-slot": "0", "tz-cpu-job": "j"}, {}, 1.0)


def test_a_container_on_another_cpu_is_replaced_and_its_host_still_runs():
    hosts = Hosts()
    start(hosts)
    first = hosts.started[0]
    first.done, first.said = True, f"{modal_hosts.WRONG_CPU}AuthenticAMD-25-1"
    running = hosts.instances("j")
    assert [h["name"] for h in running] == ["h-0"]
    assert len(hosts.started) == 2 and hosts.calls["h-0"][2] is hosts.started[1]


def test_a_container_that_ends_otherwise_has_ended():
    hosts = Hosts()
    start(hosts)
    hosts.started[0].done = True
    assert hosts.instances("j") == []
    assert len(hosts.started) == 1


def test_a_host_gives_up_after_its_rerolls():
    hosts = Hosts()
    start(hosts)
    running: list[dict[str, str]] = []
    for _ in range(modal_hosts.REROLLS + 1):
        call = hosts.started[-1]
        call.done, call.said = True, f"{modal_hosts.WRONG_CPU}AuthenticAMD-25-1"
        running = hosts.instances("j")
    assert running == []
    assert len(hosts.started) == modal_hosts.REROLLS + 1
