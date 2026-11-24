"""Modal containers as cloud hosts, with stand-ins for Modal's app and calls: one app a launch, a
call a host, and a container on another CPU than the one asked for is swapped for a new call
without costing its slot an attempt."""

from typing import Any

from example_modules import load_example

(modal_hosts,) = load_example("othello", "modal_hosts")

WRONG = f"{modal_hosts.WRONG_CPU}AuthenticAMD-25-1"


class Call:
    """A spawned call: running until `done`, then what it returned."""

    def __init__(self) -> None:
        self.said: Any = None
        self.done = False
        self.cancelled = False

    def get(self, timeout: float) -> Any:
        if not self.done:
            raise TimeoutError
        return self.said

    def cancel(self) -> None:
        self.cancelled = True


class Hosts(modal_hosts.ModalHosts):
    def __init__(self, cpu: str | None = "avx512-vnni"):
        super().__init__("CPU", 8, 16, cpu)
        self.opened: list[tuple[str, float]] = []
        self.started: list[Call] = []

    def _open(self, job: str, hours: float) -> None:
        self.opened.append((job, hours))

    def _spawn(self, metadata: dict[str, str]) -> Any:
        call = Call()
        self.started.append(call)
        return call


def start(hosts: Hosts, slot: str = "0") -> None:
    labels = {"tz-cpu-slot": slot, "tz-cpu-job": "j"}
    assert hosts.create(f"h-{slot}", "modal", labels, {"tz-slot": slot}, 1.0) is None


def test_every_host_is_a_call_of_the_launchs_one_app():
    hosts = Hosts()
    start(hosts, "0")
    start(hosts, "1")
    assert {h["name"] for h in hosts.instances("j")} == {"h-0", "h-1"}
    assert hosts.opened == [("j", 1.0), ("j", 1.0)]
    assert len(hosts.started) == 2


def test_a_container_on_another_cpu_is_replaced_and_its_host_still_runs():
    hosts = Hosts()
    start(hosts)
    first = hosts.started[0]
    first.done, first.said = True, WRONG
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
        call.done, call.said = True, WRONG
        running = hosts.instances("j")
    assert running == []
    assert len(hosts.started) == modal_hosts.REROLLS + 1


def test_deleting_a_host_cancels_its_call():
    hosts = Hosts()
    start(hosts)
    hosts.delete([("h-0", "modal")])
    assert hosts.started[0].cancelled and hosts.instances("j") == []
