"""The GGS client (examples/othello/ggs_client): its parser on transcripts recorded from a local
GGS server (tests/fixtures/ggs), its session logic driven by those transcripts and by hand, its
accept policy, and the player bridge with a fake player. No server, no network.

A transcript is the client's own record: a line per line sent (">") or received ("<"), after
the seconds since it started. Its moves were played by `FirstLegal`, so a session replaying
what it received must send the same moves.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "examples" / "othello"))
sys.path.insert(0, str(ROOT / "examples" / "othello" / "ggs_client"))

import ggs_protocol as gp
from engines import Board
from ggs_client import GameState, Job, Policy, Session
from ggs_player import (
    Decision,
    FirstLegal,
    GameView,
    TimePolicy,
    colored_moves,
    eval_discs,
    replay,
    value_discs,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ggs"
TRANSCRIPTS = sorted(FIXTURES.glob("*.transcript"))


def received(path: Path) -> str:
    return "".join(
        line.split(" < ", 1)[1] + "\n"
        for line in path.read_text().splitlines()
        if " < " in line
    )


def sent(path: Path) -> list[str]:
    return [
        line.split(" > ", 1)[1]
        for line in path.read_text().splitlines()
        if " > " in line
    ]


def plays(commands: list[str]) -> dict[str, list[str]]:
    """Each game's moves, from `tell /os play <game> <move>/<eval>/<seconds>` commands."""
    out: dict[str, list[str]] = {}
    for c in commands:
        if c.startswith("tell /os play "):
            _, _, _, game, move = c.split()
            out.setdefault(game, []).append(move.split("/")[0].upper())
    return out


# --- the parser ------------------------------------------------------------------------------


def test_clock_spec():
    c = gp.ClockSpec.parse("01:00//02:00")
    assert (c.ini, c.inc, c.ext) == (60, 0, 120)
    assert str(c) == "01:00//02:00"
    c = gp.ClockSpec.parse("05:00,N0/00:05/1:00:00,N10")
    assert (c.ini, c.inc, c.ext, c.ext_moves) == (300, 5, 3600, 10)
    assert not c.ini_loss and not c.ext_add
    assert gp.ClockSpec.parse(str(c)) == c
    with pytest.raises(ValueError):
        gp.ClockSpec.parse("01:00")


def test_clock_state():
    s = gp.ClockState.parse("00:59,1:0//02:00,1:0")
    assert (s.left, s.moves, s.inc, s.ext_left) == (59, 1, 0, 120)
    s = gp.ClockState.parse("01:00,0:0/00:05,0:1/02:00,0:0")
    assert (s.left, s.inc) == (60, 5)


def test_match_type():
    t = gp.MatchType.parse("s8r14")
    assert (t.synchro, t.size, t.random_discs, t.komi, t.games) == (
        True,
        8,
        14,
        False,
        2,
    )
    t = gp.MatchType.parse("8k")
    assert (t.synchro, t.komi, t.games) == (False, True, 1)
    assert gp.MatchType.parse("10b").color == "b"
    assert gp.MatchType.parse("88kr20").size == 88


def test_moves_and_squares():
    assert gp.square(0) == "A1" and gp.square(63) == "H8"
    assert gp.square_index("d3") == 19 and gp.square_index("pass") is None
    m = gp.MoveInfo.parse("c3/19.00/0.21")
    assert (m.square, m.eval, m.seconds) == (18, 19.0, 0.21)
    assert gp.MoveInfo.parse("F2") == gp.MoveInfo(13)
    assert gp.MoveInfo.parse("pass/0.00/0.01").square is None
    assert (
        gp.move_command(".1.0", 18, -3.456, 1.234) == "tell /os play .1.0 C3/-3.46/1.23"
    )
    assert gp.move_command(".1", None, 0, -1) == "tell /os play .1 pass/0.00/0.00"


def test_requests():
    r = gp.parse(["/os: +   .1 1720.0 edax     01:00//02:00    s8r14 U 1720.0 alice"])
    assert isinstance(r, gp.Request)
    assert (r.id, r.added, r.asker, r.opponent, r.rated) == (
        ".1",
        True,
        "edax",
        "alice",
        False,
    )
    assert r.clock == gp.ClockSpec(60, 0, 120) and r.other("alice") == "edax"
    r = gp.parse(
        ["/os: -   .2 1720.0 tzbot    01:00//02:00    s8r14 S 1720.0 edax     .1"]
    )
    assert isinstance(r, gp.Request)
    assert (r.added, r.rating_mode, r.stored_id, r.opponent_clock) == (
        False,
        "S",
        ".1",
        None,
    )
    r = gp.parse(["/os: + .3 1800.0 bob 05:00//02:00 8 R 1700.0 alice 03:00//02:00"])
    assert isinstance(r, gp.Request)
    assert r.rated and r.clock_for("alice").ini == 180 and r.clock_for("bob").ini == 300


def test_match_and_game_ends():
    e = gp.parse(["/os: - match .2 2038 edax 1402 tzbot s8r14 R +64.00  .1"])
    assert isinstance(e, gp.MatchEnd) and e.result == 64 and e.rated
    e = gp.parse(["/os: - match .4 1720 edax 1720 alice 8 U alice left"])
    assert isinstance(e, gp.MatchEnd) and e.result is None and e.reason == "alice left"
    e = gp.parse(["/os: end .1.1 ( edax vs. alice ) +64.00"])
    assert isinstance(e, gp.GameEnd) and e.result == 64
    e = gp.parse(["/os: end .1.0 ( edax vs. alice ) aborted"])
    assert isinstance(e, gp.GameEnd) and e.result is None and e.reason == "aborted"
    e = gp.parse(["/os: fatal-timeout .1.1 alice (*)"])
    assert e == gp.Timeout(".1.1", "alice", "*", True)
    assert gp.parse(["/os: undo .3 bob is asking"]) == gp.Offer("undo", ".3", "bob")


def test_prompts_and_errors():
    assert gp.parse([": Enter login (yours, or one you'd like to use)."]) == gp.Prompt(
        "login"
    )
    assert gp.parse([": Enter your password."]) == gp.Prompt("password")
    assert gp.parse(["READY"]) == gp.Ready() and gp.parse(["ALERT"]) == gp.Ready(
        alert=True
    )
    assert gp.parse([": ERR Password is incorrect."]) == gp.ServerError(
        "Password is incorrect."
    )
    assert gp.parse([": + /os 1"]) == gp.ServiceStatus("/os", True)
    assert gp.parse(["/os: ERR .1.0 not your turn"]) == gp.ServiceError(
        ".1.0 not your turn"
    )
    assert isinstance(gp.parse(["edax: hello"]), gp.Other)
    with pytest.raises(gp.ProtocolError):
        gp.parse(["/os: update .1.0 s8r14 K?", "|  1: F7"])


@pytest.mark.parametrize("path", TRANSCRIPTS, ids=lambda p: p.stem)
def test_transcripts_parse_alike_in_any_chunks(path):
    """Every message reads, and reads the same however the stream is cut into reads."""
    text = received(path)
    whole = list(gp.parse_stream(text))
    assert not any(isinstance(e, gp.ServiceError) for e in whole)
    assembler = gp.Assembler()
    pieces = []
    for i in range(0, len(text), 97):
        pieces += [gp.parse(m) for m in assembler.feed(text[i : i + 97])]
    pieces += [gp.parse(m) for m in assembler.flush()]
    assert pieces == whole


@pytest.mark.parametrize("path", TRANSCRIPTS, ids=lambda p: p.stem)
def test_boards_follow_the_moves(path):
    """Each update's board is the game's start replayed through its moves, passes and all;
    a join's move list replays to its board."""
    games: dict[str, tuple[tuple[int, int, str], list]] = {}
    boards = 0
    for e in gp.parse_stream(received(path)):
        if not isinstance(e, gp.BoardMessage):
            continue
        if e.join:
            start_board = e.start_board if e.number else e.board
            start_color = e.start_to_move if e.number else e.to_move
            black, white = gp.bitboards(start_board)
            start = (black, white, gp.color_letter(start_color))
            games[e.game_id] = (start, [m.square for m in e.moves])
            assert len(e.moves) == e.number
        else:
            start, moves = games[e.game_id]
            assert e.last_move is not None and e.number == len(moves) + 1
            moves.append(e.last_move.square)
        start, moves = games[e.game_id]
        assert replay(start, moves).discs == list(gp.bitboards(e.board))
        boards += 1
    assert boards > 20


def test_join_with_moves():
    """A resumed match's join: the clocks as set, the start, every move, then the state."""
    joins = [
        e
        for e in gp.parse_stream(received(FIXTURES / "resume.transcript"))
        if isinstance(e, gp.BoardMessage) and e.join
    ]
    assert len(joins) == 2 and all(j.number > 0 for j in joins)
    join = joins[0]
    assert len(join.moves) == join.number and len(join.start_board) == 64
    assert all(isinstance(p.clock, gp.ClockState) for p in join.players)
    assert {p.name for p in join.players} == {"edax", "tzbot"}


# --- the session ----------------------------------------------------------------------------


def make_session(policy: Policy | None = None, login: str = "tzbot", **kwargs):
    commands: list[str] = []
    jobs: list[Job] = []
    session = Session(
        login,
        "secret",
        policy or Policy(types=("s8r14", "8k")),
        send=commands.append,
        ask_move=jobs.append,
        **kwargs,
    )
    return session, commands, jobs


@pytest.mark.parametrize("path", TRANSCRIPTS, ids=lambda p: p.stem)
def test_session_replays_the_recorded_moves(path):
    """Fed what the server sent in a recorded run, with the same player deciding at once, the
    session sends the moves the client sent then."""
    player = FirstLegal()
    session, commands, _ = make_session()
    session.ask_move = lambda job: session.decided(job, player.decide(job.view, 1))
    session.logged_in = True
    for e in gp.parse_stream(received(path)):
        session.receive(e)
    assert plays(commands) == plays(sent(path))
    assert session.desyncs == 0 and not session.errors
    assert all(r[3] is not None for r in session.results)


def test_login_and_settings():
    session, commands, _ = make_session(aform="rand & discs == 14")
    session.receive(gp.Prompt("login"))
    session.receive(gp.Prompt("password"))
    assert commands[:2] == ["tzbot", "secret"]
    for expected in ("chann %", "tell /os client -", "tell /os trust +",
                     "tell /os aform rand & discs == 14", "tell /os open 2",
                     "tell /os continue"):  # fmt: skip
        assert expected in commands
    commands.clear()
    session.receive(gp.ServerError("Password is incorrect."))
    session.receive(gp.Prompt("login"))
    assert session.fatal and commands == []


def test_keepalive():
    now = [0.0]
    session, commands, _ = make_session(clock=lambda: now[0], keepalive=60)
    session.receive(gp.Prompt("password"))
    commands.clear()
    session.tick()
    assert commands == []
    now[0] = 61
    session.tick()
    assert commands == ["tell /os open 2", "tell /os continue"]


def request(**changes) -> gp.Request:
    fields = {
        "id": ".7",
        "added": True,
        "asker": "edax",
        "asker_rating": 1720,
        "clock": gp.ClockSpec.parse("01:00//02:00"),
        "type": gp.MatchType.parse("s8r14"),
        "rating_mode": "R",
        "opponent": "tzbot",
        "opponent_rating": 1720,
    }
    fields.update(changes)
    return gp.Request(**fields)


def test_policy():
    policy = Policy(types=("s8r14", "8"), max_games=2, blocked=frozenset({"pest"}))
    assert policy.decide(request(), "tzbot", 0) == (True, "")
    assert not policy.decide(request(), "tzbot", 1)[0]  # a synchro match needs 2 games
    assert policy.decide(request(type=gp.MatchType.parse("8")), "tzbot", 1)[0]
    assert not policy.decide(request(type=gp.MatchType.parse("s8r20")), "tzbot", 0)[0]
    anywhere = Policy(types=("8a", "10", "s8r14"))
    assert not anywhere.decide(request(type=gp.MatchType.parse("8a")), "tzbot", 0)[0]
    assert not anywhere.decide(request(type=gp.MatchType.parse("10")), "tzbot", 0)[0]
    assert not policy.decide(request(asker="pest"), "tzbot", 0)[0]
    assert not policy.decide(
        request(clock=gp.ClockSpec.parse("00:30//02:00")), "tzbot", 0
    )[0]
    other = gp.ClockSpec.parse("05:00//02:00")
    assert not policy.decide(request(opponent_clock=other), "tzbot", 0)[0]
    assert not Policy(rated="unrated").decide(request(), "tzbot", 0)[0]
    assert not Policy(rated="rated").decide(request(rating_mode="U"), "tzbot", 0)[0]
    # a stored match resumes whatever its settings
    stored = request(rating_mode="S", type=gp.MatchType.parse("s8r20"), stored_id=".1")
    assert policy.decide(stored, "tzbot", 0)[0]
    assert not Policy(resume_stored=False).decide(stored, "tzbot", 0)[0]
    assert not policy.decide(
        request(clock=gp.ClockSpec.parse("05:00,N0//02:00")), "tzbot", 0
    )[0]


def test_requests_answered():
    session, commands, _ = make_session()
    session.receive(request())
    session.receive(request(id=".8", type=gp.MatchType.parse("10")))
    session.receive(request(id=".9", asker="tzbot", opponent="edax"))  # our own
    session.receive(request(id=".10", opponent="someone"))  # not ours
    assert commands == ["tell /os accept .7", "tell /os decline .8"]
    session.drain()
    commands.clear()
    session.receive(request(id=".11"))
    assert commands == ["tell /os decline .11"]


def board_message(game_id=".3", number=0, to_move="*", me="*", board=None, last=None):
    board = board or Board().text()[:64]
    colors = {"*": ("tzbot", "edax"), "O": ("edax", "tzbot")}[me]
    clock = gp.ClockState(55, number // 2, 0, 120)
    players = (
        gp.PlayerLine(colors[0], 1720, "*", clock),
        gp.PlayerLine(colors[1], 1720, "O", clock),
    )
    return gp.BoardMessage(
        join=number == 0, game_id=game_id, type=gp.MatchType.parse("8"), komi=0.0,
        number=number, last_move=last, players=players, board=board, to_move=to_move,
    )  # fmt: skip


def start_game(session, me="*"):
    session.receive(
        gp.MatchStart(
            ".3", ("tzbot", "edax"), (1720, 1720), gp.MatchType.parse("8"), False
        )
    )
    session.receive(board_message(me=me))


def test_moves_asked_once_and_sent_once():
    now = [100.0]
    session, commands, jobs = make_session(clock=lambda: now[0])
    session.logged_in = True
    start_game(session)
    assert len(jobs) == 1 and jobs[0].number == 0
    view = jobs[0].view
    assert view.color == "b" and view.seconds_left == 55 and view.extension == 120
    session.receive(board_message())  # the same board again: no second ask
    assert len(jobs) == 1
    now[0] = 102.5
    session.decided(jobs[0], Decision(19, 1.5, 2.0))
    assert commands[-1] == "tell /os play .3 D3/1.50/2.50"
    session.decided(
        jobs[0], Decision(19, 1.5, 2.0)
    )  # sent already: the move's no longer ours
    assert commands.count("tell /os play .3 D3/1.50/2.50") == 1


def test_stale_decision_dropped():
    session, commands, jobs = make_session()
    session.logged_in = True
    start_game(session)
    board = Board()
    board.play(19)
    session.receive(
        board_message(
            number=1, to_move="O", board=board.text()[:64], last=gp.MoveInfo(19)
        )
    )
    session.decided(jobs[0], Decision(19, 0, 1))
    assert not any(c.startswith("tell /os play") for c in commands)


def test_decision_held_over_a_reconnect():
    session, commands, jobs = make_session()
    session.logged_in = True
    start_game(session)
    session.disconnected()
    session.decided(jobs[0], Decision(19, 0, 1))
    assert not any(c.startswith("tell /os play") for c in commands)
    session.receive(gp.Prompt("password"))
    session.receive(board_message())  # the server shows the game again
    assert sum(c.startswith("tell /os play .3 D3") for c in commands) == 1
    assert len(jobs) == 1


def test_match_lost_to_a_dropped_connection():
    now = [0.0]
    session, _commands, _ = make_session(clock=lambda: now[0])
    session.logged_in = True
    start_game(session)
    session.disconnected()
    session.receive(gp.Prompt("password"))
    now[0] = 20
    session.tick()
    assert not session.matches and not session.games
    assert session.results == [
        (".3", "edax", "8", None, "adjourned: our connection dropped")
    ]


def test_decision_for_a_match_gone_while_away():
    """A move decided before the connection dropped isn't sent to a match the server may have
    ended meanwhile; "not found" ends it."""
    session, commands, jobs = make_session()
    session.logged_in = True
    start_game(session)
    session.disconnected()
    session.receive(gp.Prompt("password"))
    session.decided(jobs[0], Decision(19, 0, 1))
    assert not any(c.startswith("tell /os play") for c in commands)
    session.receive(gp.ServiceError("match '.3' not found."))
    assert not session.matches and not session.errors and not session.pending
    assert session.results[-1][3] is None


def test_gap_in_moves_resyncs():
    session, commands, _ = make_session()
    session.logged_in = True
    start_game(session)
    board = Board()
    for move in (19, 18, 17):
        board.play(move)
    session.receive(
        board_message(
            number=3, to_move="O", board=board.text()[:64], last=gp.MoveInfo(17)
        )
    )
    assert "tell /os moves .3" in commands


def test_error_after_a_move_resyncs():
    session, commands, jobs = make_session()
    session.logged_in = True
    start_game(session)
    session.decided(jobs[0], Decision(19, 0, 1))
    session.receive(gp.ServiceError(".3 illegal move"))
    assert "tell /os moves .3" in commands
    session.receive(board_message())  # the join it asked for: the move's asked again
    assert len(jobs) == 2


def test_offers_not_agreed():
    session, commands, _ = make_session()
    session.logged_in = True
    start_game(session, me="O")
    commands.clear()
    for kind in ("undo", "abort", "score"):
        session.receive(gp.Offer(kind, ".3", "edax"))
    assert commands == []


def test_our_timeout_puts_us_on_the_extension():
    session, _, _jobs = make_session()
    session.logged_in = True
    start_game(session)
    session.receive(gp.Timeout(".3", "tzbot", "*", False))
    game: GameState = session.games[".3"]
    view = session.view(game)
    assert view.extended and view.extension == 55


def test_komi_first_move():
    """Before a komi game's colours are set, both players send a first move."""
    session, _commands, jobs = make_session()
    session.logged_in = True
    msg = board_message()
    msg.type = gp.MatchType.parse("8k")
    msg.komi = None
    clock = msg.players[0].clock
    msg.players = (
        gp.PlayerLine("edax", 1720, "?", clock),
        gp.PlayerLine("tzbot", 1720, "?", clock),
    )
    session.receive(msg)
    assert len(jobs) == 1 and jobs[0].view.komi and jobs[0].view.color == "b"


def test_match_end_and_result():
    session, commands, _ = make_session(max_matches=1)
    session.logged_in = True
    start_game(session)
    session.receive(
        gp.MatchEnd(".3", ("edax", "tzbot"), gp.MatchType.parse("8"), False, 10.0)
    )
    assert session.results == [(".3", "edax", "8", -10.0, "")]
    assert session.done and commands[-1] == "tell /os open 0"


# --- the player's side ----------------------------------------------------------------------


def test_replay_with_passes():
    # a position where white, to move, has no move: the pass is a move on GGS
    black = sum(1 << s for s in range(64) if s not in (0, 1, 63))
    white = 1 << 1
    start = (black, white, "w")
    assert not Board.at(*start).legal()
    board = replay(start, [None, 0])
    assert board.discs[0] >> 0 & 1 and not board.discs[1]
    assert colored_moves(start, [None, 0]) == [("b", 0)]


def test_time_policy():
    view = GameView(
        ".1", (0, 0, "b"), (), 1 << 27 | 1 << 36, 1 << 28 | 1 << 35, "b", 60.0, 120.0
    )
    policy = TimePolicy(reserve=1.5, reserve_share=0.03)
    assert policy.seconds_left(view) == pytest.approx(60 * 0.97 - 1.5)
    on_ext = GameView(*view.__dict__.values())
    object.__setattr__(on_ext, "extended", True)
    assert policy.seconds_left(on_ext) == 0.0
    assert policy.extension_move(on_ext) == pytest.approx((120 - 1.5) * 0.1)


def test_evals_in_discs():
    assert value_discs(0) == 0 and value_discs(1) <= 64 and value_discs(-1) >= -64
    assert value_discs(0.5) > 0 > value_discs(-0.5)
    assert eval_discs(("margin", -6.0)) == -6
    assert eval_discs(("wdl", 1.0)) == 2 and eval_discs(None, 3.0) == 3


class FakeNetPlayer:
    """`cpu_match.NetPlayer`'s interface: plays the first legal move, recording what it's
    told; a pass is implied by a colour moving out of turn, as there."""

    def __init__(self, seed: int):
        self.seed = seed
        self.calls: list[tuple] = []
        self.board = Board()
        self.evaluate = None
        self.taper = 0
        self.last_value: tuple[str, float] | None = None

    def set_position(self, black: int, white: int, color: str) -> None:
        self.calls.append(("set_position", black, white, color))
        self.board = Board.at(black, white, color)

    def play(self, color: str, move: int) -> None:
        self.calls.append(("play", color, move))
        if color != self.board.color:
            self.board.black_to_move = not self.board.black_to_move
        self.board.play(move)

    def move(self, color: str, seconds_left: float) -> int:
        self.calls.append(("move", color, round(seconds_left, 6), self.evaluate))
        if color != self.board.color:
            self.board.black_to_move = not self.board.black_to_move
        move = self.board.legal()[0]
        self.board.play(move)
        self.last_value = ("q", 0.5)
        return move


def fake_bridge(cores=4):
    players: list[FakeNetPlayer] = []
    engines: list[int] = []

    def new_player(seed):
        players.append(FakeNetPlayer(seed))
        return players[-1]

    def make_engine(n):
        engines.append(n)
        return f"engine{len(engines)}x{n}"

    from ggs_player import PlayerBridge

    return PlayerBridge(new_player, make_engine, cores), players, engines


def view_after(moves, start=None, game_id=".1.0", seconds=60.0, **kwargs):
    board = Board()
    start = start or (board.discs[0], board.discs[1], "b")
    board = replay(start, moves)
    return GameView(
        game_id, start, tuple(moves), board.discs[0], board.discs[1], board.color,
        seconds, **kwargs,
    )  # fmt: skip


def test_bridge_keeps_its_tree():
    bridge, players, engines = fake_bridge()
    moves: list = []
    decisions = []
    for _ in range(3):
        decision = bridge.decide(view_after(moves), 4)
        decisions.append(decision)
        moves.append(decision.square)
        # the opponent's reply
        moves.append(replay(view_after(moves).start, moves).legal()[0])
    assert len(players) == 1
    calls = players[0].calls
    assert [c[0] for c in calls].count("set_position") == 1
    # told each opponent move once, never its own
    assert [c for c in calls if c[0] == "play"] == [
        ("play", "w", moves[1]),
        ("play", "w", moves[3]),
    ]
    assert engines == [4]  # one engine, reused
    assert decisions[-1].eval == value_discs(0.5)


def test_bridge_starts_again_when_the_game_differs():
    bridge, players, _ = fake_bridge()
    d = bridge.decide(view_after([]), 4)
    other = next(m for m in Board().legal() if m != d.square)
    reply = replay(view_after([]).start, [other]).legal()[0]
    bridge.decide(view_after([other, reply]), 4)
    assert len(players) == 2


def test_bridge_follows_the_server_board():
    """A board that isn't the replay of the moves: the player starts from the board."""
    bridge, players, _ = fake_bridge()
    view = view_after([19, 18])
    wrong = GameView(view.game_id, view.start, view.moves, view.black ^ (1 << 0), view.white,
                     view.color, 60.0)  # fmt: skip
    bridge.decide(wrong, 4)
    assert ("set_position", wrong.black, wrong.white, "b") in players[-1].calls


def test_bridge_passes_without_searching():
    bridge, players, _ = fake_bridge()
    black = sum(1 << s for s in range(64) if s not in (0, 1, 63))
    start = (black, 1 << 1, "w")
    d = bridge.decide(GameView(".2", start, (), black, 1 << 1, "w", 60.0), 4)
    assert d.square is None and not players


def test_bridge_time_and_cores():
    bridge, players, engines = fake_bridge(cores=8)
    view = view_after([], seconds=60.0)
    bridge.decide(view, 3)
    assert players[0].calls[-1][2] == pytest.approx(bridge.time.seconds_left(view))
    assert players[0].calls[-1][3] == "engine1x3" and engines == [3]
    bridge.decide(
        view_after([], game_id=".2.0", seconds=0.0, extension=100.0, extended=True), 16
    )
    seconds = players[-1].calls[-1][2]
    # move_budget's share of that over the moves left is a tenth of the extension, less reserve
    assert seconds == pytest.approx((100 - 1.5) * 0.1 * 61 / 2 / 0.95)
    assert engines == [3, 8]


def test_observe_sees_every_board():
    """A player that ponders is shown each board of our games, whoever's to move."""
    seen: list[GameView] = []
    session, _, jobs = make_session(observe=seen.append)
    session.logged_in = True
    start_game(session, me="O")  # the opponent to move
    assert len(seen) == 1 and seen[0].mine == "w" and seen[0].color == "b" and not jobs
    board = Board()
    board.play(19)
    session.receive(
        board_message(
            number=1, to_move="O", me="O", board=board.text()[:64], last=gp.MoveInfo(19)
        )
    )
    assert len(seen) == 2 and seen[1].moves == (19,) and len(jobs) == 1
