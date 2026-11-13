# GGS client

Plays our Othello player on GGS, Michael Buro's Generic Game Server
(<https://skatgame.net/mburo/ggs/>, `skatgame.net:5000`), where the strongest Othello programs
play each other. Most of its games are synchro matches, `s8r14` at `01:00//02:00`: two games
at once from the same random 14-disc start, colours swapped, each with a minute a player and a
two-minute extension, scored by the mean of the two games' disc results.

| File | |
|---|---|
| `ggs_client.py` | The client: logs in, accepts the matches its policy allows, plays them; keeps the connection alive and reconnects |
| `ggs_protocol.py` | What the server says, parsed, and the commands sent to it (no sockets) |
| `ggs_player.py` | Our player's side: the CPU net player of `clock_match.py`'s `net:`, from any start position, on GGS clocks |
| `setup_ggs.py` | Downloads (hash-pinned) and builds a private copy of the GGS server for testing |
| `ggs_server.py` | Runs that server on 127.0.0.1 in a directory of its own; registers logins |
| `ggs_e2e.py` | The client end to end against Edax's GGS bot on the private server |

The server's start positions, clock and scoring are modelled in `../ggs.py` (and played locally,
without a server, by `../clock_match.py`).

## Trying it locally

Build the server once (needs `g++` and gdbm's header, from `libgdbm-dev`; without root,
`apt-get download libgdbm-dev` into an empty directory and `dpkg-deb -x` it there, then pass
its `usr/include`):

```
uv run python -I examples/othello/ggs_client/setup_ggs.py --gdbm-include /path/to/usr/include
```

It lands in `~/.cache/turbozero/ggs` (`$GGS_DIR`). Then the end-to-end test: a private server,
our client and Edax (built by `../setup_edax.sh`) logged in, Edax asking us for matches. Pick
idle cores (whole cores, their SMT siblings idle too: here CPU n and n+12 share a core), ours
apart from Edax's:

```
JAX_PLATFORMS=cpu uv run examples/othello/ggs_client/ggs_e2e.py \
    --net 4x64-tandem-s13t-511s-solve20past3.5-e339-avg4 \
    --client-cores 4,5,16,17 --edax-cores 6,18
```

It plays 3 synchro matches (`--matches`), the second rated; one where our connection is
dropped mid-game; one where our client is killed mid-game and started again; and a komi game
(`8k`). It passes if every match finished with a result and the client saw no protocol errors,
no timeouts of its own and no board that disagreed with its replay of the moves; about 15
minutes. `--fake-player` plays the first legal move instead of the network: the protocol alone,
in about 2 minutes. Everything (transcripts, logs, the server's runtime) stays in the directory
it prints.

To run the pieces by hand:

```
uv run examples/othello/ggs_client/ggs_server.py start --run /tmp/ggs-run --port 15000
uv run examples/othello/ggs_client/ggs_server.py register --run /tmp/ggs-run tzbot PASSWORD
printf PASSWORD > ~/.ggs-tzbot && chmod 600 ~/.ggs-tzbot
JAX_PLATFORMS=cpu taskset -c 4,5,16,17 uv run examples/othello/ggs_client/ggs_client.py \
    --port 15000 --login tzbot --password-file ~/.ggs-tzbot \
    --player net=4x64-tandem-s13t-511s-solve20past3.5-e339-avg4,batch=8,solve=20 \
    --transcript /tmp/ggs-run/tzbot.transcript
uv run examples/othello/ggs_client/ggs_server.py stop --run /tmp/ggs-run
```

The server binds 127.0.0.1 only. A login that isn't registered needs no account and takes any
password, every time. `register` gives it one (its password then checked) and lets it play
rated games. `ggs_server.py admin --run DIR COMMAND` runs a command as root.

## On the public server

Not done yet: logging in claims a name there, and that's for us to decide.

- **Login.** 2 to 8 characters, not starting with `.` or `_`, no `,` or `\`. A name that
  isn't registered logs in with any password and plays unrated games at once, but anyone can
  log in as it whenever it's offline. Rated play, and owning the name, need the account
  **registered**: give your name and email to an admin (`group _admin` lists them; Michael
  Buro is `mic`). Until then pass `--rated unrated`, or the server says "you are not
  registered" (logged, harmless).
- **Running.**

  ```
  JAX_PLATFORMS=cpu taskset -c CORES uv run examples/othello/ggs_client/ggs_client.py \
      --host skatgame.net --port 5000 --login NAME --password-file ~/.ggs-NAME \
      --player net=4x64-tandem-s13t-511s-solve20past3.5-e339-avg4,batch=8,solve=20 \
      --types s8r14,s8r16,s8r18 --transcript ~/ggs-NAME.transcript --results ~/ggs-NAME.json
  ```

  Then people (or bots) ask it for matches: `tell /os ask s8r14 01:00//02:00 NAME`. To ask
  others, `--ask "s8r14 01:00//02:00 OPPONENT"` (repeatable) asks one at random whenever
  it's idle. Many bots there accept automatically with a formula (see `--aform`).
- **Stopping.** SIGINT or SIGTERM once: no new matches (`open 0`), the ones under way
  finished, then it logs out. Twice: at once.
- **A dropped connection.** On the server, a player whose connection drops leaves its matches,
  unless in a tournament: an unrated match is over, unscored; a rated one is stored, its clocks
  stopped. The client reconnects (backoff from 1 to 60 s) and sends `continue`, which offers to
  resume our stored matches; it accepts the opponent's offer to resume (`S` requests). A
  tournament's matches go on meanwhile, our clock running, and come back with their boards.

## Settings

`ggs_client.py --help` lists them with their defaults.

| Setting | Default | |
|---|---|---|
| `--host`, `--port` | 127.0.0.1, 5000 | The server (plain port; the public one is `skatgame.net 5000`) |
| `--login`, `--password-file` | | The login, and a file holding its password (mode 600) |
| `--player` | | The player: `net=NAME` (a network of the league's store, quantized for the native engine), `device` (`cpu`, the default; `gpu`, the GPU player of `../gpu_play.py`, its own options added), and `clock_match.py`'s `net:` options: `batch` (leaves each worker evaluates at once), `solve=N` (solve exactly from N empties), `solve_exact` (here 1 by default: GGS scores discs, so a won endgame should win by the most), `solve_share`, `taper`, and the search's (`c_init`, `virtual_loss`, ...) |
| `--cores` | all it may run on | Cores the player uses (run it under `taskset`) |
| `--types` | `s8r14,s8r16,s8r18,8` | Match types it accepts; 8x8 non-anti only, whatever this says |
| `--min-seconds`, `--max-seconds` | 60, 900 | Our clock's initial time, accepted range |
| `--max-increment`, `--min-extension` | 60, 0 | Accepted increments and extensions |
| `--handicap-clocks` | off | Accept matches where the two clocks differ |
| `--rated` | `any` | `rated` or `unrated` only; also its `rated +/-` setting |
| `--max-games` | 2 | Games at once (a synchro match is 2), the server's `open` |
| `--opponents`, `--blocked` | | Only these opponents; never these (comma-separated) |
| `--aform` | | A server-side auto-accept formula (`tell /os help formula`), e.g. `rand & discs == 14 & mt1 == 60`; the server then accepts those itself, and the client's policy answers the rest |
| `--ask` | | `"TYPE CLOCK OPPONENT"`: asked when idle (repeatable) |
| `--matches` | 0 | Quit after this many matches (0: never) |
| `--keepalive` | 60 | Seconds between `open N` + `continue` (keeps the connection and resumes stored matches); 200 s of silence from the server means a dead connection |
| `--transcript`, `--results` | | Everything sent and received (password left out); the results as JSON at exit |
| `--fake-player` | off | First legal move, at once: for testing the protocol |

Fixed in the code, `ggs_player.TimePolicy`: 1.5 s plus 3% of the clock held back (the server
prints clocks in whole seconds; network delay isn't charged: with `trust +` it charges the time
we report, if no more than 4 s under its own count); with `solve=N`, 4 s (or 5%) kept for the
solved endgame and the rest spread evenly over our moves until N empties; past a soft timeout,
a tenth of the extension a move. `ggs_player.DISC_SCALE`: evals sent are `6.5 × atanh(value)` discs for a
search's root value, a solve's margin when solved.

## How it plays

- **The server is the source of truth.** Each game's start and moves come from its `join`
  (sent on starting, and on rejoining) and `update` messages, checked against a replay of the
  moves; a gap or a mismatch, or an error after one of our moves, asks for the game whole
  again (`tell /os moves`).
- **Synchro.** The server plays the two games in rounds: a round's clocks start once both of
  its moves are in, and the opponent's move in the other game shows only then. Passes are moves
  (`pass`), so each player moves in one game a round, and our cores all go to it. If two of our
  games want a move at once (two matches), each search gets half.
- **Tree reuse.** A player per game, started from the game's start (`set_position`), is told
  every move, so the tree carries over between our moves; if what it was told disagrees with
  the server it starts again from the server's board.
- **Time.** A move's search time is the usable clock over our moves left until the solve
  (`TimePolicy.planned_left`, through `cpu_match.move_budget`); positions with `solve` or fewer
  empties are solved exactly (`solve_share` of the clock). Planning to the end of the game
  instead left 26 of 60 s unused at `01:00//02:00` (from 20 empties a solve takes about a
  second); planned to the solve, about 6.
- **Komi games** (`K`): before colours are set, both players send a first move with the result
  they expect for the player to move; ours is the search's eval, in discs.
- **Offers.** Undo, abort and score offers are left unanswered, which declines them.

## Another player

The client sees its player only through `ggs_player.Bridge`: `decide(view, cores)` for our
move (on a thread a game), `observe(view)` for every board of our games, whoever's to move (to
ponder; it must return at once), `forget(game_id)` at a game's end, `close()`. A `GameView` is a
game as the server has it: its start, every move since, the position, our colour and clock.
`make_bridge` picks it from `--player`: the CPU's `NetBridge` (a `cpu_match.NetPlayer` a game,
driven by `PlayerBridge`, which doesn't ponder), or with `device=gpu`,
`gpu_play.make_ggs_bridge`, whose trees share one batch stream on the GPU and ponder the game
the opponent's moving in.

## Tests

`tests/test_ggs_client.py` (in the suite): the parser on transcripts recorded from a local
server (`tests/fixtures/ggs`: a synchro match with passes, a stored match resumed, a komi
game), a session replaying them (it must send the moves the recorded client sent), the policy,
and the session's reconnects, resyncs and timeouts by hand. `ggs_e2e.py` is the end-to-end
test, by hand.
