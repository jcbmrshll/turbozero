"""A continuously running Elo league for two-player games: players are networks at a search
budget and external engines, all rated from every game they've played. A game joins by
implementing `game.LeagueGame` (see `examples/othello/elo_league.py`).

- `store`: every player and game, permanently, in a SQLite file; ratings come from these
- `ratings`: fits ratings and their uncertainty to all the games
- `scheduler`: picks the next batch of games, where it reduces uncertainty most per second
- `match`: plays a batch, each player with its own search, parameters and tree
- `gate`: says whether anyone else needs the GPU
- `worker`: plays batches while the GPU is free, giving it up as soon as anyone else needs it
- `dashboard`: the web page
"""
