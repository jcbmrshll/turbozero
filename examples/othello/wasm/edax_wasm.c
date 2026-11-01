/* Edax's search as functions for JavaScript: built by ../setup_wasm_engines.sh, with Edax's
 * sources on the include path. Only the search is compiled, as in Edax's all.c, without its
 * book or protocols. One thread: a search holds the worker it runs in until it ends, so
 * stopping one means terminating the worker. */

#include "options.c"
#include "util.c"
#include "stats.c"
#include "bit.c"
#include "board.c"
#include "move.c"
#include "eval.c"
#include "hash.c"
#include "ybwc.c"
#include "search.c"
#include "endgame.c"
#include "midgame.c"
#include "root.c"

#include <emscripten.h>

/* in the protocols left out */
Log engine_log, xboard_log;

/* Edax has this only for SSE and Neon: their version, from its generic tables */
int get_edge_stability(const unsigned long long P, const unsigned long long O)
{
	unsigned int packedstable = edge_stability[((unsigned int) P & 0xff) * 256 + ((unsigned int) O & 0xff)]
	  | edge_stability[(unsigned int) (P >> 56) * 256 + (unsigned int) (O >> 56)] << 8
	  | edge_stability[packA1A8(P) * 256 + packA1A8(O)] << 16
	  | edge_stability[packH1H8(P) * 256 + packH1H8(O)] << 24;
	return bit_count_32(packedstable & 0xffff7e7e);
}

static Search *search;

EM_JS(void, report, (int depth, int selectivity, int move, int score, double nodes), {
  Module.onResult({ depth, percent: [73, 87, 95, 98, 99, 100][selectivity], move, score, nodes });
});

static void observer(Result *result)
{
	report(result->depth, result->selectivity, result->move, result->score, (double) search_count_nodes(search));
}

/* Loads the weights from data/eval.dat in the virtual filesystem; its hash table is the size it has
 * natively. */
EMSCRIPTEN_KEEPALIVE void engine_init(void)
{
	options.n_task = 1;
	options.verbosity = 2;
	options.info = 0;
	options_bound();
	bit_init();
	edge_stability_init();
	statistics_init();
	eval_open(options.eval_file);
	search_global_init();
	search = (Search*) mm_malloc(sizeof *search);
	search_init(search);
	search_resize_hashtable(search);
	search_set_observer(search, observer);
	search->options.verbosity = 2;
}

/* A fresh search of a position, as a new Edax process would make it, for the player to move:
 * `player` and `opponent` are their discs, a1 the lowest bit. */
EMSCRIPTEN_KEEPALIVE void engine_search(unsigned long long player, unsigned long long opponent, int level)
{
	Board board = { player, opponent };
	Result *result = search->result;

	search_cleanup(search);
	hash_cleanup(&search->thread_hash);
	search_set_board(search, &board, BLACK);
	search_set_level(search, level, search->eval.n_empties);
	search_set_game_time(search, TIME_MAX);
	search_run(search);
	report(result->depth, result->selectivity, result->move, result->score, (double) result->n_nodes);
}
