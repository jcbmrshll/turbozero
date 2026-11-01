/* Egaroucid's search as functions for JavaScript: built by ../setup_wasm_engines.sh, from its
 * sources with egaroucid.patch, on its generic path (no AVX2). One thread, no book: a search holds
 * the worker it runs in until it ends, so stopping one means terminating the worker. */

#include "engine/engine_all.hpp"

#include <emscripten.h>

EM_JS(void, report, (int depth, int percent, int move, int score, double nodes), {
  Module.onResult({ depth, percent, move, score, nodes });
});

/* Loads the weights from resources/ in the virtual filesystem. The hash keys come from Egaroucid's
 * file for its smallest table, cut to the size of this build's, so that they're the same each time. */
extern "C" EMSCRIPTEN_KEEPALIVE void engine_init() {
    thread_pool.resize(0);
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
    mpc_init();
    move_ordering_init();
    transposition_table.resize(DEFAULT_HASH_LEVEL);
    hash_init(25, "");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 65536; ++j) {
            hash_rand_player[i][j] &= hash_sizes[DEFAULT_HASH_LEVEL] - 1;
            hash_rand_opponent[i][j] &= hash_sizes[DEFAULT_HASH_LEVEL] - 1;
        }
    }
    global_hash_level = DEFAULT_HASH_LEVEL;
    stability_init();
    if (!evaluate_init("resources/eval.egev2", "resources/eval_move_ordering_end.egev", false)) {
        std::cerr << "can't load the weights" << std::endl;
        abort();
    }
}

/* A fresh search of a position, as a new process would make it, for the player to move: `player`
 * and `opponent` are their discs, a1 the lowest bit (Egaroucid's a1 is the highest), and so is the
 * move reported. Its log, to stderr, says each step of the search. */
extern "C" EMSCRIPTEN_KEEPALIVE void engine_search(uint64_t player, uint64_t opponent, int level) {
    Board board;
    board.player = __builtin_bitreverse64(player);
    board.opponent = __builtin_bitreverse64(opponent);
    transposition_table.init();
    std::memset(lttable, 0, sizeof lttable);
    Search_result result = ai(board, level, false, 0, true, true);
    report(result.depth, result.probability, 63 - result.policy, result.value, (double) result.nodes);
}
