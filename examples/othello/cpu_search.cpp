// A single game's MCTS on CPU: the tree and Othello's rules, with leaves collected a batch at a time
// for the caller to evaluate (see cpu_search.py). AlphaZero's PUCT and rules as core/evaluators/mcts
// and pgx's Othello have them, with virtual loss steering a batch's traversals apart. A search can show
// the network each leaf in a random one of the board's 8 symmetries.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <openvino/openvino.hpp>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int PASS = 64;
constexpr float C_BASE = 19652.0f;

// how a traversal in flight counts against the edges it took
enum VirtualLoss {
    // it counts as a lost visit
    VL_LOSS = 0,
    // it counts as a visit in the exploration term only, the value untouched (Leela Chess Zero's)
    VL_VISIT = 1,
    // it counts as a drawn visit
    VL_DRAW = 2,
};

constexpr uint64_t NOT_A = 0xfefefefefefefefeULL;  // no square in the first column
constexpr uint64_t NOT_H = 0x7f7f7f7f7f7f7f7fULL;  // no square in the last column

// squares are bits, row by row, as pgx's actions
template <int S>
inline uint64_t step(uint64_t x) {
    if constexpr (S == 1) return (x & NOT_H) << 1;
    if constexpr (S == -1) return (x & NOT_A) >> 1;
    if constexpr (S == 8) return x << 8;
    if constexpr (S == -8) return x >> 8;
    if constexpr (S == 7) return (x & NOT_A) << 7;
    if constexpr (S == -7) return (x & NOT_H) >> 7;
    if constexpr (S == 9) return (x & NOT_H) << 9;
    if constexpr (S == -9) return (x & NOT_A) >> 9;
}

template <int S>
inline uint64_t moves_along(uint64_t me, uint64_t opp) {
    uint64_t run = step<S>(me) & opp;
    for (int i = 0; i < 5; i++) run |= step<S>(run) & opp;
    return step<S>(run);
}

inline uint64_t legal_moves(uint64_t me, uint64_t opp) {
    uint64_t found = moves_along<1>(me, opp) | moves_along<-1>(me, opp) | moves_along<8>(me, opp) |
                     moves_along<-8>(me, opp) | moves_along<7>(me, opp) | moves_along<-7>(me, opp) |
                     moves_along<9>(me, opp) | moves_along<-9>(me, opp);
    return found & ~(me | opp);
}

template <int S>
inline uint64_t flips_along(uint64_t pos, uint64_t me, uint64_t opp) {
    uint64_t run = step<S>(pos) & opp;
    for (int i = 0; i < 5; i++) run |= step<S>(run) & opp;
    return (step<S>(run) & me) ? run : 0;
}

inline uint64_t flips(uint64_t pos, uint64_t me, uint64_t opp) {
    return flips_along<1>(pos, me, opp) | flips_along<-1>(pos, me, opp) | flips_along<8>(pos, me, opp) |
           flips_along<-8>(pos, me, opp) | flips_along<7>(pos, me, opp) | flips_along<-7>(pos, me, opp) |
           flips_along<9>(pos, me, opp) | flips_along<-9>(pos, me, opp);
}

// The board's 8 symmetries, numbered by what they do in order: 1, swap rows and columns; 2, reverse the
// rows; 4, reverse the columns. 0 is the identity.
inline uint64_t transpose(uint64_t x) {
    uint64_t t = (x ^ (x << 28)) & 0x0f0f0f0f00000000ULL;
    x ^= t ^ (t >> 28);
    t = (x ^ (x << 14)) & 0x3333000033330000ULL;
    x ^= t ^ (t >> 14);
    t = (x ^ (x << 7)) & 0x5500550055005500ULL;
    x ^= t ^ (t >> 7);
    return x;
}

inline uint64_t reverse_columns(uint64_t x) {
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    return ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL) | ((x & 0x0f0f0f0f0f0f0f0fULL) << 4);
}

inline uint64_t symmetric_bits(uint64_t x, int symmetry) {
    if (symmetry & 1) x = transpose(x);
    if (symmetry & 2) x = __builtin_bswap64(x);
    if (symmetry & 4) x = reverse_columns(x);
    return x;
}

// where each action's square goes under each symmetry; a pass stays a pass
struct SymmetricActions {
    uint8_t to[8][PASS + 1];
};

constexpr SymmetricActions make_symmetric_actions() {
    SymmetricActions actions{};
    for (int symmetry = 0; symmetry < 8; symmetry++) {
        for (int action = 0; action < PASS; action++) {
            int row = action / 8, column = action % 8;
            if (symmetry & 1) {
                int swapped = row;
                row = column;
                column = swapped;
            }
            if (symmetry & 2) row = 7 - row;
            if (symmetry & 4) column = 7 - column;
            actions.to[symmetry][action] = (uint8_t)(row * 8 + column);
        }
        actions.to[symmetry][PASS] = PASS;
    }
    return actions;
}

constexpr SymmetricActions SYMMETRIC_ACTIONS = make_symmetric_actions();

struct Position {
    uint64_t me, opp;
    bool passed;
    bool terminated;
    // for the player to move, once it's terminated
    float reward;
};

Position play(const Position& p, int action) {
    Position next;
    uint64_t me = p.me, opp = p.opp;
    if (action != PASS) {
        uint64_t pos = 1ULL << action;
        uint64_t turned = flips(pos, me, opp);
        me ^= pos | turned;
        opp ^= turned;
    }
    next.me = opp;
    next.opp = me;
    next.passed = action == PASS;
    next.terminated = ((me | opp) == ~0ULL) || opp == 0 || (p.passed && action == PASS);
    next.reward = 0;
    if (next.terminated) {
        int mine = __builtin_popcountll(me), theirs = __builtin_popcountll(opp);
        next.reward = mine > theirs ? -1.0f : mine < theirs ? 1.0f : 0.0f;
    }
    return next;
}

struct Edge {
    float prior;
    // from the parent's side
    float w;
    int32_t n;
    int32_t in_flight;
    int32_t child;
    int32_t action;
};

struct Node {
    Position position;
    int32_t n;
    int32_t in_flight;
    int32_t first_edge;
    int32_t num_edges;  // 0 until it's been evaluated
    float w;            // for the player to move
    float value;        // the network's, once it's been evaluated
};

// The nodes evaluated so far, by position: a leaf whose position the game's searches have evaluated
// before, reached by another order of moves, takes that evaluation instead of the network's
// (Leela Chess Zero's cache of network evaluations, here the tree itself). Open addressing.
struct Evaluated {
    struct Slot {
        uint64_t me, opp;
        int32_t node;  // -1 for an empty slot
    };
    std::vector<Slot> slots = std::vector<Slot>(1024, Slot{0, 0, -1});
    size_t used = 0;

    static uint64_t hash(uint64_t me, uint64_t opp) {
        uint64_t z = me ^ (opp * 0x9e3779b97f4a7c15ULL) ^ (opp >> 29);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    int32_t find(uint64_t me, uint64_t opp) const {
        size_t mask = slots.size() - 1;
        for (size_t i = hash(me, opp) & mask;; i = (i + 1) & mask) {
            const Slot& slot = slots[i];
            if (slot.node < 0) return -1;
            if (slot.me == me && slot.opp == opp) return slot.node;
        }
    }

    void insert(uint64_t me, uint64_t opp, int32_t node) {
        if (2 * (used + 1) > slots.size()) {
            std::vector<Slot> old(slots.size() * 2, Slot{0, 0, -1});
            old.swap(slots);
            used = 0;
            for (const Slot& slot : old)
                if (slot.node >= 0) insert(slot.me, slot.opp, slot.node);
        }
        size_t mask = slots.size() - 1;
        for (size_t i = hash(me, opp) & mask;; i = (i + 1) & mask) {
            Slot& slot = slots[i];
            if (slot.node < 0) {
                slot = Slot{me, opp, node};
                used++;
                return;
            }
            if (slot.me == me && slot.opp == opp) return;
        }
    }

    void clear() {
        slots.assign(1024, Slot{0, 0, -1});
        used = 0;
    }
};

struct PathStep {
    int32_t node;
    int32_t edge;  // the edge taken to the node, -1 for the root
};

// the paths of a batch collected, end to end, for one caller at a time: a search's own (see
// `search_collect`), or an engine worker's
struct Batch {
    std::vector<PathStep> path;
    std::vector<PathStep> paths;
    std::vector<PathStep> collisions;
    std::vector<int32_t> leaf_ends;       // where each leaf's path ends in `paths`
    std::vector<int32_t> collision_ends;  // and each collision's, after the leaves'
    std::vector<uint8_t> leaf_symmetries;  // the symmetry each leaf was shown in
};

struct Search {
    int mode;
    float weight;
    float c_init;
    int max_collisions;
    bool random_symmetry;
    // whether a leaf may take an evaluation of its position from elsewhere in the tree
    bool reuse;
    uint64_t rng;  // splitmix64's state
    int32_t root;
    int remaining;
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    Batch batch;
    Evaluated evaluated;
    int collisions_total;
    int visits_total;
    // leaves that took an evaluation from the tree
    long reused_total;
    // traversals in a row that ended the game, with no leaf between
    int ended_in_a_row;
};

inline float edge_q(const Search& s, const Edge& e) {
    // q in [-1, 1] mapped to [0, 1], as core's `bounded_q_values`; unvisited edges are 0, a loss
    if (s.mode == VL_VISIT) {
        return e.n > 0 ? (e.w / e.n + 1) / 2 : 0;
    }
    float flying = s.weight * e.in_flight;
    float n = e.n + flying;
    if (n <= 0) return 0;
    float w = s.mode == VL_LOSS ? e.w - flying : e.w;
    return (w / n + 1) / 2;
}

inline int select(const Search& s, const Node& node) {
    float visits = node.n + s.weight * node.in_flight;
    float c = s.c_init + std::log((visits + C_BASE + 1) / C_BASE);
    float scale = c * std::sqrt(visits);
    int best = node.first_edge;
    float best_score = -1e30f;
    const Edge* edges = s.edges.data();
    for (int i = node.first_edge; i < node.first_edge + node.num_edges; i++) {
        const Edge& e = edges[i];
        float score = edge_q(s, e) + scale * e.prior / (e.n + s.weight * e.in_flight + 1);
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

int32_t new_node(Search& s, const Position& p) {
    Node node{};
    node.position = p;
    s.nodes.push_back(node);
    return (int32_t)s.nodes.size() - 1;
}

// the visit of a whole path, of a value for the player to move at its end
void back_up(Search& s, const PathStep* path, int length, float value, bool was_flying) {
    for (int i = length - 1; i >= 0; i--) {
        Node& node = s.nodes[path[i].node];
        node.n++;
        node.w += value;
        if (was_flying) node.in_flight--;
        if (path[i].edge >= 0) {
            Edge& e = s.edges[path[i].edge];
            e.n++;
            e.w -= value;
            if (was_flying) e.in_flight--;
        }
        value = -value;
    }
}

void fly(Search& s, const std::vector<PathStep>& path, int delta) {
    for (const PathStep& step : path) {
        s.nodes[step.node].in_flight += delta;
        if (step.edge >= 0) s.edges[step.edge].in_flight += delta;
    }
}

void write_observation(uint64_t me, uint64_t opp, float* out) {
    for (int i = 0; i < 64; i++) {
        out[i] = (float)((me >> i) & 1);
        out[64 + i] = (float)((opp >> i) & 1);
    }
}

// a random one of the 8, the top bits of splitmix64
inline int draw_symmetry(Search& s) {
    uint64_t z = (s.rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return (int)((z ^ (z >> 31)) >> 61);
}


// Expands a leaf from the evaluation of a node of the same position, if there's one. Its edges list the
// same legal moves in the same order.
bool reuse_evaluation(Search& s, int32_t index) {
    const Position& p = s.nodes[index].position;
    int32_t source = s.evaluated.find(p.me, p.opp);
    if (source < 0) return false;
    int first = (int)s.edges.size();
    int from = s.nodes[source].first_edge, count = s.nodes[source].num_edges;
    for (int i = 0; i < count; i++) {
        const Edge& e = s.edges[from + i];
        s.edges.push_back(Edge{e.prior, 0, 0, 0, -1, e.action});
    }
    Node& node = s.nodes[index];
    node.first_edge = first;
    node.num_edges = count;
    node.value = s.nodes[source].value;
    return true;
}

// `search_collect`'s traversals, their paths kept in `b` until `apply`
int collect(Search& s, Batch& b, int batch, float* out) {
    b.paths.clear();
    b.collisions.clear();
    b.leaf_ends.clear();
    b.collision_ends.clear();
    b.leaf_symmetries.clear();
    int leaves = 0, visits = 0, collided = 0;
    int want = std::min(batch, s.remaining);
    while (visits < want && collided <= s.max_collisions) {
        b.path.clear();
        int32_t index = s.root;
        b.path.push_back({index, -1});
        enum { LEAF, END, COLLISION } outcome;
        while (true) {
            Node& node = s.nodes[index];
            if (node.position.terminated) {
                outcome = END;
                break;
            }
            if (node.num_edges == 0) {
                outcome = node.in_flight > 0 ? COLLISION : LEAF;
                break;
            }
            int e = select(s, node);
            int32_t child = s.edges[e].child;
            if (child < 0) {
                Position next = play(node.position, s.edges[e].action);
                // (new_node may move `nodes`: `node` isn't used again)
                child = new_node(s, next);
                s.edges[e].child = child;
            }
            index = child;
            b.path.push_back({index, e});
        }
        if (outcome == END) {
            back_up(s, b.path.data(), (int)b.path.size(), s.nodes[index].position.reward, false);
            visits++;
            s.ended_in_a_row++;
        } else if (outcome == LEAF && s.reuse && reuse_evaluation(s, index)) {
            back_up(s, b.path.data(), (int)b.path.size(), s.nodes[index].value, false);
            visits++;
            s.reused_total++;
            s.ended_in_a_row = 0;
        } else if (outcome == LEAF) {
            fly(s, b.path, 1);
            b.paths.insert(b.paths.end(), b.path.begin(), b.path.end());
            b.leaf_ends.push_back((int32_t)b.paths.size());
            int symmetry = s.random_symmetry ? draw_symmetry(s) : 0;
            b.leaf_symmetries.push_back((uint8_t)symmetry);
            const Position& p = s.nodes[index].position;
            write_observation(symmetric_bits(p.me, symmetry), symmetric_bits(p.opp, symmetry),
                              out + (size_t)leaves * 128);
            leaves++;
            visits++;
            s.ended_in_a_row = 0;
        } else {
            fly(s, b.path, 1);
            b.collisions.insert(b.collisions.end(), b.path.begin(), b.path.end());
            b.collision_ends.push_back((int32_t)b.collisions.size());
            collided++;
        }
    }
    s.remaining -= visits;
    s.visits_total += visits;
    s.collisions_total += collided;
    return leaves;
}

// the evaluations of the leaves `collect` put in `b`, and its collisions undone
void apply(Search& s, Batch& b, const float* logits, const float* values) {
    int leaves = (int)b.leaf_ends.size();
    int start = 0;
    for (int i = 0; i < leaves; i++) {
        int end = b.leaf_ends[i];
        int32_t index = b.paths[end - 1].node;
        const Position p = s.nodes[index].position;
        const float* l = logits + (size_t)i * 65;
        const uint8_t* shown_at = SYMMETRIC_ACTIONS.to[b.leaf_symmetries[i]];
        uint64_t legal = legal_moves(p.me, p.opp);
        int first = (int)s.edges.size();
        float top = -1e30f;
        if (legal) {
            for (uint64_t m = legal; m; m &= m - 1) top = std::max(top, l[shown_at[__builtin_ctzll(m)]]);
        } else {
            top = l[PASS];
        }
        float total = 0;
        auto add = [&](int action) {
            float prior = std::exp(l[shown_at[action]] - top);
            total += prior;
            s.edges.push_back(Edge{prior, 0, 0, 0, -1, action});
        };
        if (legal) {
            for (uint64_t m = legal; m; m &= m - 1) add(__builtin_ctzll(m));
        } else {
            add(PASS);
        }
        for (size_t e = first; e < s.edges.size(); e++) s.edges[e].prior /= total;
        s.nodes[index].first_edge = first;
        s.nodes[index].num_edges = (int32_t)s.edges.size() - first;
        s.nodes[index].value = values[i];
        s.evaluated.insert(p.me, p.opp, index);
        back_up(s, b.paths.data() + start, end - start, values[i], true);
        start = end;
    }
    for (const PathStep& step : b.collisions) {
        s.nodes[step.node].in_flight--;
        if (step.edge >= 0) s.edges[step.edge].in_flight--;
    }
}

}  // namespace

extern "C" {

Search* search_new(int mode, float weight, float c_init, int max_collisions, int random_symmetry, int reuse,
                   uint64_t seed) {
    Search* s = new Search();
    s->mode = mode;
    s->weight = weight;
    s->c_init = c_init;
    s->max_collisions = max_collisions;
    s->random_symmetry = random_symmetry != 0;
    s->reuse = reuse != 0;
    s->rng = seed;
    s->root = -1;
    s->remaining = 0;
    s->collisions_total = 0;
    s->visits_total = 0;
    s->ended_in_a_row = 0;
    s->reused_total = 0;
    return s;
}

void search_free(Search* s) { delete s; }

// the game's start: black to move
void search_reset(Search* s) {
    s->nodes.clear();
    s->edges.clear();
    s->evaluated.clear();
    Position p{};
    p.me = (1ULL << 28) | (1ULL << 35);
    p.opp = (1ULL << 27) | (1ULL << 36);
    s->root = new_node(*s, p);
}

// the search for a move: `visits` more of the tree, each a leaf to evaluate or a game's end reached
void search_begin(Search* s, int visits) { s->remaining = visits; }

int search_remaining(const Search* s) { return s->remaining; }

// Up to `batch` traversals, as many as are left, each ending in a leaf to evaluate (whose observation
// is written to `out`, (batch, 2, 8, 8), in the symmetry drawn for it if the search draws them, and which
// `search_apply` takes the evaluation of, in that same symmetry), or a game's
// end, which counts as a visit without one. A traversal ending in a leaf already in flight is a
// collision: it visits nothing, but counts against the edges it took until `search_apply`. After
// `max_collisions` of them the batch stops short. Returns the leaves.
int search_collect(Search* s, int batch, float* out) { return collect(*s, s->batch, batch, out); }

// The evaluations of the leaves `search_collect` returned, (leaves, 65) policy logits and (leaves) values.
void search_apply(Search* s, const float* logits, const float* values) { apply(*s, s->batch, logits, values); }

int search_num_legal(const Search* s) {
    const Position& p = s->nodes[s->root].position;
    uint64_t legal = legal_moves(p.me, p.opp);
    return legal ? __builtin_popcountll(legal) : 1;
}

// A legal action, the first of them.
int search_first_legal(const Search* s) {
    const Position& p = s->nodes[s->root].position;
    uint64_t legal = legal_moves(p.me, p.opp);
    return legal ? __builtin_ctzll(legal) : PASS;
}

// The most visited action, the higher prior of equals.
int search_best_action(const Search* s) {
    const Node& root = s->nodes[s->root];
    int best = -1;
    int most = -1;
    float prior = -1;
    for (int i = root.first_edge; i < root.first_edge + root.num_edges; i++) {
        const Edge& e = s->edges[i];
        if (e.n > most || (e.n == most && e.prior > prior)) {
            most = e.n;
            prior = e.prior;
            best = e.action;
        }
    }
    return best;
}

// `out`, (65): the root's edges' visits
void search_visits(const Search* s, int32_t* out) {
    std::memset(out, 0, 65 * sizeof(int32_t));
    const Node& root = s->nodes[s->root];
    for (int i = root.first_edge; i < root.first_edge + root.num_edges; i++) {
        out[s->edges[i].action] = s->edges[i].n;
    }
}

float search_root_q(const Search* s) {
    const Node& root = s->nodes[s->root];
    return root.n ? root.w / root.n : 0;
}

// Moves the root down the tree, keeping what it searched under the action.
void search_play(Search* s, int action) {
    Node& root = s->nodes[s->root];
    for (int i = root.first_edge; i < root.first_edge + root.num_edges; i++) {
        if (s->edges[i].action == action && s->edges[i].child >= 0) {
            s->root = s->edges[i].child;
            return;
        }
    }
    Position next = play(root.position, action);
    s->root = new_node(*s, next);
}

// (me, opp, passed, terminated) of the root
void search_position(const Search* s, uint64_t* me, uint64_t* opp, int* passed, int* terminated) {
    const Position& p = s->nodes[s->root].position;
    *me = p.me;
    *opp = p.opp;
    *passed = p.passed;
    *terminated = p.terminated;
}

// The result at the root, a game's end, for the player to move.
float search_reward(const Search* s) { return s->nodes[s->root].position.reward; }

long search_collisions(const Search* s) { return s->collisions_total; }

long search_total_visits(const Search* s) { return s->visits_total; }

// leaves that took an evaluation from the tree rather than the network
long search_reused(const Search* s) { return s->reused_total; }

long search_num_nodes(const Search* s) { return (long)s->nodes.size(); }

void search_legal(const Search* s, int32_t* out) {
    const Position& p = s->nodes[s->root].position;
    std::memset(out, 0, 65 * sizeof(int32_t));
    uint64_t legal = legal_moves(p.me, p.opp);
    if (!legal) out[PASS] = 1;
    for (uint64_t m = legal; m; m &= m - 1) out[__builtin_ctzll(m)] = 1;
}

uint64_t search_symmetric_bits(uint64_t x, int symmetry) { return symmetric_bits(x, symmetry); }

int search_symmetric_action(int action, int symmetry) { return SYMMETRIC_ACTIONS.to[symmetry][action]; }

// An engine: a network compiled by OpenVINO for `threads` workers, each with its own requests on a core,
// which search a tree together, each collecting up to `batch` leaves, evaluating them while the others
// go on, and backing them up. The tree is shared under a lock; its virtual visits keep their
// traversals apart, as a batch's are. Python only starts a move's search and reads its result.
struct Worker {
    std::vector<float> observations;  // (batch, 2, 8, 8)
    // a request for each number of leaves, 1 to batch, their input `observations`
    std::vector<ov::InferRequest> requests;
    Batch batch;
};

struct Engine {
    ov::Core core;
    std::vector<ov::CompiledModel> models;
    std::vector<Worker> workers;
    int batch;
    long calls = 0;
    long leaves = 0;
    double infer_seconds = 0;
};

Engine* engine_new(const char* model_path, const char* precision, int threads, int batch) {
    try {
        Engine* engine = new Engine();
        engine->batch = batch;
        engine->workers.resize(threads);
        auto network = engine->core.read_model(model_path);
        ov::element::Type type = std::string(precision) == "bf16" ? ov::element::bf16 : ov::element::f32;
        std::vector<std::shared_ptr<ov::Model>> sized;
        for (int k = 1; k <= batch; k++) {
            auto copy = network->clone();
            copy->reshape({{copy->input().get_any_name(), ov::PartialShape{k, 2, 8, 8}}});
            sized.push_back(copy);
        }
        // a model of each size for each worker, one thread and one stream each, compiled in parallel
        engine->models.resize((size_t)threads * batch);
        std::vector<std::thread> compiling;
        for (int t = 0; t < threads; t++) {
            compiling.emplace_back([engine, &sized, type, t, batch] {
                Worker& w = engine->workers[t];
                w.observations.assign((size_t)batch * 128, 0.0f);
                for (int k = 1; k <= batch; k++) {
                    ov::CompiledModel& m = engine->models[(size_t)t * batch + k - 1];
                    m = engine->core.compile_model(sized[k - 1], "CPU", ov::inference_num_threads(1),
                                                   ov::num_streams(1), ov::hint::inference_precision(type),
                                                   ov::hint::enable_cpu_pinning(false));
                    ov::InferRequest request = m.create_infer_request();
                    request.set_input_tensor(
                        ov::Tensor(ov::element::f32, ov::Shape{(size_t)k, 2, 8, 8}, w.observations.data()));
                    request.infer();
                    w.requests.push_back(request);
                }
            });
        }
        for (auto& c : compiling) c.join();
        return engine;
    } catch (const std::exception& e) {
        fprintf(stderr, "engine_new: %s\n", e.what());
        return nullptr;
    }
}

void engine_free(Engine* engine) { delete engine; }

// Searches `s`'s tree with all the engine's workers: `visits` visits, or with none, until `seconds`
// pass, or, either way, until `stop_ended` traversals in a row end the game with no leaf between (the
// tree has nothing left to evaluate), 0 for never. Batches in flight then are finished. Returns the
// leaves evaluated.
long engine_search(Engine* engine, Search* s, int visits, double seconds, int stop_ended) {
    using clock = std::chrono::steady_clock;
    auto start = clock::now();
    auto deadline = start + std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(seconds));
    bool timed = visits <= 0;
    s->remaining = timed ? INT_MAX : visits;
    s->ended_in_a_row = 0;
    std::mutex lock;
    std::condition_variable applied;
    long generation = 0;
    bool stop = false;
    long leaves = 0;
    auto work = [&](Worker& w) {
        long calls = 0, mine = 0;
        double inferring = 0;
        std::unique_lock<std::mutex> held(lock);
        while (true) {
            if (stop || s->remaining <= 0 || (timed && clock::now() >= deadline) ||
                (stop_ended > 0 && s->ended_in_a_row >= stop_ended)) {
                stop = true;
                applied.notify_all();
                break;
            }
            int n = collect(*s, w.batch, engine->batch, w.observations.data());
            if (n == 0) {
                bool collided = !w.batch.collisions.empty();
                apply(*s, w.batch, nullptr, nullptr);
                if (collided) {
                    // every leaf it found is another's in flight: wait for one of them
                    long seen = generation;
                    applied.wait_for(held, std::chrono::milliseconds(2),
                                     [&] { return generation != seen || stop; });
                }
                continue;
            }
            held.unlock();
            ov::InferRequest& request = w.requests[n - 1];
            auto t0 = clock::now();
            request.infer();
            inferring += std::chrono::duration<double>(clock::now() - t0).count();
            const float* logits = request.get_output_tensor(0).data<float>();
            const float* values = request.get_output_tensor(1).data<float>();
            held.lock();
            apply(*s, w.batch, logits, values);
            generation++;
            applied.notify_all();
            calls++;
            mine += n;
        }
        leaves += mine;
        engine->calls += calls;
        engine->infer_seconds += inferring;
    };
    std::vector<std::thread> threads;
    for (size_t t = 1; t < engine->workers.size(); t++) threads.emplace_back(work, std::ref(engine->workers[t]));
    work(engine->workers[0]);
    for (auto& t : threads) t.join();
    s->remaining = 0;
    engine->leaves += leaves;
    return leaves;
}

// (calls, leaves, seconds in inference summed over workers) since the engine was made
void engine_stats(const Engine* engine, double* out) {
    out[0] = (double)engine->calls;
    out[1] = (double)engine->leaves;
    out[2] = engine->infer_seconds;
}
}
