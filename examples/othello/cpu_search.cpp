// A single game's MCTS on CPU: the tree and Othello's rules, with leaves collected a batch at a time
// for the caller to evaluate (see cpu_search.py). AlphaZero's PUCT and rules as core/evaluators/mcts
// and pgx's Othello have them, with virtual loss steering a batch's traversals apart.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
};

struct PathStep {
    int32_t node;
    int32_t edge;  // the edge taken to the node, -1 for the root
};

struct Search {
    int mode;
    float weight;
    float c_init;
    int max_collisions;
    int32_t root;
    int remaining;
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<PathStep> path;
    // the paths of the batch collected, end to end
    std::vector<PathStep> paths;
    std::vector<int32_t> leaf_ends;       // where each leaf's path ends in `paths`
    std::vector<int32_t> collision_ends;  // and each collision's, after the leaves'
    int collisions_total;
    int visits_total;
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

void fly(Search& s, int delta) {
    for (const PathStep& step : s.path) {
        s.nodes[step.node].in_flight += delta;
        if (step.edge >= 0) s.edges[step.edge].in_flight += delta;
    }
}

void write_observation(const Position& p, float* out) {
    for (int i = 0; i < 64; i++) {
        out[i] = (float)((p.me >> i) & 1);
        out[64 + i] = (float)((p.opp >> i) & 1);
    }
}

}  // namespace

extern "C" {

Search* search_new(int mode, float weight, float c_init, int max_collisions) {
    Search* s = new Search();
    s->mode = mode;
    s->weight = weight;
    s->c_init = c_init;
    s->max_collisions = max_collisions;
    s->root = -1;
    s->remaining = 0;
    s->collisions_total = 0;
    s->visits_total = 0;
    return s;
}

void search_free(Search* s) { delete s; }

// the game's start: black to move
void search_reset(Search* s) {
    s->nodes.clear();
    s->edges.clear();
    Position p{};
    p.me = (1ULL << 28) | (1ULL << 35);
    p.opp = (1ULL << 27) | (1ULL << 36);
    s->root = new_node(*s, p);
}

// the search for a move: `visits` more of the tree, each a leaf to evaluate or a game's end reached
void search_begin(Search* s, int visits) { s->remaining = visits; }

int search_remaining(const Search* s) { return s->remaining; }

// Up to `batch` traversals, as many as are left, each ending in a leaf to evaluate (whose observation
// is written to `out`, (batch, 2, 8, 8), and which `search_apply` takes the evaluation of), or a game's
// end, which counts as a visit without one. A traversal ending in a leaf already in flight is a
// collision: it visits nothing, but counts against the edges it took until `search_apply`. After
// `max_collisions` of them the batch stops short. Returns the leaves.
int search_collect(Search* s, int batch, float* out) {
    s->paths.clear();
    s->leaf_ends.clear();
    s->collision_ends.clear();
    std::vector<PathStep> collisions;
    int leaves = 0, visits = 0, collided = 0;
    int want = std::min(batch, s->remaining);
    while (visits < want && collided <= s->max_collisions) {
        s->path.clear();
        int32_t index = s->root;
        s->path.push_back({index, -1});
        enum { LEAF, END, COLLISION } outcome;
        while (true) {
            Node& node = s->nodes[index];
            if (node.position.terminated) {
                outcome = END;
                break;
            }
            if (node.num_edges == 0) {
                outcome = node.in_flight > 0 ? COLLISION : LEAF;
                break;
            }
            int e = select(*s, node);
            int32_t child = s->edges[e].child;
            if (child < 0) {
                Position next = play(node.position, s->edges[e].action);
                // (new_node may move `nodes`: `node` isn't used again)
                child = new_node(*s, next);
                s->edges[e].child = child;
            }
            index = child;
            s->path.push_back({index, e});
        }
        if (outcome == END) {
            back_up(*s, s->path.data(), (int)s->path.size(), s->nodes[index].position.reward, false);
            visits++;
        } else if (outcome == LEAF) {
            fly(*s, 1);
            s->paths.insert(s->paths.end(), s->path.begin(), s->path.end());
            s->leaf_ends.push_back((int32_t)s->paths.size());
            write_observation(s->nodes[index].position, out + (size_t)leaves * 128);
            leaves++;
            visits++;
        } else {
            fly(*s, 1);
            collisions.insert(collisions.end(), s->path.begin(), s->path.end());
            s->collision_ends.push_back((int32_t)collisions.size());
            collided++;
        }
    }
    s->remaining -= visits;
    s->visits_total += visits;
    s->collisions_total += collided;
    // collisions are undone with `search_apply`
    s->paths.insert(s->paths.end(), collisions.begin(), collisions.end());
    return leaves;
}

// The evaluations of the leaves `search_collect` returned, (leaves, 65) policy logits and (leaves) values.
void search_apply(Search* s, const float* logits, const float* values) {
    int leaves = (int)s->leaf_ends.size();
    int start = 0;
    for (int i = 0; i < leaves; i++) {
        int end = s->leaf_ends[i];
        int32_t index = s->paths[end - 1].node;
        const Position p = s->nodes[index].position;
        const float* l = logits + (size_t)i * 65;
        uint64_t legal = legal_moves(p.me, p.opp);
        int first = (int)s->edges.size();
        float top = -1e30f;
        if (legal) {
            for (uint64_t m = legal; m; m &= m - 1) top = std::max(top, l[__builtin_ctzll(m)]);
        } else {
            top = l[PASS];
        }
        float total = 0;
        auto add = [&](int action) {
            float prior = std::exp(l[action] - top);
            total += prior;
            s->edges.push_back(Edge{prior, 0, 0, 0, -1, action});
        };
        if (legal) {
            for (uint64_t m = legal; m; m &= m - 1) add(__builtin_ctzll(m));
        } else {
            add(PASS);
        }
        for (size_t e = first; e < s->edges.size(); e++) s->edges[e].prior /= total;
        s->nodes[index].first_edge = first;
        s->nodes[index].num_edges = (int32_t)s->edges.size() - first;
        back_up(*s, s->paths.data() + start, end - start, values[i], true);
        start = end;
    }
    size_t offset = start;
    for (int32_t end : s->collision_ends) {
        for (size_t i = offset; i < (size_t)start + end; i++) {
            const PathStep& step = s->paths[i];
            s->nodes[step.node].in_flight--;
            if (step.edge >= 0) s->edges[step.edge].in_flight--;
        }
        offset = (size_t)start + end;
    }
}

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

long search_num_nodes(const Search* s) { return (long)s->nodes.size(); }

void search_legal(const Search* s, int32_t* out) {
    const Position& p = s->nodes[s->root].position;
    std::memset(out, 0, 65 * sizeof(int32_t));
    uint64_t legal = legal_moves(p.me, p.opp);
    if (!legal) out[PASS] = 1;
    for (uint64_t m = legal; m; m &= m - 1) out[__builtin_ctzll(m)] = 1;
}
}
