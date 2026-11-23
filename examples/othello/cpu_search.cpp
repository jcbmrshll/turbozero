// A single game's MCTS on CPU: the tree and Othello's rules, with leaves collected a batch at a time
// for the caller to evaluate (see cpu_search.py). AlphaZero's PUCT and rules as core/evaluators/mcts
// and pgx's Othello have them, with virtual loss steering a batch's traversals apart. A search can show
// the network each leaf in a random one of the board's 8 symmetries.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <openvino/openvino.hpp>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <utility>
#include <vector>

#include "cpu_net.h"

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

// what a fixed number of visits does once a move is decided (see `leading_edge`)
enum Descend {
    // nothing: PUCT chooses every edge
    DESCEND_OFF = 0,
    // at the root: the rest of the search goes on under the move it plays
    DESCEND_ROOT = 1,
    // at the root, and then at each of our moves below it, a turn on, whose move the visits left now
    // and every search up to it can't change
    DESCEND_RECURSIVE = 2,
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

// The tree is searched by several threads at once (an engine's workers, see `engine_search`) with no
// lock over it: a node's and an edge's counters change by atomic operations, read as they stand by
// the others' traversals (Leela Chess Zero's and KataGo's way). These are GCC's and Clang's builtins
// on plain fields, so the tree stays plain data that a copy of the search (`search_clone`) copies.
template <class T>
inline T relaxed(const T& x) {
    T v;
    __atomic_load(&x, &v, __ATOMIC_RELAXED);
    return v;
}

// a field another thread published (`publish`), and what it wrote before
template <class T>
inline T acquired(const T& x) {
    T v;
    __atomic_load(&x, &v, __ATOMIC_ACQUIRE);
    return v;
}

template <class T>
inline void publish(T& x, T v) {
    __atomic_store(&x, &v, __ATOMIC_RELEASE);
}

inline void bump(int32_t& x, int32_t by) { __atomic_fetch_add(&x, by, __ATOMIC_RELAXED); }

inline void bump(float& x, float by) {
    float old = relaxed(x), sum;
    do {
        sum = old + by;
    } while (!__atomic_compare_exchange(&x, &old, &sum, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

// a traversal's virtual visit taken back: released, so that a traversal claiming a leaf (`collect`)
// after it sees the leaf's evaluation, if that's why it ended
inline void land(int32_t& in_flight) { __atomic_fetch_sub(&in_flight, 1, __ATOMIC_RELEASE); }

struct Edge {
    float prior;
    // from the parent's side
    float w;
    int32_t n;
    int32_t in_flight;
    int32_t child;  // -1 until a traversal takes the edge
    int32_t action;
};

// A node's visits (`n`, `w`, `in_flight`) are those of the edge to it, from the other side, but for the
// root's, which has none: only the root's are kept up to date, and taken from its edge when it's made the
// root (`search_play`), so a visit changes half as many shared counters. A leaf's `in_flight` claims it
// for evaluation (see `collect`).
struct Node {
    Position position;
    int32_t n;
    int32_t in_flight;
    int32_t first_edge;
    int32_t num_edges;  // 0 until it's been evaluated; published once its edges and value are written
    float w;            // for the player to move
    float value;        // the network's, once it's been evaluated
};

// Asks for `bytes` from `start` to be backed by huge pages (Linux's transparent ones, where they're
// granted only to memory that asks): the tree fills its room a page at a time, and a 2 MB page is one
// fault where 4 KB pages are 512, and one TLB entry.
void prefer_huge_pages(const void* start, size_t bytes) {
#ifdef MADV_HUGEPAGE
    const uintptr_t page = 4096;
    uintptr_t from = ((uintptr_t)start + page - 1) & ~(page - 1);
    uintptr_t to = ((uintptr_t)start + bytes) & ~(page - 1);
    if (to > from) madvise((void*)from, to - from, MADV_HUGEPAGE);
#endif
}

// An array that grows a segment at a time and never moves, so that threads read it while others add to
// it: no growth copies the whole of it at once, with every worker waiting. `allocate` hands out indices
// contiguous within a segment (a node's edges are read as one run); an allocation that would cross a
// segment's end skips to the next one. A segment is a whole number of 2 MB pages, on huge pages if the
// system grants them; its memory is used only as its elements are.
template <class T>
struct Segments {
    static constexpr size_t HUGE_PAGE = size_t(1) << 21;
    static constexpr int SHIFT = std::max(16, 21 - __builtin_ctzll(sizeof(T)));
    static constexpr int64_t SIZE = int64_t(1) << SHIFT;
    static constexpr int64_t COUNT = (int64_t(1) << 31) / SIZE;
    static constexpr size_t BYTES = (SIZE * sizeof(T) + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE;
    T** table;
    // indices handed out, those skipped included
    int64_t used = 0;

    Segments() : table(static_cast<T**>(std::calloc(COUNT, sizeof(T*)))) {
        if (!table) throw std::bad_alloc();
    }
    Segments(const Segments& other) : Segments() { copy(other); }
    Segments& operator=(const Segments& other) {
        if (this != &other) {
            release();
            copy(other);
        }
        return *this;
    }
    ~Segments() {
        release();
        std::free(table);
    }

    T& operator[](int64_t i) const { return relaxed(table[i >> SHIFT])[i & (SIZE - 1)]; }

    static T* fresh() {
        T* segment = static_cast<T*>(std::aligned_alloc(HUGE_PAGE, BYTES));
        if (!segment) throw std::bad_alloc();
        prefer_huge_pages(segment, BYTES);
        return segment;
    }

    // segment `k`, made now if no thread has
    void ensure(int64_t k) {
        if (acquired(table[k])) return;
        T* made = fresh();
        T* none = nullptr;
        if (!__atomic_compare_exchange_n(&table[k], &none, made, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            std::free(made);
    }

    int32_t allocate(int count) {
        while (true) {
            int64_t first = __atomic_fetch_add(&used, (int64_t)count, __ATOMIC_RELAXED);
            int64_t segment = first >> SHIFT;
            if (segment == (first + count - 1) >> SHIFT) {
                if (first + count > (int64_t(1) << 31)) throw std::bad_alloc();
                ensure(segment);
                return (int32_t)first;
            }
        }
    }

    // the segments for `count` elements made now, rather than as they fill; false if they can't be
    // (those made kept)
    bool reserve(int64_t count) {
        if (count > (int64_t(1) << 31)) return false;
        try {
            for (int64_t k = 0; k < (count + SIZE - 1) >> SHIFT; k++) ensure(k);
        } catch (const std::bad_alloc&) {
            return false;
        }
        return true;
    }

    // the elements the segments made so far hold
    int64_t capacity() const {
        int64_t made = 0;
        for (int64_t k = 0; k < COUNT; k++) made += table[k] != nullptr;
        return made * SIZE;
    }

    // no indices handed out (the segments kept for the next)
    void clear() { used = 0; }

    void copy(const Segments& other) {
        used = other.used;
        for (int64_t k = 0; k < COUNT; k++) {
            if (!other.table[k]) continue;
            table[k] = fresh();
            int64_t filled = std::min(SIZE, std::max(int64_t(0), used - k * SIZE));
            std::memcpy(static_cast<void*>(table[k]), other.table[k], filled * sizeof(T));
        }
    }

    void release() {
        for (int64_t k = 0; k < COUNT; k++) {
            std::free(table[k]);
            table[k] = nullptr;
        }
        used = 0;
    }
};

// Memory handed out zeroed by calloc, its elements default-constructed by doing nothing: a large
// buffer comes straight from the kernel, its pages zeroed when first touched (`std::vector`'s own
// allocator would write every element up front).
template <class T>
struct ZeroedAllocator {
    using value_type = T;
    ZeroedAllocator() = default;
    template <class U>
    ZeroedAllocator(const ZeroedAllocator<U>&) {}
    T* allocate(size_t n) {
        void* p = std::calloc(n, sizeof(T));
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) { std::free(p); }
    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        if constexpr (sizeof...(Args) > 0) ::new ((void*)p) U(std::forward<Args>(args)...);
    }
    bool operator==(const ZeroedAllocator&) const { return true; }
    bool operator!=(const ZeroedAllocator&) const { return false; }
};

// The nodes evaluated so far, by position: a leaf whose position the game's searches have evaluated
// before, reached by another order of moves, takes that evaluation instead of the network's
// (Leela Chess Zero's cache of network evaluations, here the tree itself). Open addressing, in
// shards of their own lock each, by the hash's top bits: workers seldom meet at one, and one growing
// (rehashed whole) holds up only those that do. A slot of all zeros is empty, so a fresh shard is
// calloc's, its pages zeroed by the kernel as they're touched; its entries spread over the whole of it,
// so room reserved here is paid for as it fills, not as it's used up.
struct Evaluated {
    struct Slot {
        uint64_t me, opp;
        int32_t node_plus_one;  // 0 for an empty slot
    };
    using Slots = std::vector<Slot, ZeroedAllocator<Slot>>;
    static constexpr size_t SMALLEST = 64;
    struct alignas(64) Shard {
        std::mutex lock;
        Slots slots = Slots(SMALLEST);
        size_t used = 0;
        // the size `reserve` asked for, which `clear` keeps
        size_t reserved = SMALLEST;
        Shard() = default;
        Shard(const Shard& other) : slots(other.slots), used(other.used), reserved(other.reserved) {}
        Shard& operator=(const Shard& other) {
            slots = other.slots;
            used = other.used;
            reserved = other.reserved;
            return *this;
        }
    };
    static constexpr int SHARD_BITS = 6;
    std::vector<Shard> shards = std::vector<Shard>(1 << SHARD_BITS);

    static uint64_t hash(uint64_t me, uint64_t opp) {
        uint64_t z = me ^ (opp * 0x9e3779b97f4a7c15ULL) ^ (opp >> 29);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    Shard& shard_of(uint64_t h) { return shards[h >> (64 - SHARD_BITS)]; }

    int32_t find(uint64_t me, uint64_t opp) {
        uint64_t h = hash(me, opp);
        Shard& shard = shard_of(h);
        std::lock_guard<std::mutex> held(shard.lock);
        size_t mask = shard.slots.size() - 1;
        for (size_t i = h & mask;; i = (i + 1) & mask) {
            const Slot& slot = shard.slots[i];
            if (slot.node_plus_one == 0) return -1;
            if (slot.me == me && slot.opp == opp) return slot.node_plus_one - 1;
        }
    }

    static void put(Shard& shard, uint64_t h, const Slot& entry) {
        size_t mask = shard.slots.size() - 1;
        for (size_t i = h & mask;; i = (i + 1) & mask) {
            Slot& slot = shard.slots[i];
            if (slot.node_plus_one == 0) {
                slot = entry;
                shard.used++;
                return;
            }
            if (slot.me == entry.me && slot.opp == entry.opp) return;
        }
    }

    // a shard to `size` slots (a power of 2), every entry kept; its lock held
    static void resize(Shard& shard, size_t size) {
        Slots old(size);
        old.swap(shard.slots);
        shard.used = 0;
        for (const Slot& slot : old)
            if (slot.node_plus_one) put(shard, hash(slot.me, slot.opp), slot);
    }

    void insert(uint64_t me, uint64_t opp, int32_t node) {
        uint64_t h = hash(me, opp);
        Shard& shard = shard_of(h);
        std::lock_guard<std::mutex> held(shard.lock);
        if (2 * (shard.used + 1) > shard.slots.size()) resize(shard, shard.slots.size() * 2);
        put(shard, h, Slot{me, opp, node + 1});
    }

    // room for about `entries` without growing (a shard's share, and a quarter more for the shards
    // the hash favours)
    void reserve(size_t entries) {
        size_t share = (entries >> SHARD_BITS) + (entries >> (SHARD_BITS + 2));
        size_t size = SMALLEST;
        while (size < 2 * share) size *= 2;
        for (Shard& shard : shards) {
            std::lock_guard<std::mutex> held(shard.lock);
            shard.reserved = std::max(shard.reserved, size);
            if (size > shard.slots.size()) resize(shard, size);
            prefer_huge_pages(shard.slots.data(), shard.slots.size() * sizeof(Slot));
        }
    }

    void clear() {
        for (Shard& shard : shards) {
            shard.slots = Slots(shard.reserved);
            if (shard.reserved > SMALLEST) prefer_huge_pages(shard.slots.data(), shard.reserved * sizeof(Slot));
            shard.used = 0;
        }
    }
};

struct PathStep {
    int32_t node;
    int32_t edge;  // the edge taken to the node, -1 for the root
};

// what a caller's traversals came to, added to the search's totals when it's done (rather than
// as they happen, by every worker on one cache line)
struct Tally {
    long visits = 0;
    long collisions = 0;
    long reused = 0;
    long descended = 0;
    long descended_deeper = 0;
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
    // a leaf's edges as they're made, before they go into the tree
    std::vector<Edge> made;
    Tally tally;
};

// A flag set from another thread; a copy of it (`search_clone`) starts unset.
struct Halt {
    std::atomic<bool> on{false};
    Halt() = default;
    Halt(const Halt&) {}
    Halt& operator=(const Halt&) { return *this; }
};

struct Search {
    int mode;
    float weight;
    float c_init;
    // 1 / the prior's softmax temperature (see core's `MCTS`): logits are scaled by it
    float inverse_temperature;
    int max_collisions;
    bool random_symmetry;
    // whether a leaf may take an evaluation of its position from elsewhere in the tree
    bool reuse;
    // a `Descend`: whether a fixed number of visits, once its move is decided, goes on under that
    // move only (see `leading_edge`)
    int descend;
    // the root's edge the search's move is decided on, -1 until it is (set once, atomically)
    int32_t decided;
    // the visits of this search, and so of each search to come (0 on a clock)
    int budget;
    // positions with so many empties or fewer the player solves rather than searches: their move
    // isn't the most visited, so `DESCEND_RECURSIVE` never takes it as decided
    int solved_from;
    uint64_t rng;  // splitmix64's state, advanced atomically
    int32_t root;
    // visits not yet taken by a traversal (each takes one atomically as it starts)
    int remaining;
    Segments<Node> nodes;
    Segments<Edge> edges;
    Batch batch;
    Evaluated evaluated;
    int collisions_total;
    int visits_total;
    // leaves that took an evaluation from the tree
    long reused_total;
    // visits sent down a decided move's edge rather than chosen by PUCT: at the root, and at one
    // of our moves below it
    long descended_total;
    long descended_deeper_total;
    // set from another thread (`search_halt`) to end an engine's search under way, or the next one,
    // at once: a ponder stopped when the opponent's move comes
    Halt halted;
};

// q in [-1, 1] mapped to [0, 1], as core's `bounded_q_values`; unvisited edges are 0, a loss. Of an
// edge's counters as one traversal read them.
inline float edge_q(const Search& s, float w, int32_t n, int32_t in_flight) {
    if (s.mode == VL_VISIT) {
        return n > 0 ? (w / n + 1) / 2 : 0;
    }
    float flying = s.weight * in_flight;
    float visits = n + flying;
    if (visits <= 0) return 0;
    float wins = s.mode == VL_LOSS ? w - flying : w;
    return (wins / visits + 1) / 2;
}

// PUCT's edge of an evaluated node, of its `count` edges, reached by edge `from` (-1 for the root):
// the node's visits are the edge's, but for the traversal choosing, in flight on it already
inline int select(const Search& s, const Node& node, int count, int32_t from) {
    float visits = from < 0 ? relaxed(node.n) + s.weight * relaxed(node.in_flight)
                            : relaxed(s.edges[from].n) + s.weight * (relaxed(s.edges[from].in_flight) - 1);
    float c = s.c_init + std::log((visits + C_BASE + 1) / C_BASE);
    float scale = c * std::sqrt(visits);
    int best = node.first_edge;
    float best_score = -1e30f;
    const Edge* edges = &s.edges[node.first_edge];
    for (int i = 0; i < count; i++) {
        const Edge& e = edges[i];
        int32_t n = relaxed(e.n), in_flight = relaxed(e.in_flight);
        float score = edge_q(s, relaxed(e.w), n, in_flight) + scale * e.prior / (n + s.weight * in_flight + 1);
        if (score > best_score) {
            best_score = score;
            best = node.first_edge + i;
        }
    }
    return best;
}

// The edge of `node` the move from it is decided on: its visits more than any other edge's could reach
// with `extra` more visits and every traversal in flight, so that it stays the most visited
// (`search_best_action`'s choice) however they go. -1 if none is. Once every traversal through the
// node takes it, no other edge gains, and the decision holds.
int leading_edge(const Search& s, const Node& node, long extra) {
    const Edge* edges = &s.edges[node.first_edge];
    int count = acquired(node.num_edges);
    int top = 0;
    int32_t most = relaxed(edges[0].n);
    for (int i = 1; i < count; i++) {
        int32_t n = relaxed(edges[i].n);
        if (n > most || (n == most && edges[i].prior > edges[top].prior)) {
            top = i;
            most = n;
        }
    }
    for (int i = 0; i < count; i++) {
        if (i != top && (long)relaxed(edges[i].n) + relaxed(edges[i].in_flight) + extra >= most) return -1;
    }
    return node.first_edge + top;
}

// With `descend`, the root's edge once the search's move is decided, against the `left` visits still
// to come. The search's visits then go on under it, where the tree the next search starts from keeps
// them, rather than to moves that can't be played. -1 until then, and always without a fixed number
// of visits (`remaining` then is unbounded).
int decided_edge(Search& s, const Node& root, int left) {
    int32_t decided = relaxed(s.decided);
    if (decided < 0 && s.descend != DESCEND_OFF) {
        decided = leading_edge(s, root, left);
        int32_t none = -1;
        // the first worker's to decide it
        if (decided >= 0 && !__atomic_compare_exchange_n(&s.decided, &none, decided, false, __ATOMIC_RELAXED,
                                                          __ATOMIC_RELAXED))
            decided = none;
    }
    return decided;
}

inline int empties(const Position& p) { return 64 - __builtin_popcountll(p.me | p.opp); }

int32_t new_node(Search& s, const Position& p) {
    int32_t index = s.nodes.allocate(1);
    Node& node = s.nodes[index];
    node = Node{};
    node.position = p;
    return index;
}

// The node `edge` of `node` leads to, made now if it's the first traversal to take it. Of two at once,
// the node of the one that publishes it first; the other's is left unused.
int32_t child_of(Search& s, const Node& node, Edge& edge) {
    int32_t child = acquired(edge.child);
    if (child >= 0) return child;
    int32_t made = new_node(s, play(node.position, edge.action));
    if (__atomic_compare_exchange_n(&edge.child, &child, made, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        return made;
    return child;
}

// the visit of a whole path in flight, of a value for the player to move at its end
void back_up(Search& s, const PathStep* path, int length, float value) {
    for (int i = length - 1; i >= 0; i--) {
        Node& node = s.nodes[path[i].node];
        if (i == 0) {
            bump(node.n, 1);
            bump(node.w, value);
        }
        if (i == 0 || i == length - 1) land(node.in_flight);
        if (path[i].edge >= 0) {
            Edge& e = s.edges[path[i].edge];
            bump(e.n, 1);
            bump(e.w, -value);
            land(e.in_flight);
        }
        value = -value;
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
    uint64_t z = __atomic_add_fetch(&s.rng, 0x9e3779b97f4a7c15ULL, __ATOMIC_RELAXED);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return (int)((z ^ (z >> 31)) >> 61);
}

// A leaf's edges, `count` of them made in `made`, put into the tree with its value: published, its
// evaluation is there for every traversal (and, with `reuse`, every leaf of the same position).
void expand(Search& s, int32_t index, const Edge* made, int count, float value) {
    int32_t first = s.edges.allocate(count);
    std::copy(made, made + count, &s.edges[first]);
    Node& node = s.nodes[index];
    node.first_edge = first;
    node.value = value;
    publish(node.num_edges, (int32_t)count);
}

// Expands a leaf from the evaluation of a node of the same position, if there's one. Its edges list the
// same legal moves in the same order.
bool reuse_evaluation(Search& s, Batch& b, int32_t index) {
    const Position& p = s.nodes[index].position;
    int32_t source = s.evaluated.find(p.me, p.opp);
    if (source < 0) return false;
    const Node& from = s.nodes[source];
    int count = acquired(from.num_edges);
    const Edge* edges = &s.edges[from.first_edge];
    b.made.clear();
    for (int i = 0; i < count; i++) b.made.push_back(Edge{edges[i].prior, 0, 0, 0, -1, edges[i].action});
    expand(s, index, b.made.data(), count, from.value);
    return true;
}

// `search_collect`'s traversals, their paths kept in `b` until `apply`; each leaf's observation written
// to `out` as (2, 8, 8) floats. A traversal takes a visit of those left as it starts, and adds a
// virtual visit to each node and edge as it goes; reaching a leaf, it claims it (its virtual visit the
// first), or, if another traversal has, collides there.
int collect(Search& s, Batch& b, int batch, float* out) {
    b.paths.clear();
    b.collisions.clear();
    b.leaf_ends.clear();
    b.collision_ends.clear();
    b.leaf_symmetries.clear();
    int leaves = 0, visits = 0, collided = 0;
    while (visits < batch && collided <= s.max_collisions) {
        // the visits left as this one is taken, it included
        int left = __atomic_fetch_sub(&s.remaining, 1, __ATOMIC_RELAXED);
        if (left <= 0) {
            __atomic_fetch_add(&s.remaining, 1, __ATOMIC_RELAXED);
            break;
        }
        b.path.clear();
        int32_t index = s.root;
        b.path.push_back({index, -1});
        enum { LEAF, END, COLLISION } outcome;
        // whether the path so far took a decided move at each of our turns (see `Descend`); and
        // whether it did at the root, and below it
        bool chain = s.descend != DESCEND_OFF;
        bool descended = false, deeper = false;
        int depth = 0;
        while (true) {
            Node& node = s.nodes[index];
            if (node.position.terminated) {
                // (in flight at the root and at its end, as a leaf would be)
                bump(node.in_flight, 1);
                outcome = END;
                break;
            }
            int count = acquired(node.num_edges);
            if (count == 0) {
                if (__atomic_fetch_add(&node.in_flight, 1, __ATOMIC_ACQ_REL) > 0) {
                    outcome = COLLISION;
                    break;
                }
                count = acquired(node.num_edges);
                if (count == 0) {
                    outcome = LEAF;
                    break;
                }
                // evaluated since: on through it
                land(node.in_flight);
            }
            int e = -1;
            if (chain && depth == 0) {
                e = decided_edge(s, node, left);
                descended = e >= 0;
                chain = descended;
            } else if (chain && depth % 2 == 0) {
                // our move `depth / 2` turns on, decided against this search's visits left and the
                // whole of each search up to it: their root rule keeps it, so visits to its other
                // moves would be lost by then
                if (s.descend == DESCEND_RECURSIVE && empties(node.position) > s.solved_from)
                    e = leading_edge(s, node, (long)left + (long)(depth / 2) * s.budget);
                deeper = deeper || e >= 0;
                chain = e >= 0;
            }
            depth++;
            if (e < 0) e = select(s, node, count, b.path.back().edge);
            Edge& edge = s.edges[e];
            if (b.path.size() == 1) bump(node.in_flight, 1);
            bump(edge.in_flight, 1);
            index = child_of(s, node, edge);
            b.path.push_back({index, e});
        }
        if (outcome != COLLISION && descended) b.tally.descended++;
        if (outcome != COLLISION && deeper) b.tally.descended_deeper++;
        if (outcome == END) {
            back_up(s, b.path.data(), (int)b.path.size(), s.nodes[index].position.reward);
            visits++;
        } else if (outcome == LEAF && s.reuse && reuse_evaluation(s, b, index)) {
            back_up(s, b.path.data(), (int)b.path.size(), s.nodes[index].value);
            visits++;
            b.tally.reused++;
        } else if (outcome == LEAF) {
            b.paths.insert(b.paths.end(), b.path.begin(), b.path.end());
            b.leaf_ends.push_back((int32_t)b.paths.size());
            int symmetry = s.random_symmetry ? draw_symmetry(s) : 0;
            b.leaf_symmetries.push_back((uint8_t)symmetry);
            const Position& p = s.nodes[index].position;
            write_observation(symmetric_bits(p.me, symmetry), symmetric_bits(p.opp, symmetry),
                              out + (size_t)leaves * 128);
            leaves++;
            visits++;
        } else {
            // a visit of none: given back
            __atomic_fetch_add(&s.remaining, 1, __ATOMIC_RELAXED);
            b.collisions.insert(b.collisions.end(), b.path.begin(), b.path.end());
            b.collision_ends.push_back((int32_t)b.collisions.size());
            collided++;
        }
    }
    b.tally.visits += visits;
    b.tally.collisions += collided;
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
        float top = -1e30f;
        if (legal) {
            for (uint64_t m = legal; m; m &= m - 1) top = std::max(top, l[shown_at[__builtin_ctzll(m)]]);
        } else {
            top = l[PASS];
        }
        float total = 0;
        b.made.clear();
        auto add = [&](int action) {
            float prior = std::exp((l[shown_at[action]] - top) * s.inverse_temperature);
            total += prior;
            b.made.push_back(Edge{prior, 0, 0, 0, -1, action});
        };
        if (legal) {
            for (uint64_t m = legal; m; m &= m - 1) add(__builtin_ctzll(m));
        } else {
            add(PASS);
        }
        for (Edge& e : b.made) e.prior /= total;
        expand(s, index, b.made.data(), (int)b.made.size(), values[i]);
        s.evaluated.insert(p.me, p.opp, index);
        back_up(s, b.paths.data() + start, end - start, values[i]);
        start = end;
    }
    int from = 0;
    for (int32_t end : b.collision_ends) {
        for (int i = from; i < end; i++) {
            const PathStep& step = b.collisions[i];
            if (i == from || i == end - 1) land(s.nodes[step.node].in_flight);
            if (step.edge >= 0) land(s.edges[step.edge].in_flight);
        }
        from = end;
    }
}

// a caller's traversals added to the search's totals
void count_up(Search& s, Tally& tally) {
    s.visits_total += (int)tally.visits;
    s.collisions_total += (int)tally.collisions;
    s.reused_total += tally.reused;
    s.descended_total += tally.descended;
    s.descended_deeper_total += tally.descended_deeper;
    tally = Tally{};
}

}  // namespace

extern "C" {

Search* search_new(int mode, float weight, float c_init, float prior_temperature, int max_collisions,
                   int random_symmetry, int reuse, int descend, uint64_t seed) {
    Search* s = new Search();
    s->mode = mode;
    s->weight = weight;
    s->c_init = c_init;
    s->inverse_temperature = 1.0f / prior_temperature;
    s->max_collisions = max_collisions;
    s->random_symmetry = random_symmetry != 0;
    s->reuse = reuse != 0;
    s->descend = descend;
    s->decided = -1;
    s->budget = 0;
    s->solved_from = 0;
    s->descended_total = 0;
    s->descended_deeper_total = 0;
    s->rng = seed;
    s->root = -1;
    s->remaining = 0;
    s->collisions_total = 0;
    s->visits_total = 0;
    s->reused_total = 0;
    return s;
}

void search_free(Search* s) { delete s; }

// A position given whole, its tree cleared: `me` the discs of the player to move, `opp` the other's.
void search_set_position(Search* s, uint64_t me, uint64_t opp) {
    s->nodes.clear();
    s->edges.clear();
    s->evaluated.clear();
    Position p{};
    p.me = me;
    p.opp = opp;
    s->root = new_node(*s, p);
    s->decided = -1;
}

// A copy of the search, tree and settings, that goes on by itself; with `descend` a `Descend`, that
// setting changed (-1 keeps it).
Search* search_clone(const Search* s, int descend) {
    Search* copy = new Search(*s);
    if (descend >= 0) copy->descend = descend;
    return copy;
}

// Positions with `empties` or fewer the player solves rather than searches (see `Search::solved_from`).
void search_set_solved_from(Search* s, int empties) { s->solved_from = empties; }

// the game's start: black to move
void search_reset(Search* s) { search_set_position(s, (1ULL << 28) | (1ULL << 35), (1ULL << 27) | (1ULL << 36)); }

// the search for a move: `visits` more of the tree, each a leaf to evaluate or a game's end reached
void search_begin(Search* s, int visits) {
    s->remaining = visits;
    s->budget = visits;
    s->decided = -1;
}

int search_remaining(const Search* s) { return s->remaining; }

// Up to `batch` traversals, as many as are left, each ending in a leaf to evaluate (whose observation
// is written to `out`, (batch, 2, 8, 8), in the symmetry drawn for it if the search draws them, and which
// `search_apply` takes the evaluation of, in that same symmetry), or a game's
// end, which counts as a visit without one. A traversal ending in a leaf already in flight is a
// collision: it visits nothing, but counts against the edges it took until `search_apply`. After
// `max_collisions` of them the batch stops short. Returns the leaves.
int search_collect(Search* s, int batch, float* out) {
    int leaves = collect(*s, s->batch, batch, out);
    count_up(*s, s->batch.tally);
    return leaves;
}

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
    s->decided = -1;
    Node& root = s->nodes[s->root];
    for (int i = root.first_edge; i < root.first_edge + root.num_edges; i++) {
        const Edge& e = s->edges[i];
        if (e.action == action && e.child >= 0) {
            // its visits, the root's to keep (see `Node`)
            Node& child = s->nodes[e.child];
            child.n = e.n;
            child.w = -e.w;
            child.in_flight = e.in_flight;
            s->root = e.child;
            return;
        }
    }
    Position next = play(root.position, action);
    s->root = new_node(*s, next);
}

// Room for `nodes` nodes and `edges` edges made now, and the table of evaluated positions for
// `positions`, all kept when the tree is started afresh (`search_set_position`). The tree's storage
// never moves, so growing it holds up no one; reserved, it's on hand at once rather than a segment at a
// time. Its nodes and edges cost memory only as they're used; the table's entries land all over it (see
// `Evaluated`). Returns 0 if the memory couldn't be had, what was made kept.
int search_reserve(Search* s, long nodes, long edges, long positions) {
    if (!s->nodes.reserve(nodes) || !s->edges.reserve(edges)) return 0;
    try {
        s->evaluated.reserve((size_t)positions);
        return 1;
    } catch (const std::exception&) {
        return 0;
    }
}

// (nodes, edges) the tree has room for before it makes more
void search_capacity(const Search* s, long* out) {
    out[0] = (long)s->nodes.capacity();
    out[1] = (long)s->edges.capacity();
}

// the edges handed out, those an allocation skipped at a segment's end included
long search_num_edges(const Search* s) { return (long)s->edges.used; }

// Ends an engine's search of `s` under way, and any started while it's set, at once (the batches in
// flight finished); safe from any thread. 0 lets searches run again.
void search_halt(Search* s, int halted) { s->halted.on.store(halted != 0); }

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

// visits a decided move took at the root (see `decided_edge`), and visits one took below it
long search_descended(const Search* s) { return s->descended_total; }
long search_descended_deeper(const Search* s) { return s->descended_deeper_total; }

long search_num_nodes(const Search* s) { return (long)s->nodes.used; }

// the nodes of the tree under the root, the root's own included: what a search from it starts with
long search_subtree_nodes(const Search* s) {
    long count = 0;
    std::vector<int32_t> stack{s->root};
    while (!stack.empty()) {
        const Node& node = s->nodes[stack.back()];
        stack.pop_back();
        count++;
        for (int i = node.first_edge; i < node.first_edge + node.num_edges; i++)
            if (s->edges[i].child >= 0) stack.push_back(s->edges[i].child);
    }
    return count;
}

// Checks the tree under the root against what its visits imply, for tests of searches with several
// workers: an evaluated node's visits are its evaluation's and its edges' (each visit to it but the
// first goes on down one), and so are its values; a game's end's are its result each; nothing is in
// flight once a search is done; and every node an edge leads to has been evaluated or ends the game.
// `out`: nodes checked, those whose visits don't add up, counters still in flight, nodes linked but
// not evaluated, and the largest gap between a node's sum of values and its parts' (float rounding
// apart, 0).
void search_audit(const Search* s, double* out) {
    long checked = 0, uneven = 0, flying = 0, unevaluated = 0;
    double gap = 0;
    // (node, the edge to it, -1 for the root)
    std::vector<std::pair<int32_t, int32_t>> stack{{s->root, -1}};
    while (!stack.empty()) {
        auto [index, from] = stack.back();
        stack.pop_back();
        const Node& node = s->nodes[index];
        checked++;
        long n = from < 0 ? node.n : s->edges[from].n;
        double w = from < 0 ? node.w : -s->edges[from].w;
        if (node.in_flight || (from >= 0 && s->edges[from].in_flight)) flying++;
        if (node.position.terminated) {
            gap = std::max(gap, std::fabs(w - (double)n * node.position.reward));
            continue;
        }
        if (node.num_edges == 0) {
            if (from >= 0 || n > 0) unevaluated++;
            continue;
        }
        long below = 0;
        double parts = node.value;
        for (int i = node.first_edge; i < node.first_edge + node.num_edges; i++) {
            const Edge& e = s->edges[i];
            below += e.n;
            parts += e.w;
            if (e.child >= 0) stack.push_back({e.child, i});
            else if (e.n || e.in_flight) uneven++;
        }
        if (n != 1 + below) uneven++;
        gap = std::max(gap, std::fabs(w - parts));
    }
    out[0] = (double)checked;
    out[1] = (double)uneven;
    out[2] = (double)flying;
    out[3] = (double)unevaluated;
    out[4] = gap;
}

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
// go on, and backing them up. The tree is shared with no lock over it (see `relaxed`); its virtual
// visits keep their traversals apart, as a batch's are. Python only starts a move's search and reads its result.
struct Worker {
    std::vector<float> observations;  // (batch, 2, 8, 8)
    // a request for each number of leaves, 1 to batch, their input `observations`, with OpenVINO
    std::vector<ov::InferRequest> requests;
    // or the native network's buffers, and its outputs
    std::unique_ptr<net::Scratch> scratch;
    std::vector<float> logits, values;
    Batch batch;
};

struct Engine {
    ov::Core core;
    std::vector<ov::CompiledModel> models;
    // the native network, if it has one rather than OpenVINO's
    const net::Net* network = nullptr;
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

// An engine of the native network `n` (see `cpu_net.h`), which must outlive it.
Engine* engine_new_net(const net::Net* n, int threads, int batch) {
    if (!NET_NATIVE) return nullptr;
    Engine* engine = new Engine();
    engine->network = n;
    engine->batch = batch;
    engine->workers.resize(threads);
    for (Worker& w : engine->workers) {
        w.observations.assign((size_t)batch * 128, 0.0f);
        w.scratch = std::make_unique<net::Scratch>(*n);
        w.logits.assign((size_t)batch * 65, 0.0f);
        w.values.assign(batch, 0.0f);
    }
    return engine;
}

void engine_free(Engine* engine) { delete engine; }

// Waits until a leaf the batch collided on is evaluated (the traversal in flight to it applied),
// the search stops, or 2 ms pass: tried again at once, the same traversal would likely collide again.
// It spins for about an evaluation's time, then sleeps, so as not to keep a core from the worker it
// waits on.
void await_collided(const Search& s, const Batch& b, const std::atomic<bool>& stop) {
    using clock = std::chrono::steady_clock;
    auto start = clock::now();
    while (!stop.load(std::memory_order_relaxed)) {
        for (int32_t end : b.collision_ends)
            if (acquired(s.nodes[b.collisions[end - 1].node].num_edges) > 0) return;
        auto waited = clock::now() - start;
        if (waited >= std::chrono::milliseconds(2)) return;
        if (waited < std::chrono::microseconds(50))
            std::this_thread::yield();
        else
            std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

// A search on a clock with `stop_decided` ends once its move can't change: the root's most visited
// edge leads every other by more than the visits the time left would bring at the rate it has gone
// so far (`leading_edge`). Each worker checks every so many of its backups, once the rate means
// something.
constexpr long DECIDED_CHECK_EVERY = 16;
constexpr long DECIDED_MIN_VISITS = 256;

static bool decided_on_clock(const Search& s, long visits_before, double elapsed, double left) {
    const Node& root = s.nodes[s.root];
    int count = acquired(root.num_edges);
    long done = (long)relaxed(root.n) - visits_before;
    if (count < 2 || done < DECIDED_MIN_VISITS || elapsed <= 0) return count == 1;
    long extra = (long)(done / elapsed * left) + 1;
    return leading_edge(s, root, extra) >= 0;
}

// Searches `s`'s tree with all the engine's workers: `visits` visits, or with none, until `seconds`
// pass (or, with `stop_decided`, its move is decided: `decided_on_clock`), or either way until
// halted (`search_halt`). Batches in flight then are finished. Returns the leaves evaluated.
long engine_search(Engine* engine, Search* s, int visits, double seconds, int stop_decided) {
    using clock = std::chrono::steady_clock;
    auto start = clock::now();
    auto deadline = start + std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(seconds));
    bool timed = visits <= 0;
    s->remaining = timed ? INT_MAX : visits;
    s->budget = timed ? 0 : visits;
    s->decided = -1;
    bool watch = timed && stop_decided && s->root >= 0;
    long visits_before = watch ? (long)s->nodes[s->root].n : 0;
    std::atomic<bool> stop{false};
    struct Done {
        long calls = 0, leaves = 0;
        double inferring = 0;
    };
    std::vector<Done> done(engine->workers.size());
    auto work = [&](size_t t) {
        Worker& w = engine->workers[t];
        Done& mine = done[t];
        while (true) {
            auto now = timed ? clock::now() : start;
            bool check = watch && mine.calls % DECIDED_CHECK_EVERY == 0;
            if (stop.load(std::memory_order_relaxed) || s->halted.on.load(std::memory_order_relaxed) ||
                (timed && now >= deadline) ||
                (check && decided_on_clock(*s, visits_before, std::chrono::duration<double>(now - start).count(),
                                           std::chrono::duration<double>(deadline - now).count()))) {
                stop.store(true, std::memory_order_relaxed);
                break;
            }
            // (not a stop for the others: a traversal of theirs that collides gives its visit back,
            // and goes on to make it)
            if (relaxed(s->remaining) <= 0) break;
            int n = collect(*s, w.batch, engine->batch, w.observations.data());
            if (n == 0) {
                bool collided = !w.batch.collisions.empty();
                apply(*s, w.batch, nullptr, nullptr);
                // every leaf it found is another's in flight: wait for one of them
                if (collided) await_collided(*s, w.batch, stop);
                continue;
            }
            auto t0 = clock::now();
            const float *logits, *values;
            if (engine->network) {
                for (int i = 0; i < n; i++)
                    net::forward(*engine->network, *w.scratch, w.observations.data() + (size_t)i * 128,
                                 w.logits.data() + (size_t)i * 65, w.values.data() + i);
                logits = w.logits.data();
                values = w.values.data();
            } else {
                ov::InferRequest& request = w.requests[n - 1];
                request.infer();
                logits = request.get_output_tensor(0).data<float>();
                values = request.get_output_tensor(1).data<float>();
            }
            mine.inferring += std::chrono::duration<double>(clock::now() - t0).count();
            apply(*s, w.batch, logits, values);
            mine.calls++;
            mine.leaves += n;
        }
    };
    std::vector<std::thread> threads;
    for (size_t t = 1; t < engine->workers.size(); t++) threads.emplace_back(work, t);
    work(0);
    for (auto& t : threads) t.join();
    s->remaining = 0;
    long leaves = 0;
    for (size_t t = 0; t < engine->workers.size(); t++) {
        count_up(*s, engine->workers[t].batch.tally);
        leaves += done[t].leaves;
        engine->calls += done[t].calls;
        engine->infer_seconds += done[t].inferring;
    }
    engine->leaves += leaves;
    return leaves;
}

// (calls, leaves, seconds in inference summed over workers) since the engine was made
void engine_stats(const Engine* engine, double* out) {
    out[0] = (double)engine->calls;
    out[1] = (double)engine->leaves;
    out[2] = engine->infer_seconds;
}

// whether this build has the native network (see `cpu_net.h`)
int net_supported() { return NET_NATIVE; }

// The native network, built a part at a time by `cpu_net.py`.
net::Net* net_new(int channels, int pre, int head_channels) {
    net::Net* n = new net::Net();
    n->channels = channels;
    n->pre = pre != 0;
    n->head_channels = head_channels;
    return n;
}

void net_free(net::Net* n) { delete n; }

void net_add_block(net::Net* n, int pool_channels) {
    n->blocks.emplace_back();
    n->blocks.back().pool_channels = pool_channels;
}

// A convolution: "head" (the heads' three 1x1 convolutions in one), or a block's "conv1", "conv2",
// "expand" or "project" (the stem's is `net_set_stem`). `q`, (cout, cin, k, k), its weights; `scale`
// and `bias`, of each output, what its integer sums are multiplied by and then added to; `inverse`,
// of each input channel, and `zero_point`, how its input is quantized: round(x * inverse) + zero_point.
int net_set_conv(net::Net* n, const char* where, int block, int cin, int cout, int k, const int8_t* q,
                 const float* scale, const float* bias, const float* inverse, int zero_point) {
    std::string name(where);
    net::Conv* c = name == "head"    ? &n->head
                   : name == "conv1" ? &n->blocks.at(block).conv1
                   : name == "conv2" ? &n->blocks.at(block).conv2
                   : name == "expand" ? &n->blocks.at(block).expand
                   : name == "project" ? &n->blocks.at(block).project
                                     : nullptr;
    if (!c || cin % 4 || cout % 16 || (k != 1 && k != 3)) return -1;
    c->cin = cin;
    c->cout = cout;
    c->k = k;
    if (name != "expand" && name != "project") {
        net::pack(*c, q);
    } else if (k != 1 || cin % 16 || (name == "project" && zero_point != 0)) {
        // (a project's input is ReLU's, from 0: AVX2's kernels take its zero point as 0)
        return -1;
    }
    c->scale.assign(scale, scale + cout);
    c->bias.assign(bias, bias + cout);
    c->inverse.assign(inverse, inverse + cin);
    c->zero_point = zero_point;
    if (name == "expand" || name == "project") {
#ifdef NET_SEVEN_BIT
        net::seven_bit(*c, q);
#endif
        net::pack1x1(*c, q, name == "expand");
    }
    return 0;
}

// The stem's convolution: `q`, (cout, 4, 3, 3), its 16-bit weights, the last 2 input channels padding
// (see `net::stem_table`); `scale` and `bias` as `net_set_conv`'s. Its input, the discs, is 0 or 1.
int net_set_stem(net::Net* n, int cout, const int16_t* q, const float* scale, const float* bias) {
    if (cout % 16) return -1;
    net::Conv& c = n->stem;
    c.cin = 4;
    c.cout = cout;
    c.k = 3;
    net::stem_table(c, q);
    c.scale.assign(scale, scale + cout);
    c.bias.assign(bias, bias + cout);
    return 0;
}

// A normalization (see `net::Norm`): "stem", "final", a block's "norm1" or "norm2", or the heads'
// "policy", "pool" or "value".
int net_set_norm(net::Net* n, const char* where, int block, int type, int channels, const float* a,
                 const float* c, float eps) {
    std::string name(where);
    net::Norm* norm = name == "stem"     ? &n->stem_norm
                      : name == "final"  ? &n->final_norm
                      : name == "norm1"  ? &n->blocks.at(block).norm1
                      : name == "norm2"  ? &n->blocks.at(block).norm2
                      : name == "policy" ? &n->policy_norm
                      : name == "pool"   ? &n->pool_norm
                      : name == "value"  ? &n->value_norm
                                         : nullptr;
    if (!norm) return -1;
    norm->type = type;
    norm->channels = channels;
    if (type != net::NORM_NONE) {
        norm->a.assign(a, a + channels);
        norm->c.assign(c, c + channels);
    }
    norm->eps = eps;
    return 0;
}

// A linear layer, `w` (out, in): a block's "pool", or the heads' "pool", "pass", "value_hidden" or
// "value_out".
int net_set_linear(net::Net* n, const char* where, int block, int in, int out, const float* w, const float* b) {
    std::string name(where);
    net::Linear* l = block >= 0 && name == "pool" ? &n->blocks.at(block).pool
                     : name == "pool"             ? &n->pool_linear
                     : name == "pass"             ? &n->pass_linear
                     : name == "value_hidden"     ? &n->value_hidden
                     : name == "value_out"        ? &n->value_out
                                                  : nullptr;
    if (!l) return -1;
    net::set_linear(*l, in, out, w, b);
    return 0;
}

// Makes block `block` ConvNeXt's (pre layout only), its depthwise filter of `taps` taps, `at`, (taps, 2),
// each one's (dy, dx) in order of dy then dx, within 7 squares; `w`, (taps, channels), BatchNorm's
// scale folded in, and `bias`, (channels), its offset. Its 1x1s are its "expand" and "project".
int net_set_depthwise(net::Net* n, int block, int taps, const int* at, const float* w, const float* bias) {
    if (!n->pre || taps < 1) return -1;
    net::Block& b = n->blocks.at(block);
    b.convnext = true;
    b.dw.dy.clear();
    b.dw.dx.clear();
    for (int t = 0; t < taps; t++) {
        const int dy = at[2 * t], dx = at[2 * t + 1];
        if (std::abs(dy) > 7 || std::abs(dx) > 7) return -1;
        if (t && std::make_pair(dy, dx) <= std::make_pair(b.dw.dy.back(), b.dw.dx.back())) return -1;
        b.dw.dy.push_back(dy);
        b.dw.dx.push_back(dx);
    }
    b.dw.w.assign(w, w + (size_t)taps * n->channels);
    b.dw.bias.assign(bias, bias + n->channels);
    net::prepare(b.dw);
    return 0;
}

// The policy's last 1x1 convolution, `square_w` (head_channels).
void net_set_heads(net::Net* n, const float* square_w) { n->square_w.assign(square_w, square_w + n->head_channels); }

// `count` positions, (count, 2, 8, 8), to (count, 65) logits and (count) values, split over `threads`.
void net_forward(const net::Net* n, int count, const float* obs, float* logits, float* values, int threads) {
    auto part = [&](int t) {
        net::Scratch scratch(*n);
        for (int i = t; i < count; i += threads)
            net::forward(*n, scratch, obs + (size_t)i * 128, logits + (size_t)i * 65, values + i);
    };
    std::vector<std::thread> others;
    for (int t = 1; t < threads; t++) others.emplace_back(part, t);
    part(0);
    for (auto& t : others) t.join();
}
}
