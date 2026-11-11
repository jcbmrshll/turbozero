// An exact Othello endgame solver: the final disc margin under perfect play by both sides, empties going to
// the winner (as Edax and Egaroucid count them), and a move that reaches it. See endgame.py.
//
// Negamax alpha-beta over bitboards, fail-soft. Nodes near the end are searched by the simplest means:
// the last empty square has its own count, and with a few empties left the empty squares are tried
// directly, those in regions with an odd number of empties first (parity). Higher up, moves are ordered
// fastest first (fewest replies for the opponent), searched with principal variation search (a null
// window for all but the first move), and cut by the opponent's stable discs, which bound the score. With
// enough empties a transposition table keeps bounds on positions' values; an exact solver's bounds hold
// for a position however it was reached, so the table is kept across a game's moves and shared by the
// threads searching one position.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

using u64 = uint64_t;
using Clock = std::chrono::steady_clock;

constexpr int PASS = 64;
constexpr int NO_MOVE = -1;
// beyond any score: a window of (-INF, INF) is exact
constexpr int INF = 65;
// empties at or below which squares are tried directly, without move generation
constexpr int SMALL = 4;
// empties at or above which the table is used
constexpr int TABLE_MIN = 8;
// empties at or above which children's bounds in the table are looked at before searching them
constexpr int ETC_MIN = 10;
// room for any position's moves, one to an empty square at most
constexpr int MAX_MOVES = 64;

constexpr u64 NOT_A = 0xfefefefefefefefeULL;  // no square in column a
constexpr u64 NOT_H = 0x7f7f7f7f7f7f7f7fULL;  // no square in column h
constexpr u64 CORNERS = 0x8100000000000081ULL;
constexpr u64 BORDER = 0xff818181818181ffULL;
constexpr u64 COLUMNS_AH = 0x8181818181818181ULL;
constexpr u64 ROWS_18 = 0xff000000000000ffULL;

inline int count(u64 x) { return __builtin_popcountll(x); }
inline int first(u64 x) { return __builtin_ctzll(x); }

// Squares are bits row by row, a1 = 0, b1 = 1, ..., h8 = 63. A shift by `D` moves each disc a step
// along a direction; `DEST<D>` masks off the column a step east or west wraps into.
template <int D>
constexpr u64 dest() {
    return (D == 1 || D == 9 || D == -7) ? NOT_A : (D == -1 || D == -9 || D == 7) ? NOT_H : ~0ULL;
}

template <int D>
inline u64 shift(u64 x) {
    if constexpr (D > 0)
        return x << D;
    else
        return x >> -D;
}

// `g` grown along D through the squares of `o`, as far as they run (Kogge-Stone's fill)
template <int D>
inline u64 fill(u64 g, u64 o) {
    u64 pro = o & dest<D>();
    g |= pro & shift<D>(g);
    pro &= shift<D>(pro);
    g |= pro & shift<2 * D>(g);
    pro &= shift<2 * D>(pro);
    g |= pro & shift<4 * D>(g);
    return g;
}

template <int D>
inline u64 moves_along(u64 P, u64 O) {
    return shift<D>(fill<D>(P, O) & ~P) & dest<D>();
}

template <int D>
inline u64 flips_along(u64 x, u64 P, u64 O) {
    u64 g = fill<D>(x, O);
    return (shift<D>(g) & dest<D>() & P) ? g ^ x : 0;
}

u64 moves_scalar(u64 P, u64 O) {
    u64 m = moves_along<1>(P, O) | moves_along<-1>(P, O) | moves_along<8>(P, O) | moves_along<-8>(P, O) |
            moves_along<7>(P, O) | moves_along<-7>(P, O) | moves_along<9>(P, O) | moves_along<-9>(P, O);
    return m & ~(P | O);
}

u64 flips_scalar(int sq, u64 P, u64 O) {
    u64 x = 1ULL << sq;
    return flips_along<1>(x, P, O) | flips_along<-1>(x, P, O) | flips_along<8>(x, P, O) |
           flips_along<-8>(x, P, O) | flips_along<7>(x, P, O) | flips_along<-7>(x, P, O) |
           flips_along<9>(x, P, O) | flips_along<-9>(x, P, O);
}

#if defined(__AVX2__)
// The same, four directions to a vector: lanes east (1), north (8), north-west (7) and north-east (9)
// shifted left, and their opposites shifted right.
struct Lanes {
    __m256i s1, s2, s4, left, right;
    Lanes()
        : s1(_mm256_set_epi64x(9, 7, 8, 1)),
          s2(_mm256_set_epi64x(18, 14, 16, 2)),
          s4(_mm256_set_epi64x(36, 28, 32, 4)),
          left(_mm256_set_epi64x((long long)NOT_A, (long long)NOT_H, -1LL, (long long)NOT_A)),
          right(_mm256_set_epi64x((long long)NOT_H, (long long)NOT_A, -1LL, (long long)NOT_H)) {}
};

inline u64 either(__m256i x) {
    __m128i y = _mm_or_si128(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    return (u64)(_mm_cvtsi128_si64(y) | _mm_extract_epi64(y, 1));
}

inline __m256i fill_left(const Lanes& l, __m256i g, __m256i o) {
    __m256i pro = _mm256_and_si256(o, l.left);
    g = _mm256_or_si256(g, _mm256_and_si256(pro, _mm256_sllv_epi64(g, l.s1)));
    pro = _mm256_and_si256(pro, _mm256_sllv_epi64(pro, l.s1));
    g = _mm256_or_si256(g, _mm256_and_si256(pro, _mm256_sllv_epi64(g, l.s2)));
    pro = _mm256_and_si256(pro, _mm256_sllv_epi64(pro, l.s2));
    return _mm256_or_si256(g, _mm256_and_si256(pro, _mm256_sllv_epi64(g, l.s4)));
}

inline __m256i fill_right(const Lanes& l, __m256i g, __m256i o) {
    __m256i pro = _mm256_and_si256(o, l.right);
    g = _mm256_or_si256(g, _mm256_and_si256(pro, _mm256_srlv_epi64(g, l.s1)));
    pro = _mm256_and_si256(pro, _mm256_srlv_epi64(pro, l.s1));
    g = _mm256_or_si256(g, _mm256_and_si256(pro, _mm256_srlv_epi64(g, l.s2)));
    pro = _mm256_and_si256(pro, _mm256_srlv_epi64(pro, l.s2));
    return _mm256_or_si256(g, _mm256_and_si256(pro, _mm256_srlv_epi64(g, l.s4)));
}

inline u64 moves(u64 P, u64 O) {
    static const Lanes l;
    __m256i p = _mm256_set1_epi64x((long long)P), o = _mm256_set1_epi64x((long long)O);
    __m256i gl = _mm256_andnot_si256(p, fill_left(l, p, o));
    __m256i gr = _mm256_andnot_si256(p, fill_right(l, p, o));
    __m256i m = _mm256_or_si256(_mm256_and_si256(_mm256_sllv_epi64(gl, l.s1), l.left),
                                _mm256_and_si256(_mm256_srlv_epi64(gr, l.s1), l.right));
    return either(m) & ~(P | O);
}

inline u64 flips(int sq, u64 P, u64 O) {
    static const Lanes l;
    __m256i x = _mm256_set1_epi64x((long long)(1ULL << sq));
    __m256i p = _mm256_set1_epi64x((long long)P), o = _mm256_set1_epi64x((long long)O);
    __m256i zero = _mm256_setzero_si256();
    __m256i gl = fill_left(l, x, o), gr = fill_right(l, x, o);
    // a run counts if the square past its end is the player's
    __m256i endl = _mm256_and_si256(_mm256_and_si256(_mm256_sllv_epi64(gl, l.s1), l.left), p);
    __m256i endr = _mm256_and_si256(_mm256_and_si256(_mm256_srlv_epi64(gr, l.s1), l.right), p);
    __m256i fl = _mm256_andnot_si256(_mm256_cmpeq_epi64(endl, zero), _mm256_xor_si256(gl, x));
    __m256i fr = _mm256_andnot_si256(_mm256_cmpeq_epi64(endr, zero), _mm256_xor_si256(gr, x));
    return either(_mm256_or_si256(fl, fr));
}
#else
inline u64 moves(u64 P, u64 O) { return moves_scalar(P, O); }
inline u64 flips(int sq, u64 P, u64 O) { return flips_scalar(sq, P, O); }
#endif

struct Squares {
    u64 neighbours[64];  // each square's eight around it
    u64 diagonals9[15];  // the lines running a1-h8's way
    u64 diagonals7[15];  // and h1-a8's
};

constexpr Squares make_squares() {
    Squares s{};
    for (int sq = 0; sq < 64; sq++) {
        int r = sq / 8, c = sq % 8;
        for (int dr = -1; dr <= 1; dr++)
            for (int dc = -1; dc <= 1; dc++) {
                int rr = r + dr, cc = c + dc;
                if ((dr || dc) && rr >= 0 && rr < 8 && cc >= 0 && cc < 8) s.neighbours[sq] |= 1ULL << (rr * 8 + cc);
            }
        s.diagonals9[c - r + 7] |= 1ULL << sq;
        s.diagonals7[c + r] |= 1ULL << sq;
    }
    return s;
}

constexpr Squares SQUARES = make_squares();



// The quadrants' squares; one with an odd number of empties is likely to give its last move to whoever
// moves into it first.
constexpr u64 QUADRANTS[4] = {0x000000000f0f0f0fULL, 0x00000000f0f0f0f0ULL, 0x0f0f0f0f00000000ULL,
                              0xf0f0f0f000000000ULL};

inline u64 odd_quadrants(u64 empty) {
    u64 odd = 0;
    for (u64 q : QUADRANTS)
        if (count(empty & q) & 1) odd |= q;
    return odd;
}

// The final score for the player to move: their discs less the opponent's, empties to the winner.
inline int final_score(u64 P, u64 O) {
    int p = count(P), o = count(O), diff = p - o, empty = 64 - p - o;
    return diff > 0 ? diff + empty : diff < 0 ? diff - empty : 0;
}

// The exact score with one empty square, `sq`, left.
inline int last_square(u64 P, u64 O, int sq) {
    int n = count(flips(sq, P, O));
    if (n) return 2 * (count(P) + 1 + n) - 64;
    n = count(flips(sq, O, P));
    if (n) return 64 - 2 * (count(O) + 1 + n);
    int diff = 2 * count(P) - 63;
    return diff > 0 ? diff + 1 : diff - 1;
}

// Lines along each of the four axes with no empty square.
inline u64 full_rows(u64 F) {
    u64 full = 0;
    for (int r = 0; r < 8; r++)
        if (((F >> (8 * r)) & 0xff) == 0xff) full |= 0xffULL << (8 * r);
    return full;
}

inline u64 full_columns(u64 F) {
    u64 t = F & (F >> 8);
    t &= t >> 16;
    t &= t >> 32;
    return (t & 0xff) * 0x0101010101010101ULL;
}

inline u64 full_lines(u64 F, const u64* lines) {
    u64 full = 0;
    for (int i = 0; i < 15; i++)
        if ((F & lines[i]) == lines[i]) full |= lines[i];
    return full;
}

// Discs of `C` that can never be turned: along each axis, the line through it is full, or the square next
// to it on one side is off the board or another such disc. Fewer than all of them, never more.
u64 stable(u64 C, u64 F) {
    u64 rows = full_rows(F) | COLUMNS_AH, columns = full_columns(F) | ROWS_18;
    u64 d9 = full_lines(F, SQUARES.diagonals9) | BORDER, d7 = full_lines(F, SQUARES.diagonals7) | BORDER;
    u64 S = 0;
    while (true) {
        u64 grown = S | (C & (rows | ((S << 1) & NOT_A) | ((S >> 1) & NOT_H)) & (columns | (S << 8) | (S >> 8)) &
                         (d9 | ((S << 9) & NOT_A) | ((S >> 9) & NOT_H)) & (d7 | ((S << 7) & NOT_H) | ((S >> 7) & NOT_A)));
        if (grown == S) return S;
        S = grown;
    }
}

// Bounds on positions' values, two entries to a bucket, the one with fewer empties replaced first. Each
// entry has a lock, taken without waiting: a probe of a locked entry misses, and a store to one is dropped.
struct Table {
    struct Entry {
        u64 P, O;
        int8_t lower, upper;
        uint8_t move;  // the best found, or 255
        uint8_t empties;
        std::atomic<uint8_t> busy;
    };
    std::unique_ptr<Entry[]> entries;
    u64 mask;
    bool locking;

    Table(int bits, bool locking) : entries(new Entry[1ULL << bits]), mask((1ULL << bits) - 1), locking(locking) {
        clear();
    }

    void clear() {
        for (u64 i = 0; i <= mask; i++) {
            Entry& e = entries[i];
            e.P = e.O = 0;
            e.lower = -64;
            e.upper = 64;
            e.move = 255;
            e.empties = 0;
            e.busy.store(0, std::memory_order_relaxed);
        }
    }

    static u64 hash(u64 P, u64 O) {
        u64 z = P ^ (O * 0x9e3779b97f4a7c15ULL) ^ (O >> 29);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    bool lock(Entry& e) { return !locking || !e.busy.exchange(1, std::memory_order_acquire); }
    void unlock(Entry& e) {
        if (locking) e.busy.store(0, std::memory_order_release);
    }

    struct Found {
        int lower, upper, move;
    };

    bool probe(u64 P, u64 O, Found& out) {
        u64 bucket = hash(P, O) & mask & ~1ULL;
        for (u64 i = bucket; i <= bucket + 1; i++) {
            Entry& e = entries[i];
            if (!lock(e)) continue;
            bool hit = e.P == P && e.O == O;
            if (hit) out = Found{e.lower, e.upper, e.move == 255 ? NO_MOVE : e.move};
            unlock(e);
            if (hit) return true;
        }
        return false;
    }

    void store(u64 P, u64 O, int empties, int lower, int upper, int move) {
        u64 bucket = hash(P, O) & mask & ~1ULL;
        Entry* target = nullptr;
        for (u64 i = bucket; i <= bucket + 1; i++) {
            Entry& e = entries[i];
            if (!lock(e)) continue;
            if (e.P == P && e.O == O) {
                // both bounds hold: keep the tighter of each
                e.lower = (int8_t)std::max<int>(e.lower, lower);
                e.upper = (int8_t)std::min<int>(e.upper, upper);
                if (move != NO_MOVE) e.move = (uint8_t)move;
                unlock(e);
                return;
            }
            unlock(e);
        }
        Entry& a = entries[bucket];
        Entry& b = entries[bucket + 1];
        target = a.empties <= b.empties ? &a : &b;
        if (!lock(*target)) return;
        target->P = P;
        target->O = O;
        target->lower = (int8_t)lower;
        target->upper = (int8_t)upper;
        target->move = move == NO_MOVE ? 255 : (uint8_t)move;
        target->empties = (uint8_t)empties;
        unlock(*target);
    }
};

struct Aborted {};

// One thread's search: its table (perhaps shared), when to give up, and what it's counted.
struct Context {
    Table* table;
    std::atomic<bool>* stop;
    bool timed = false;
    Clock::time_point deadline;
    u64 nodes = 0;
    // perturbs the order of moves of nearly equal promise, so threads searching the same position
    // go different ways; 0 for none
    u64 salt = 0;

    void tick() {
        if ((++nodes & 1023) == 0) {
            if (stop->load(std::memory_order_relaxed)) throw Aborted{};
            if (timed && Clock::now() >= deadline) {
                stop->store(true, std::memory_order_relaxed);
                throw Aborted{};
            }
        }
    }
};

int search(Context& c, u64 P, u64 O, int alpha, int beta, int n);

// A few empties: each tried in place, odd quadrants first. `passed`, the opponent just did.
int search_small(Context& c, u64 P, u64 O, int alpha, int beta, int n, bool passed) {
    c.tick();
    u64 empty = ~(P | O);
    if (n == 1) return last_square(P, O, first(empty));
    u64 odd = odd_quadrants(empty);
    int best = -INF;
    for (u64 part : {empty & odd, empty & ~odd}) {
        for (; part; part &= part - 1) {
            int sq = first(part);
            if (!(SQUARES.neighbours[sq] & O)) continue;
            u64 f = flips(sq, P, O);
            if (!f) continue;
            int v = -search_small(c, O ^ f, P ^ f ^ (1ULL << sq), -beta, -std::max(alpha, best), n - 1, false);
            if (v > best) {
                best = v;
                if (v >= beta) return best;
            }
        }
    }
    if (best == -INF) {
        if (passed) return final_score(P, O);
        return -search_small(c, O, P, -beta, -alpha, n, true);
    }
    return best;
}

struct Move {
    u64 P, O;  // the position after it, from the opponent's side
    int sq;
    int key;  // lower first
};

// The moves of `legal`, each with what it leads to and its key for ordering: the opponent's replies, a
// corner counting three; then moves into an odd quadrant first.
int list_moves(const Context& c, u64 P, u64 O, u64 legal, Move* out) {
    u64 odd = odd_quadrants(~(P | O));
    int n = 0;
    for (; legal; legal &= legal - 1) {
        int sq = first(legal);
        u64 f = flips(sq, P, O);
        Move& m = out[n++];
        m.sq = sq;
        m.P = O ^ f;
        m.O = P ^ f ^ (1ULL << sq);
        u64 replies = moves(m.P, m.O);
        m.key = 4 * (count(replies) + 2 * count(replies & CORNERS)) + (((odd >> sq) & 1) ? 0 : 2);
        if (c.salt) m.key += (int)((Table::hash(m.P, c.salt) >> 62) & 3);
    }
    return n;
}

// the lowest-keyed of moves[i:n], swapped into place i
inline void pick(Move* moves, int i, int n) {
    int best = i;
    for (int j = i + 1; j < n; j++)
        if (moves[j].key < moves[best].key) best = j;
    if (best != i) std::swap(moves[i], moves[best]);
}

// The bound the opponent's stable discs put on the score, if it's no more than `alpha`; else INF.
inline int stability_bound(u64 P, u64 O, int alpha) {
    if (alpha < 64 - 2 * count(O)) return INF;
    int bound = 64 - 2 * count(stable(O, P | O));
    return bound <= alpha ? bound : INF;
}

// More empties, no table: moves generated and ordered fastest first.
int search_middle(Context& c, u64 P, u64 O, int alpha, int beta, int n) {
    c.tick();
    u64 legal = moves(P, O);
    if (!legal) {
        if (!moves(O, P)) return final_score(P, O);
        return -search(c, O, P, -beta, -alpha, n);
    }
    int bound = stability_bound(P, O, alpha);
    if (bound != INF) return bound;
    Move list[MAX_MOVES];
    int count_ = list_moves(c, P, O, legal, list);
    int best = -INF;
    for (int i = 0; i < count_; i++) {
        pick(list, i, count_);
        int v = -search(c, list[i].P, list[i].O, -beta, -std::max(alpha, best), n - 1);
        if (v > best) {
            best = v;
            if (v >= beta) break;
        }
    }
    return best;
}

// Many empties: the table, and principal variation search.
int search_deep(Context& c, u64 P, u64 O, int alpha, int beta, int n) {
    c.tick();
    u64 legal = moves(P, O);
    if (!legal) {
        if (!moves(O, P)) return final_score(P, O);
        return -search(c, O, P, -beta, -alpha, n);
    }
    int bound = stability_bound(P, O, alpha);
    if (bound != INF) return bound;
    Table::Found found{-64, 64, NO_MOVE};
    if (c.table->probe(P, O, found)) {
        if (found.lower >= beta) return found.lower;
        if (found.upper <= alpha) return found.upper;
        if (found.lower == found.upper) return found.lower;
    }
    Move list[MAX_MOVES];
    int count_ = list_moves(c, P, O, legal, list);
    for (int i = 0; i < count_; i++) {
        if (list[i].sq == found.move) {
            list[i].key = -1000;
            break;
        }
    }
    if (n >= ETC_MIN) {
        // a child whose value is already known to be low enough cuts this node off without a search
        for (int i = 0; i < count_; i++) {
            Table::Found child;
            if (c.table->probe(list[i].P, list[i].O, child) && -child.upper >= beta) {
                c.table->store(P, O, n, -child.upper, 64, list[i].sq);
                return -child.upper;
            }
        }
    }
    int best = -INF, best_move = NO_MOVE;
    for (int i = 0; i < count_; i++) {
        pick(list, i, count_);
        const Move& m = list[i];
        int a = std::max(alpha, best);
        int v;
        if (i == 0) {
            v = -search(c, m.P, m.O, -beta, -a, n - 1);
        } else {
            v = -search(c, m.P, m.O, -a - 1, -a, n - 1);
            // better than the best so far, and maybe below beta: what it's worth exactly
            if (v > a && v < beta) v = -search(c, m.P, m.O, -beta, -(v - 1), n - 1);
        }
        if (v > best) {
            best = v;
            best_move = m.sq;
            if (v >= beta) break;
        }
    }
    if (best <= alpha)
        c.table->store(P, O, n, -64, best, NO_MOVE);
    else if (best >= beta)
        c.table->store(P, O, n, best, 64, best_move);
    else
        c.table->store(P, O, n, best, best, best_move);
    return best;
}

int search(Context& c, u64 P, u64 O, int alpha, int beta, int n) {
    if (n == 0) return final_score(P, O);
    if (n <= SMALL) return search_small(c, P, O, alpha, beta, n, false);
    if (n < TABLE_MIN) return search_middle(c, P, O, alpha, beta, n);
    return search_deep(c, P, O, alpha, beta, n);
}

inline int empties(u64 P, u64 O) { return 64 - count(P | O); }

struct Result {
    int score;
    int move;  // PASS for a pass, NO_MOVE once the game's over
};

// The root's moves, ordered, the table's best first.
int root_moves(Context& c, u64 P, u64 O, Move* list) {
    int n = list_moves(c, P, O, moves(P, O), list);
    Table::Found found{-64, 64, NO_MOVE};
    c.table->probe(P, O, found);
    for (int i = 0; i < n; i++)
        if (list[i].sq == found.move) list[i].key = -1000;
    std::stable_sort(list, list + n, [](const Move& a, const Move& b) { return a.key < b.key; });
    return n;
}

// Positions with no move: the game's over, or the player passes.
bool no_move(Context& c, u64 P, u64 O, int alpha, int beta, Result& out) {
    if (moves(P, O)) return false;
    if (!moves(O, P))
        out = Result{final_score(P, O), NO_MOVE};
    else
        out = Result{-search(c, O, P, -beta, -alpha, empties(P, O)), PASS};
    return true;
}

// what the root's search found, kept as any node's
void store_root(Context& c, u64 P, u64 O, int alpha, int beta, const Result& r) {
    int n = empties(P, O);
    if (r.score <= alpha)
        c.table->store(P, O, n, -64, r.score, NO_MOVE);
    else if (r.score >= beta)
        c.table->store(P, O, n, r.score, 64, r.move);
    else
        c.table->store(P, O, n, r.score, r.score, r.move);
}

// The root on one thread, in the window (alpha, beta): the first move searched in the window, the others
// first tested against the best so far. A score in the window is exact; one at or past either end of it is
// a bound, but on the right side of it.
Result solve_root(Context& c, u64 P, u64 O, int alpha, int beta) {
    Result out;
    if (no_move(c, P, O, alpha, beta, out)) return out;
    int n = empties(P, O);
    Move list[MAX_MOVES];
    int count_ = root_moves(c, P, O, list);
    int best = -INF, best_move = list[0].sq;
    for (int i = 0; i < count_; i++) {
        const Move& m = list[i];
        int a = std::max(alpha, best);
        int v;
        if (i == 0) {
            v = -search(c, m.P, m.O, -beta, -alpha, n - 1);
        } else {
            v = -search(c, m.P, m.O, -a - 1, -a, n - 1);
            if (v > a && v < beta) v = -search(c, m.P, m.O, -beta, -(v - 1), n - 1);
        }
        if (v > best) {
            best = v;
            best_move = m.sq;
            if (v >= beta) break;
        }
    }
    out = Result{best, best_move};
    store_root(c, P, O, alpha, beta, out);
    return out;
}

// The root on several threads, which share the table, as `solve_root`. All of them search the first move,
// each in its own order below it, the first to finish stopping the others. Then each takes the next move
// not yet taken, tests it against the best so far, and if it's better, searches it in the window.
Result solve_root_parallel(std::vector<Context>& contexts, u64 P, u64 O, int alpha, int beta) {
    Context& c0 = contexts[0];
    Result out;
    if (no_move(c0, P, O, alpha, beta, out)) return out;
    int n = empties(P, O);
    Move list[MAX_MOVES];
    int count_ = root_moves(c0, P, O, list);
    int threads = (int)contexts.size();
    // the threads stop on a flag of each phase's: set by a result that settles the phase, or by the
    // deadline, which doesn't settle it
    std::atomic<bool> stop{false};
    std::atomic<bool> settled{false};
    std::atomic<bool>* outer = c0.stop;
    for (Context& c : contexts) c.stop = &stop;
    auto restore = [&] {
        for (Context& c : contexts) c.stop = outer;
    };
    auto run = [&](auto work) {
        std::vector<std::thread> pool;
        for (int t = 1; t < threads; t++) pool.emplace_back(work, t);
        work(0);
        for (auto& t : pool) t.join();
        if (stop && !settled) {
            restore();
            throw Aborted{};
        }
    };

    // the first move, by all
    int first_value = 0;
    run([&](int t) {
        try {
            int v = -search(contexts[t], list[0].P, list[0].O, -beta, -alpha, n - 1);
            if (!settled.exchange(true)) first_value = v;
            stop = true;
        } catch (Aborted&) {
        }
    });
    int best = first_value, best_move = list[0].sq;
    if (best < beta) {
        // the rest, a move to a thread at a time, until one reaches beta
        stop = false;
        settled = false;
        std::mutex lock;
        std::atomic<int> next{1};
        run([&](int t) {
            Context& c = contexts[t];
            try {
                for (int i = next++; i < count_; i = next++) {
                    const Move& m = list[i];
                    int a;
                    {
                        std::lock_guard<std::mutex> held(lock);
                        a = std::max(alpha, best);
                    }
                    int v = -search(c, m.P, m.O, -a - 1, -a, n - 1);
                    if (v <= a) continue;
                    // better than the best when it started: by how much
                    if (v < beta) v = -search(c, m.P, m.O, -beta, -(v - 1), n - 1);
                    std::lock_guard<std::mutex> held(lock);
                    if (v > best) {
                        best = v;
                        best_move = m.sq;
                    }
                    if (v >= beta) {
                        settled = true;
                        stop = true;
                    }
                }
            } catch (Aborted&) {
            }
        });
    }
    restore();
    out = Result{best, best_move};
    store_root(c0, P, O, alpha, beta, out);
    return out;
}

// Every line of play to the end, without pruning, by the plain rules: what the solver is checked against.
int minimax(u64 P, u64 O, bool passed) {
    u64 legal = moves_scalar(P, O);
    if (!legal) return passed ? final_score(P, O) : -minimax(O, P, true);
    int best = -INF;
    for (; legal; legal &= legal - 1) {
        int sq = first(legal);
        u64 f = flips_scalar(sq, P, O);
        best = std::max(best, -minimax(O ^ f, P ^ f ^ (1ULL << sq), false));
    }
    return best;
}

}  // namespace

// A solver for one position at a time, on `threads` threads sharing a table of 2^table_bits entries
// (24 bytes each), kept until it's freed: what it learns of one position helps with those after it.
struct Solver {
    std::unique_ptr<Table> table;
    std::vector<Context> contexts;
    std::atomic<bool> stop{false};
    u64 nodes = 0;
};

extern "C" {

Solver* endgame_solver_new(int threads, int table_bits) {
    Solver* s = new Solver();
    threads = std::max(threads, 1);
    s->table = std::make_unique<Table>(table_bits, threads > 1);
    s->contexts.resize(threads);
    for (int t = 0; t < threads; t++) {
        Context& c = s->contexts[t];
        c.table = s->table.get();
        c.stop = &s->stop;
        c.salt = t == 0 ? 0 : Table::hash((u64)t, 0x5eed);
    }
    return s;
}

void endgame_solver_free(Solver* s) { delete s; }

// Solves the position with `me` to move against `opp`: writes the score for the player to move and a best
// move (0-63, 64 to pass, -1 if the game's over). The score is exact, or without `exact`, right in sign
// only: whether it's a win, a draw or a loss, the move then winning or drawing if the position does. Gives
// up after `seconds`, if more than 0. Returns 1 if it solved it, 0 if it gave up.
int endgame_solver_solve(Solver* s, uint64_t me, uint64_t opp, double seconds, int exact, int* score, int* move) {
    int alpha = exact ? -INF : -1, beta = exact ? INF : 1;
    s->stop = false;
    auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    for (Context& c : s->contexts) {
        c.timed = seconds > 0;
        c.deadline = deadline;
        c.nodes = 0;
        c.stop = &s->stop;
    }
    int solved = 1;
    try {
        auto root = [&](int a, int b) {
            return s->contexts.size() == 1 ? solve_root(s->contexts[0], me, opp, a, b)
                                           : solve_root_parallel(s->contexts, me, opp, a, b);
        };
        Result r = root(alpha, beta);
        *score = r.score;
        *move = r.move;
    } catch (Aborted&) {
        solved = 0;
    }
    for (const Context& c : s->contexts) s->nodes += c.nodes;
    return solved;
}

// nodes searched since the solver was made
uint64_t endgame_solver_nodes(const Solver* s) { return s->nodes; }

// `n` positions, each solved on one of `threads` threads, each thread with its own table of
// 2^table_bits entries: the score of each for the player to move, and a best move (as
// `endgame_solver_solve`).
void endgame_solve_batch(const uint64_t* me, const uint64_t* opp, int n, int threads, int table_bits, int exact,
                         int8_t* scores, int8_t* moves) {
    int alpha = exact ? -INF : -1, beta = exact ? INF : 1;
    std::atomic<int> next{0};
    auto work = [&] {
        Table table(table_bits, false);
        std::atomic<bool> stop{false};
        Context c;
        c.table = &table;
        c.stop = &stop;
        for (int i = next++; i < n; i = next++) {
            Result r = solve_root(c, me[i], opp[i], alpha, beta);
            scores[i] = (int8_t)r.score;
            moves[i] = (int8_t)r.move;
        }
    };
    threads = std::max(1, std::min(threads, n));
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; t++) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
}

// The exact score of each legal move of the position, from the mover's side, into `out` (65): -128 for
// those that aren't legal; a pass is 64, legal only with no other move. All -128 once the game's over.
void endgame_move_scores(uint64_t me, uint64_t opp, int table_bits, int8_t* out) {
    Table table(table_bits, false);
    std::atomic<bool> stop{false};
    Context c;
    c.table = &table;
    c.stop = &stop;
    for (int i = 0; i <= PASS; i++) out[i] = -128;
    int n = empties(me, opp);
    u64 legal = moves(me, opp);
    if (!legal) {
        if (moves(opp, me)) out[PASS] = (int8_t)-search(c, opp, me, -INF, INF, n);
        return;
    }
    for (; legal; legal &= legal - 1) {
        int sq = first(legal);
        u64 f = flips(sq, me, opp);
        out[sq] = (int8_t)-search(c, opp ^ f, me ^ f ^ (1ULL << sq), -INF, INF, n - 1);
    }
}

// For tests: legal moves and the discs a move turns, by the fast and plain ways; stable discs; and the
// score by plain minimax.
uint64_t endgame_moves(uint64_t me, uint64_t opp) { return moves(me, opp); }
uint64_t endgame_moves_scalar(uint64_t me, uint64_t opp) { return moves_scalar(me, opp); }
uint64_t endgame_flips(int sq, uint64_t me, uint64_t opp) { return flips(sq, me, opp); }
uint64_t endgame_flips_scalar(int sq, uint64_t me, uint64_t opp) { return flips_scalar(sq, me, opp); }
uint64_t endgame_stable(uint64_t me, uint64_t opp) { return stable(me, me | opp); }
int endgame_minimax(uint64_t me, uint64_t opp) { return minimax(me, opp, false); }
}
