// Checks of `cpu_net.h`'s pieces that must compute exactly what a plainer computation of the same
// numbers does, bit for bit, at the level it's compiled for (see `test_cpu_search.py`). Prints each
// failure; exits 1 if any.
#include <cstdio>
#include <random>

#include "cpu_net.h"

using namespace net;

static int failures = 0;

static void expect(bool ok, const char* what, int i) {
    if (!ok && failures++ < 20) std::printf("%s: wrong at %d\n", what, i);
}

static bool same(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

static std::mt19937 rng(1);

// floats of many magnitudes and signs, some -0, so that the order of a sum shows in its rounding
static float any_float() {
    if (rng() % 16 == 0) return -0.0f;
    return std::ldexp(std::uniform_real_distribution<float>(-1, 1)(rng), (int)(rng() % 40) - 20);
}

// each vector's sum as each level summed one vector's lanes before (`Lanes::reduce`)
static float sum_lanes(F v) {
#if NET_ISA == 2
    return _mm512_reduce_add_ps(v);
#elif NET_ISA == 1
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    return _mm_cvtss_f32(_mm_add_ss(x, _mm_movehdup_ps(x)));
#else
    return vaddvq_f32(v);
#endif
}

static void check_reduce() {
    for (int round = 0; round < 200; round++) {
        alignas(64) float x[W][W], got[W];
        F v[W];
        for (int s = 0; s < W; s++) {
            for (int i = 0; i < W; i++) x[s][i] = any_float();
            v[s] = Lanes::load(x[s]);
        }
        Lanes::store(got, Lanes::reduce(v));
        for (int s = 0; s < W; s++) expect(same(got[s], sum_lanes(v[s])), "reduce", s);
    }
}

static void check_norm_stats() {
    for (int ch : {16, 32, 48}) {
        std::vector<float> x((size_t)SQUARES * (ch + 8));
        for (auto& v : x) v = any_float();
        const int stride = ch + 8;
        alignas(64) float mean[SQUARES], rstd[SQUARES];
        norm_stats(x.data(), stride, ch, 1e-5f, mean, rstd);
        for (int s = 0; s < SQUARES; s++) {
            // a square at a time, as the normalizations summed before
            const float* v = x.data() + s * stride;
            F sum = Lanes::zero();
            for (int i = 0; i < ch; i += W) sum = Lanes::add(sum, Lanes::load(v + i));
            float m = sum_lanes(sum) / ch;
            F sq = Lanes::zero();
            for (int i = 0; i < ch; i += W) {
                F d = Lanes::sub(Lanes::load(v + i), Lanes::set1(m));
                sq = Lanes::fma(d, d, sq);
            }
            expect(same(mean[s], m), "norm mean", s);
            expect(same(rstd[s], 1.0f / std::sqrt(sum_lanes(sq) / ch + 1e-5f)), "norm rstd", s);
        }
    }
}

// the stem's table against its convolution summed directly, on random boards
static void check_stem() {
    for (int cout : {16, 32, 48}) {
        Conv c;
        c.cin = 4;
        c.cout = cout;
        c.k = 3;
        std::vector<int8_t> q((size_t)cout * 4 * 9, 0);
        for (int o = 0; o < cout; o++)
            for (int p = 0; p < 2; p++)
                for (int t = 0; t < 9; t++) q[((size_t)o * 4 + p) * 9 + t] = (int8_t)((int)(rng() % 255) - 127);
        stem_table(c, q.data());
        for (int o = 0; o < cout; o++) {
            c.scale.push_back(std::ldexp(1.0f + (rng() % 1000) / 1000.0f, -10));
            c.bias.push_back(any_float());
        }
        for (int board = 0; board < 50; board++) {
            float obs[128];
            for (int sq = 0; sq < SQUARES; sq++) {
                int r = (int)(rng() % 3);
                obs[sq] = r == 1;
                obs[SQUARES + sq] = r == 2;
            }
            std::vector<float> got((size_t)SQUARES * cout);
            stem(c, obs, got.data());
            for (int sq = 0; sq < SQUARES; sq++)
                for (int o = 0; o < cout; o++) {
                    int acc = 0;
                    for (int t = 0; t < 9; t++) {
                        int y = sq / 8 + t / 3 - 1, x = sq % 8 + t % 3 - 1;
                        if (y < 0 || y > 7 || x < 0 || x > 7) continue;
                        for (int p = 0; p < 2; p++) acc += q[((size_t)o * 4 + p) * 9 + t] * (int)obs[p * SQUARES + y * 8 + x];
                    }
                    expect(same(got[(size_t)sq * cout + o], std::fma((float)acc, c.scale[o], c.bias[o])), "stem", sq);
                }
        }
    }
}

// a convolution's integer sums, any order, then acc * scale + bias rounded once, added to what's there
// for a residual (only ever a 3x3); channels that aren't a multiple of 16 run a kernel row's taps into
// one another
static void check_convolve() {
    for (int k : {3, 1})
        for (int cin : {16, 40, 64})
            for (int cout : {32, 48})
                for (bool add : {false, true}) {
                    if (add && k == 1) continue;
                    Conv c;
                    c.cin = cin;
                    c.cout = cout;
                    c.k = k;
                    const int taps = k * k;
                    std::vector<int8_t> q((size_t)cout * cin * taps);
                    for (auto& w : q) w = (int8_t)((int)(rng() % 255) - 127);
                    pack(c, q.data());
                    for (int o = 0; o < cout; o++) {
                        c.scale.push_back(std::ldexp(1.0f + (rng() % 1000) / 1000.0f, -10));
                        c.bias.push_back(any_float());
                    }
                    Net n;
                    n.channels = cin;
                    n.head_channels = cout;
                    Scratch s(n);
                    // inputs anywhere in their 8 bits, the border too
                    for (auto& a : s.padded) a = (Activation)((int)(rng() % 256) - ACTIVATION_SHIFT);
                    std::vector<float> before((size_t)SQUARES * cout), got;
                    for (auto& v : before) v = any_float();
                    got = before;
                    convolve(c, s, got.data(), cout, add);
                    for (int sq = 0; sq < SQUARES; sq++)
                        for (int o = 0; o < cout; o++) {
                            int acc = 0;
                            for (int t = 0; t < taps; t++) {
                                const int dy = k == 3 ? t / 3 : 1, dx = k == 3 ? t % 3 : 1;
                                const Activation* in = s.padded.data() + ((sq / 8 + dy) * 10 + sq % 8 + dx) * cin;
                                for (int ch = 0; ch < cin; ch++)
                                    acc += q[((size_t)o * cin + ch) * taps + t] * ((int)in[ch] + ACTIVATION_SHIFT);
                            }
                            const float v = std::fma((float)acc, c.scale[o], c.bias[o]);
                            const size_t at = (size_t)sq * cout + o;
                            expect(same(got[at], add ? before[at] + v : v), "convolve", (int)at);
                        }
                }
}

#if NET_ISA != 3
static void check_transpose() {
    alignas(64) float x[W][W], got[W];
    F r[W];
    for (int j = 0; j < W; j++) {
        for (int i = 0; i < W; i++) x[j][i] = (float)(j * W + i);
        r[j] = Lanes::load(x[j]);
    }
    Lanes::transpose(r);
    for (int i = 0; i < W; i++) {
        Lanes::store(got, r[i]);
        for (int j = 0; j < W; j++) expect(got[j] == x[j][i], "transpose", i * W + j);
    }
}

// a vector of outputs at a time against one at a time
static void check_linear() {
    for (int in : {16, 64}) {
        const int out = 3 * W + 5;
        std::vector<float> w((size_t)in * out), b(out);
        for (auto& v : w) v = any_float();
        for (auto& v : b) v = any_float();
        Linear l;
        set_linear(l, in, out, w.data(), b.data());
        std::vector<float> x(in), got(l.out);
        for (auto& v : x) v = any_float();
        linear(l, x.data(), got.data());
        for (int o = 0; o < l.out; o++) {
            Linear one;
            set_linear(one, in, 1, w.data() + (size_t)o * in, b.data() + o);
            float want;
            linear(one, x.data(), &want);
            expect(same(got[o], want), "linear", o);
        }
    }
}

// a square at a time, its products rounded on their own
static void check_policy_logits() {
    const int H = 32, stride = 3 * H;
    std::vector<float> h((size_t)SQUARES * stride), bias(H), w(H);
    for (auto& v : h) v = any_float();
    for (auto& v : bias) v = any_float();
    for (auto& v : w) v = any_float();
    Norm n;
    n.type = NORM_CHANNEL;
    n.channels = H;
    for (int i = 0; i < H; i++) {
        n.a.push_back(any_float());
        n.c.push_back(any_float());
    }
    float got[SQUARES];
    policy_logits(n, h.data(), stride, H, bias.data(), w.data(), got);
    alignas(64) float mean[SQUARES], rstd[SQUARES];
    norm_stats(h.data(), stride, H, n.eps, mean, rstd);
    for (int s = 0; s < SQUARES; s++) {
        alignas(64) float p[H];
        for (int i = 0; i < H; i += W)
            Lanes::store(p + i, Lanes::fma(Lanes::mul(Lanes::sub(Lanes::load(h.data() + s * stride + i), Lanes::set1(mean[s])),
                                                      Lanes::set1(rstd[s])),
                                           Lanes::load(n.a.data() + i), Lanes::load(n.c.data() + i)));
        float logit = 0;
        for (int i = 0; i < H; i++) {
            volatile float product = std::max(p[i] + bias[i], 0.0f) * w[i];
            logit = logit + product;
        }
        expect(same(got[s], logit), "policy logits", s);
    }
}
#endif

// 3x3 convolutions against their sums taken directly, each output fma(sum, scale, bias) (and the
// residual added after), at widths of 16s and of 4s (a block's second convolution), and at the
// extremes of inputs and weights, where AVX2's transformed values come nearest 16 bits' range
static void check_convolve() {
    const int shapes[][2] = {{32, 32}, {24, 48}, {40, 48}, {64, 64}, {128, 128}, {12, 16}};
    for (auto [cin, cout] : shapes)
        for (int extreme = 0; extreme < 2; extreme++) {
            Conv c;
            c.cin = cin, c.cout = cout, c.k = 3;
            std::vector<int8_t> q((size_t)cout * cin * 9);
            for (auto& w : q) w = (int8_t)(extreme ? (rng() % 2 ? 127 : -127) : (int)(rng() % 255) - 127);
            pack(c, q.data());
            for (int o = 0; o < cout; o++) {
                c.scale.push_back(std::ldexp(1.0f + (float)(rng() % 1000) / 1000, -(int)(rng() % 20)));
                c.bias.push_back(any_float());
            }
            Net n;
            n.channels = std::max(cin, cout);
            n.head_channels = 16;
            Scratch s(n);
            std::vector<int> x((size_t)PADDED * cin);  // the inputs, 0 to 255
            for (size_t i = 0; i < x.size(); i++) {
                x[i] = extreme ? (rng() % 2 ? 255 : 0) : (int)(rng() % 256);
                s.padded[i] = (Activation)(x[i] - ACTIVATION_SHIFT);
            }
            for (int add = 0; add < 2; add++) {
                std::vector<float> out((size_t)SQUARES * cout), before(out.size());
                for (auto& v : before) v = any_float();
                out = before;
                convolve(c, s, out.data(), cout, add);
                for (int sq = 0; sq < SQUARES; sq++)
                    for (int o = 0; o < cout; o++) {
                        int32_t sum = 0;
                        for (int t = 0; t < 9; t++)
                            for (int i = 0; i < cin; i++)
                                sum += q[((size_t)o * cin + i) * 9 + t] *
                                       x[((sq / 8 + t / 3) * 10 + sq % 8 + t % 3) * cin + i];
                        float want = std::fma((float)sum, c.scale[o], c.bias[o]);
                        if (add) want = before[sq * cout + o] + want;
                        expect(same(out[sq * cout + o], want), "convolve", (cin * 1000 + sq) * 1000 + o);
                    }
            }
        }
}

int main() {
    check_convolve();
    check_reduce();
    check_norm_stats();
    check_stem();
    check_convolve();
#if NET_ISA != 3
    check_transpose();
    check_linear();
    check_policy_logits();
#endif
    std::printf("%d failures\n", failures);
    return failures ? 1 : 0;
}
