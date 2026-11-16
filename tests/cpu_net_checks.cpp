// Checks of `cpu_net.h`'s pieces that must compute exactly what a plainer computation of the same
// numbers does, bit for bit, at the level it's compiled for (see `test_cpu_search.py`). Prints each
// failure; exits 1 if any.
#include <climits>
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

// a block's pooling bias and its second convolution's input in one pass, against the steps it replaces:
// the normalization in place, a ReLU of the pooled channels, their means and maxima a square at a time,
// and the kept channels plus the bias a vector at a time, for both layouts and each type of norm
static void check_pool_bias_quantize() {
    const int ch = 48, stride = 64;
    for (int count : {8, 16, 24})
        for (int type : {NORM_NONE, NORM_AFFINE, NORM_CHANNEL})
            for (bool pre : {true, false}) {
                const int kept = ch - count;
                std::vector<float> y((size_t)SQUARES * stride);
                for (auto& v : y) v = any_float();
                y[5 * stride + ch - 1] = NAN;
                Norm n;
                n.type = type;
                n.channels = ch;
                for (int i = 0; i < ch; i++) {
                    n.a.push_back(any_float());
                    n.c.push_back(any_float());
                }
                std::vector<float> w(2 * count * kept), b(kept);
                for (auto& v : w) v = any_float();
                for (auto& v : b) v = any_float();
                Linear pool;
                set_linear(pool, 2 * count, kept, w.data(), b.data());
                Conv c;
                c.cin = kept;
                c.zero_point = 3;
                for (int i = 0; i < kept; i++) c.inverse.push_back(std::ldexp(1.0f, (int)(rng() % 12) - 4));
                std::vector<Activation> got((size_t)PADDED * kept + 16), want(got.size());
                pool_bias_quantize(n, pool, y.data(), stride, ch, pre, c, got.data());

                normalize(n, y.data(), stride, ch, pre);
                if (!pre)
                    for (int s = 0; s < SQUARES; s++)
                        for (int i = 0; i < count; i += W) {
                            float* v = y.data() + s * stride + kept + i;
                            Lanes::store(v, Lanes::relu(Lanes::load(v, count - i)), count - i);
                        }
                std::vector<float> pooled(2 * count), bias(kept);
                for (int i = 0; i < count; i++) {
                    float sum = 0, most = -INFINITY;
                    for (int s = 0; s < SQUARES; s++) {
                        const float v = y[(size_t)s * stride + kept + i];
                        sum += v;
                        // AVX-512's maximum takes a NaN in either from v, the others' from most
                        most = NET_ISA == 2 ? (most > v ? most : v) : (v > most ? v : most);
                    }
                    pooled[i] = sum * (1.0f / SQUARES);
                    pooled[count + i] = most;
                }
                linear(pool, pooled.data(), bias.data());
                border(c, want.data());
                const Quantizer quantize(c);
                for (int s = 0; s < SQUARES; s++)
                    for (int i = 0; i < kept; i += W) {
                        const int m = kept - i;
                        F v = Lanes::add(Lanes::load(y.data() + s * stride + i, m), Lanes::load(bias.data() + i, m));
                        quantize(pre ? v : Lanes::relu_sum(v), i, padded_square(want.data(), s, kept), std::min(m, W));
                    }
                for (size_t i = 0; i < (size_t)PADDED * kept; i++) expect(got[i] == want[i], "pool bias quantize", (int)i);
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

// a 3x3 convolution's integer sums, any order, then acc * scale + bias rounded once, added to what's
// there for a residual; channels that aren't a multiple of 16 run a kernel row's taps into one another,
// and AVX2's transformed input from one tile into the next; at 128 channels its planes are padded; at
// the extremes of inputs and weights its transformed values come nearest 16 bits' range (the heads'
// 1x1: `check_conv_1x1`)
static void check_convolve() {
    for (int cin : {12, 16, 24, 40, 64, 128})
        for (int cout : {32, 48})
            for (int extreme = 0; extreme < 2; extreme++)
            for (bool add : {false, true}) {
                Conv c;
                c.cin = cin;
                c.cout = cout;
                c.k = 3;
                const int taps = 9;
                std::vector<int8_t> q((size_t)cout * cin * taps);
                for (auto& w : q) w = (int8_t)(extreme ? (rng() % 2 ? 127 : -127) : (int)(rng() % 255) - 127);
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
                for (auto& a : s.padded) a = (Activation)((extreme ? (rng() % 2) * 255 : (int)(rng() % 256)) - ACTIVATION_SHIFT);
                std::vector<float> before((size_t)SQUARES * cout), got;
                for (auto& v : before) v = any_float();
                got = before;
                convolve(c, s, got.data(), cout, add);
                for (int sq = 0; sq < SQUARES; sq++)
                    for (int o = 0; o < cout; o++) {
                        int acc = 0;
                        for (int t = 0; t < taps; t++) {
                            const int dy = t / 3, dx = t % 3;
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

// a convolution's input as each level quantized it a vector at a time before (a44789c): x * inverse
// rounded, ties to even, plus the zero point, within 0 to 255, out-of-range values and NaN each level's
// own way
static int quantized(float x, float inverse, int zero_point) {
    volatile float t = x * inverse;
    const float v = t;
#if NET_ISA == 2
    // the conversion takes a NaN and floats out of range to INT_MIN; the zero point's add wraps
    int32_t i = std::isnan(v) || v >= 2147483648.0f || v < -2147483648.0f ? INT32_MIN : (int32_t)std::nearbyint(v);
    return std::clamp((int32_t)((uint32_t)i + (uint32_t)zero_point), 0, 255);
#elif NET_ISA == 1
    // clamped first, a NaN to the low bound (max's second operand)
    const float low = -zero_point - 1.0f, high = 256.0f - zero_point;
    const float c = std::isnan(v) ? low : std::min(std::max(v, low), high);
    return std::clamp((int)std::nearbyint(c) + zero_point, 0, 255);
#else
    // a saturating conversion (a NaN to 0), add of the zero point less 128 and narrowing
    long long i = std::isnan(v) ? 0 : v >= 2147483647.0f ? INT_MAX : v <= -2147483648.0f ? INT_MIN : (long long)std::nearbyint(v);
    i = std::clamp(i + zero_point - 128, (long long)INT_MIN, (long long)INT_MAX);
    return (int)std::clamp(i, -128LL, 127LL) + 128;
#endif
}

// `Quantizer`'s packing of whole squares (`quantize_square`, 48 channels) and
// of a vector at a time, against `quantized`, on values about each bound, ties, infinities and NaN
static void check_quantize() {
    const int cin = 48;
    for (int zero_point : {0, 1, 3, 40, 128, 200, 255}) {
        std::vector<float> x, inverse;
        for (float v : {0.0f, -0.0f, 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, 1e10f, -1e10f, INFINITY, -INFINITY, NAN, -NAN,
                        2147483520.0f, 2147483648.0f, -2147483648.0f, -2147483904.0f, 3e9f, 0x1p-130f})
            x.push_back(v);
        for (float b : {(float)-zero_point, 255.0f - zero_point})
            for (float d = -2; d <= 2; d += 0.25f) x.push_back(b + d);
        inverse.assign(x.size(), 1.0f);
        while (x.size() % cin) {
            x.push_back(any_float() * 300);
            inverse.push_back(std::ldexp(1.0f + (rng() % 1000) / 1000.0f, (int)(rng() % 8)));
        }
        for (size_t first = 0; first < x.size(); first += cin) {
            Conv c;
            c.cin = cin;
            c.zero_point = zero_point;
            for (int i = 0; i < cin; i++) c.inverse.push_back(inverse[first + i]);
            const Quantizer quantize(c);
            alignas(64) Activation got[cin], one[cin];
            quantize_square(quantize, cin, got, [&](int i) { return Lanes::load(x.data() + first + i); });
            for (int i = 0; i < cin; i += W) quantize(Lanes::load(x.data() + first + i), i, one);
            for (int i = 0; i < cin; i++) {
                const int want = quantized(x[first + i], inverse[first + i], zero_point);
                expect(got[i] + ACTIVATION_SHIFT == want, "quantize square", (int)first + i);
                expect(one[i] + ACTIVATION_SHIFT == want, "quantize vector", (int)first + i);
            }
#if NET_ISA != 3
            // a vector's first 4 lanes (add_quantize's last channels)
            Activation part[W];
            std::fill(part, part + W, (Activation)7);
            quantize(Lanes::load(x.data() + first), 0, part, 4);
            for (int i = 0; i < W; i++)
                expect(part[i] == (i < 4 ? quantized(x[first + i], inverse[first + i], zero_point) : 7), "quantize part", i);
#endif
        }
    }
}

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

// a 1x1 convolution (the heads') against its sums done one by one, with the same rounding: every
// input value and weight, 16 to 128 channels, the heads' 48 to 144 outputs
static void check_conv_1x1() {
    for (int cin : {16, 32, 64, 128})
        for (int cout : {48, 96, 144}) {
            Conv c;
            c.cin = cin;
            c.cout = cout;
            c.k = 1;
            std::vector<int8_t> q((size_t)cout * cin);
            for (auto& v : q) v = (int8_t)((int)(rng() % 255) - 127);
            pack(c, q.data());
            for (int o = 0; o < cout; o++) {
                c.scale.push_back(any_float());
                c.bias.push_back(any_float());
            }
            Net n;
            n.channels = cin;
            n.head_channels = cout / 3;
            Scratch s(n);
            for (auto& v : s.padded) v = (Activation)((int)(rng() % 256) - ACTIVATION_SHIFT);
            std::vector<float> got((size_t)SQUARES * cout);
            convolve(c, s, got.data(), cout);
            for (int sq = 0; sq < SQUARES; sq++) {
                const Activation* a = padded_square(s.padded.data(), sq, cin);
                for (int o = 0; o < cout; o++) {
                    int32_t acc = 0;
                    for (int i = 0; i < cin; i++) acc += ((int)a[i] + ACTIVATION_SHIFT) * q[(size_t)o * cin + i];
                    expect(same(got[(size_t)sq * cout + o], std::fma((float)acc, c.scale[o], c.bias[o])), "conv_1x1", sq);
                }
            }
        }
}

int main() {
    check_reduce();
    check_norm_stats();
    check_stem();
    check_convolve();
    check_conv_1x1();
    check_quantize();
    check_pool_bias_quantize();
    check_transpose();
    check_linear();
    check_policy_logits();
    std::printf("%d failures\n", failures);
    return failures ? 1 : 0;
}
