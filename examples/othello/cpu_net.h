// An Othello ResNet's forward pass on CPU, int8 in the trunk's convolutions (AVX-512 VNNI) and float
// everywhere else, one position at a time: what `cpu_net.py` builds from a trained network, for
// `cpu_search.cpp`'s engine. It covers basic blocks in either layout, with BatchNorm folded into the
// convolutions or normalizations between them, global pooling biases, and the pooled heads (see
// `core.networks.azresnet`).
//
// Each convolution takes 8-bit unsigned inputs (a scale and a zero point for the whole tensor, from
// calibration) and 8-bit signed weights (a scale for each output channel), sums in 32-bit integers, and
// writes float outputs, acc * scale + bias, which per-square normalizations, the residual sum and the
// global pooling then take in float. A normalization of each square's channels (ConvNeXt's LayerNorm)
// costs about what a ReLU does here, where OpenVINO runs it as separate ops.
#pragma once

// the native network needs AVX-512 (F, BW, VL) and VNNI; without them the library builds without it
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#define NET_NATIVE 1
#else
#define NET_NATIVE 0
#endif

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace net {

constexpr int SQUARES = 64;
// the input of a 3x3 convolution: the board with a border of a square all round
constexpr int PADDED = 100;

struct Conv {
    int cin = 0, cout = 0, k = 3;
    // [cout / 32][k * k][cin / 4][2][16 outputs][4 inputs], the last 16 outputs alone if cout / 16 is odd
    std::vector<int8_t> packed;
    std::vector<float> scale, bias;  // of each output: acc * scale + bias
    float inv_in_scale = 1;
    int zero_point = 0;
};

enum NormType { NORM_NONE = 0, NORM_AFFINE = 1, NORM_CHANNEL = 2 };

// of each square's channels: none; x * a + c; or (x - mean) / sqrt(var + eps) * a + c, the mean and
// variance over the square's channels
struct Norm {
    int type = NORM_NONE;
    int channels = 0;
    std::vector<float> a, c;
    float eps = 1e-5f;
};

struct Linear {
    int in = 0, out = 0;
    std::vector<float> w;  // (out, in)
    std::vector<float> b;
};

struct Block {
    Conv conv1, conv2;
    Norm norm1, norm2;
    int pool_channels = 0;
    Linear pool;
};

struct Net {
    int channels = 0;
    bool pre = false;
    int head_channels = 0;
    Conv stem;
    Norm stem_norm;
    std::vector<Block> blocks;
    Norm final_norm;
    // the heads' three 1x1 convolutions in one, (channels, 3 * head_channels): policy, its pooled
    // part, value; in float
    std::vector<float> head_w, head_b;
    Norm policy_norm, pool_norm, value_norm;
    Linear pool_linear, pass_linear, value_hidden, value_out;
    std::vector<float> square_w;  // (head_channels)
};

// what a forward pass writes as it goes, one to a thread
struct Scratch {
    std::vector<uint8_t> padded;  // (100, max channels), a convolution's input
    std::vector<float> x, y, h;   // (64, max channels): the trunk, a block's middle, the heads
    explicit Scratch(const Net& n) {
        int c = std::max(n.channels, 4);
        padded.assign((size_t)PADDED * c, 0);
        x.assign((size_t)SQUARES * c, 0);
        y.assign((size_t)SQUARES * c, 0);
        h.assign((size_t)SQUARES * std::max(c, 3 * n.head_channels), 0);
    }
};

#if NET_NATIVE
inline int32_t load_u32(const uint8_t* p) {
    int32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// One output row of a convolution, `RO` blocks of 16 outputs from `o`, for the 8 squares of row `y`.
template <int K, int RO>
inline void conv_row(const Conv& c, const int8_t* w, const uint8_t* in, int y, int o, float* out, int out_stride) {
    const int c4n = c.cin / 4;
    __m512i acc[8][RO];
    for (int i = 0; i < 8; i++)
        for (int r = 0; r < RO; r++) acc[i][r] = _mm512_setzero_si512();
    for (int t = 0; t < K * K; t++) {
        int dy = K == 3 ? t / 3 : 1, dx = K == 3 ? t % 3 : 1;
        const uint8_t* row = in + ((y + dy) * 10 + dx) * c.cin;
        const int8_t* wt = w + (size_t)t * c4n * RO * 64;
#pragma GCC unroll 2
        for (int c4 = 0; c4 < c4n; c4++) {
            __m512i a[8];
            for (int i = 0; i < 8; i++) a[i] = _mm512_set1_epi32(load_u32(row + i * c.cin + c4 * 4));
            for (int r = 0; r < RO; r++) {
                __m512i wv = _mm512_loadu_si512(wt + (c4 * RO + r) * 64);
                for (int i = 0; i < 8; i++) acc[i][r] = _mm512_dpbusd_epi32(acc[i][r], a[i], wv);
            }
        }
    }
    for (int r = 0; r < RO; r++) {
        __m512 m = _mm512_loadu_ps(c.scale.data() + o + 16 * r);
        __m512 b = _mm512_loadu_ps(c.bias.data() + o + 16 * r);
        for (int i = 0; i < 8; i++)
            _mm512_storeu_ps(out + (y * 8 + i) * out_stride + o + 16 * r,
                             _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[i][r]), m, b));
    }
}

// `c` of the padded input `in`, written to `out`, (64, out_stride), its first `c.cout` columns
inline void convolve(const Conv& c, const uint8_t* in, float* out, int out_stride) {
    const int c4n = c.cin / 4;
    const int pairs = c.cout / 32;
    const size_t taps = (size_t)c.k * c.k;
    for (int g = 0; g < pairs; g++) {
        const int8_t* w = c.packed.data() + (size_t)g * taps * c4n * 128;
        for (int y = 0; y < 8; y++) {
            if (c.k == 3) conv_row<3, 2>(c, w, in, y, 32 * g, out, out_stride);
            else conv_row<1, 2>(c, w, in, y, 32 * g, out, out_stride);
        }
    }
    if (c.cout % 32) {
        const int8_t* w = c.packed.data() + (size_t)pairs * taps * c4n * 128;
        for (int y = 0; y < 8; y++) {
            if (c.k == 3) conv_row<3, 1>(c, w, in, y, 32 * pairs, out, out_stride);
            else conv_row<1, 1>(c, w, in, y, 32 * pairs, out, out_stride);
        }
    }
}

// the first `channels` of `x`, (64, stride), quantized into `padded` for `c`, its border the zero point
inline void quantize(const Conv& c, const float* x, int stride, uint8_t* padded) {
    const int channels = c.cin;
    std::memset(padded, c.zero_point, (size_t)PADDED * channels);
    __m512 inv = _mm512_set1_ps(c.inv_in_scale);
    __m512i zp = _mm512_set1_epi32(c.zero_point);
    __m512i lo = _mm512_setzero_si512(), hi = _mm512_set1_epi32(255);
    for (int s = 0; s < SQUARES; s++) {
        const float* v = x + s * stride;
        uint8_t* q = padded + ((s / 8 + 1) * 10 + s % 8 + 1) * channels;
        for (int ch = 0; ch < channels; ch += 16) {
            __m512i i = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(v + ch), inv)), zp);
            i = _mm512_min_epi32(_mm512_max_epi32(i, lo), hi);
            _mm_storeu_si128((__m128i*)(q + ch), _mm512_cvtepi32_epi8(i));
        }
    }
}

// `n` of `ch` values (a multiple of 16) at `v`, in place, then a ReLU if `relu`
inline void normalize_square(const Norm& n, float* v, int ch, bool relu) {
    const __m512 zero = _mm512_setzero_ps();
    if (n.type == NORM_CHANNEL) {
        __m512 sum = zero;
        for (int i = 0; i < ch; i += 16) sum = _mm512_add_ps(sum, _mm512_loadu_ps(v + i));
        __m512 mean = _mm512_set1_ps(_mm512_reduce_add_ps(sum) / ch);
        __m512 sq = zero;
        for (int i = 0; i < ch; i += 16) {
            __m512 d = _mm512_sub_ps(_mm512_loadu_ps(v + i), mean);
            sq = _mm512_fmadd_ps(d, d, sq);
        }
        __m512 rstd = _mm512_set1_ps(1.0f / std::sqrt(_mm512_reduce_add_ps(sq) / ch + n.eps));
        for (int i = 0; i < ch; i += 16) {
            __m512 d = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v + i), mean), rstd);
            __m512 out = _mm512_fmadd_ps(d, _mm512_loadu_ps(n.a.data() + i), _mm512_loadu_ps(n.c.data() + i));
            _mm512_storeu_ps(v + i, relu ? _mm512_max_ps(out, zero) : out);
        }
    } else if (n.type == NORM_AFFINE) {
        for (int i = 0; i < ch; i += 16) {
            __m512 out = _mm512_fmadd_ps(_mm512_loadu_ps(v + i), _mm512_loadu_ps(n.a.data() + i),
                                         _mm512_loadu_ps(n.c.data() + i));
            _mm512_storeu_ps(v + i, relu ? _mm512_max_ps(out, zero) : out);
        }
    } else if (relu) {
        for (int i = 0; i < ch; i += 16) _mm512_storeu_ps(v + i, _mm512_max_ps(_mm512_loadu_ps(v + i), zero));
    }
}

// `n` of each square's first `n.channels` of `x`, (64, stride), in place, then a ReLU if `relu`
inline void normalize(const Norm& n, float* x, int stride, bool relu) {
    if (n.type == NORM_NONE && !relu) return;
    for (int s = 0; s < SQUARES; s++) normalize_square(n, x + s * stride, n.channels ? n.channels : stride, relu);
}

// each of `count` channels from `first`'s mean and maximum over the board, (2 * count); count a
// multiple of 16
inline void board_pool(const float* x, int stride, int first, int count, float* out) {
    for (int i = 0; i < count; i += 16) {
        __m512 sum = _mm512_setzero_ps(), most = _mm512_set1_ps(-INFINITY);
        for (int s = 0; s < SQUARES; s++) {
            __m512 v = _mm512_loadu_ps(x + s * stride + first + i);
            sum = _mm512_add_ps(sum, v);
            most = _mm512_max_ps(most, v);
        }
        _mm512_storeu_ps(out + i, _mm512_mul_ps(sum, _mm512_set1_ps(1.0f / SQUARES)));
        _mm512_storeu_ps(out + count + i, most);
    }
}

inline void linear(const Linear& l, const float* in, float* out) {
    for (int o = 0; o < l.out; o++) {
        const float* w = l.w.data() + (size_t)o * l.in;
        __m512 sum = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= l.in; i += 16) sum = _mm512_fmadd_ps(_mm512_loadu_ps(w + i), _mm512_loadu_ps(in + i), sum);
        float total = l.b[o] + _mm512_reduce_add_ps(sum);
        for (; i < l.in; i++) total += w[i] * in[i];
        out[o] = total;
    }
}

inline void relu(float* x, int count) {
    for (int i = 0; i < count; i++) x[i] = std::max(x[i], 0.0f);
}

// `b`, (cols), added to each of the 64 squares' first `cols` of `x`, (64, stride), then a ReLU if
// `relu_after`; cols a multiple of 16
inline void add_rows(float* x, int stride, const float* b, int cols, bool relu_after) {
    const __m512 zero = _mm512_setzero_ps();
    for (int s = 0; s < SQUARES; s++)
        for (int i = 0; i < cols; i += 16) {
            __m512 v = _mm512_add_ps(_mm512_loadu_ps(x + s * stride + i), _mm512_loadu_ps(b + i));
            _mm512_storeu_ps(x + s * stride + i, relu_after ? _mm512_max_ps(v, zero) : v);
        }
}

// (64, cin) by (cin, cols) plus (cols), into (64, cols), float; cols a multiple of 16, up to 128
template <int BLOCKS>
inline void gemm_blocks(const float* x, int cin, const float* w, int cols, const float* b, float* out, int o) {
    for (int s0 = 0; s0 < SQUARES; s0 += 4) {
        __m512 acc[4][BLOCKS];
        for (int r = 0; r < BLOCKS; r++) {
            __m512 bias = _mm512_loadu_ps(b + o + 16 * r);
            for (int i = 0; i < 4; i++) acc[i][r] = bias;
        }
        for (int c = 0; c < cin; c++) {
            __m512 v[4];
            for (int i = 0; i < 4; i++) v[i] = _mm512_set1_ps(x[(s0 + i) * cin + c]);
            for (int r = 0; r < BLOCKS; r++) {
                __m512 wv = _mm512_loadu_ps(w + (size_t)c * cols + o + 16 * r);
                for (int i = 0; i < 4; i++) acc[i][r] = _mm512_fmadd_ps(v[i], wv, acc[i][r]);
            }
        }
        for (int i = 0; i < 4; i++)
            for (int r = 0; r < BLOCKS; r++) _mm512_storeu_ps(out + (s0 + i) * cols + o + 16 * r, acc[i][r]);
    }
}

inline void gemm(const float* x, int cin, const float* w, int cols, const float* b, float* out) {
    int o = 0;
    for (; o + 96 <= cols; o += 96) gemm_blocks<6>(x, cin, w, cols, b, out, o);
    for (; o + 32 <= cols; o += 32) gemm_blocks<2>(x, cin, w, cols, b, out, o);
    for (; o < cols; o += 16) gemm_blocks<1>(x, cin, w, cols, b, out, o);
}

// One position: `obs`, (2, 8, 8) channels-first, the mover's discs then the opponent's, to (65)
// policy logits and a value.
inline void forward(const Net& n, Scratch& s, const float* obs, float* logits, float* value) {
    const int C = n.channels;
    float* x = s.x.data();
    float* y = s.y.data();
    // the stem's input, its 2 planes padded to 4 channels
    std::memset(s.padded.data(), 0, (size_t)PADDED * 4);
    for (int sq = 0; sq < SQUARES; sq++)
        for (int p = 0; p < 2; p++)
            s.padded[((sq / 8 + 1) * 10 + sq % 8 + 1) * 4 + p] = obs[p * SQUARES + sq] > 0.5f ? 1 : 0;
    convolve(n.stem, s.padded.data(), x, C);
    if (!n.pre) normalize(n.stem_norm, x, C, true);
    for (const Block& b : n.blocks) {
        const int kept = C - b.pool_channels;
        if (n.pre) {
            // y = relu(norm1(x)), quantized for conv1
            std::memcpy(y, x, sizeof(float) * SQUARES * C);
            normalize(b.norm1, y, C, true);
            quantize(b.conv1, y, C, s.padded.data());
            convolve(b.conv1, s.padded.data(), y, C);
            normalize(b.norm2, y, C, true);
        } else {
            quantize(b.conv1, x, C, s.padded.data());
            convolve(b.conv1, s.padded.data(), y, C);
            normalize(b.norm1, y, C, false);
        }
        if (b.pool_channels) {
            float pooled[2 * 512], bias[1024];
            if (n.pre) {
                board_pool(y, C, kept, b.pool_channels, pooled);
            } else {
                // of the pooled channels after a ReLU
                for (int sq = 0; sq < SQUARES; sq++) normalize_square(Norm(), y + sq * C + kept, b.pool_channels, true);
                board_pool(y, C, kept, b.pool_channels, pooled);
            }
            linear(b.pool, pooled, bias);
            add_rows(y, C, bias, kept, !n.pre);
        } else if (!n.pre) {
            for (int sq = 0; sq < SQUARES; sq++) normalize_square(Norm(), y + sq * C, kept, true);
        }
        quantize(b.conv2, y, C, s.padded.data());
        convolve(b.conv2, s.padded.data(), y, C);
        if (n.pre) {
            for (int i = 0; i < SQUARES * C; i += 16)
                _mm512_storeu_ps(x + i, _mm512_add_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
        } else {
            normalize(b.norm2, y, C, false);
            for (int i = 0; i < SQUARES * C; i += 16)
                _mm512_storeu_ps(x + i, _mm512_max_ps(_mm512_add_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)),
                                                      _mm512_setzero_ps()));
        }
    }
    if (n.pre) normalize(n.final_norm, x, C, true);

    // the heads, in float: the three 1x1 convolutions at once
    const int H = n.head_channels, H3 = 3 * H;
    float* h = s.h.data();
    gemm(x, C, n.head_w.data(), H3, n.head_b.data(), h);
    // each part's normalization, of its own channels
    auto part = [&](const Norm& norm, int first, bool relu_after) {
        for (int sq = 0; sq < SQUARES; sq++) normalize_square(norm, h + sq * H3 + first, H, relu_after);
    };
    part(n.policy_norm, 0, false);
    part(n.pool_norm, H, true);
    part(n.value_norm, 2 * H, true);
    float pooled[256], bias[128], hidden[1024], out[1];
    board_pool(h, H3, H, H, pooled);
    linear(n.pool_linear, pooled, bias);
    for (int sq = 0; sq < SQUARES; sq++) {
        float* v = h + sq * H3;
        float logit = 0;
        for (int i = 0; i < H; i++) logit += std::max(v[i] + bias[i], 0.0f) * n.square_w[i];
        logits[sq] = logit;
    }
    linear(n.pass_linear, pooled, out);
    logits[SQUARES] = out[0];
    board_pool(h, H3, 2 * H, H, pooled);
    linear(n.value_hidden, pooled, hidden);
    relu(hidden, n.value_hidden.out);
    linear(n.value_out, hidden, out);
    *value = std::tanh(out[0]);
}

#else
inline void forward(const Net&, Scratch&, const float*, float*, float*) {
    throw std::runtime_error("the native network needs AVX-512 VNNI");
}
#endif

// Packs `q`, (cout, cin, k, k) signed 8-bit weights, for `convolve`.
inline void pack(Conv& c, const int8_t* q) {
    const int c4n = c.cin / 4, taps = c.k * c.k;
    c.packed.assign((size_t)c.cout * c.cin * taps, 0);
    size_t at = 0;
    for (int o0 = 0; o0 < c.cout;) {
        int ro = c.cout - o0 >= 32 ? 2 : 1;
        for (int t = 0; t < taps; t++)
            for (int c4 = 0; c4 < c4n; c4++)
                for (int r = 0; r < ro; r++)
                    for (int o = 0; o < 16; o++)
                        for (int k = 0; k < 4; k++) {
                            int out = o0 + 16 * r + o, in = 4 * c4 + k;
                            c.packed[at++] = q[(((size_t)out * c.cin + in) * taps) + t];
                        }
        o0 += 16 * ro;
    }
}

}  // namespace net
