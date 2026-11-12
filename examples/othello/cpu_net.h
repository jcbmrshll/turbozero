// An Othello ResNet's forward pass on CPU, int8 in the convolutions and float everywhere else, one position at a time: what `cpu_net.py` builds from a trained network, for
// `cpu_search.cpp`'s engine. It covers basic blocks in either layout, with BatchNorm folded into the
// convolutions or normalizations between them, global pooling biases, and the pooled heads (see
// `core.networks.azresnet`).
//
// Each convolution takes 8-bit unsigned inputs (a scale and a zero point for the whole tensor, from
// calibration) and 8-bit signed weights (a scale for each output channel), sums in 32-bit integers, and
// writes float outputs, acc * scale + bias, which per-square normalizations, the residual sum and the
// global pooling then take in float. A normalization of each square's channels (ConvNeXt's LayerNorm)
// costs about what a ReLU does here, where OpenVINO runs it as separate ops.
//
// The convolutions come in three kernels, for the levels engines ship builds for: AVX-512 with VNNI
// (x86-64-v4), 4 products a lane an instruction; AVX2 (x86-64-v3), Winograd's F(2x2, 3x3) in 16-bit
// integers; and ARM's NEON with the dot product (ARMv8.2, Apple's M1 on), 4 products a lane. All three
// sum exactly the same integers. Elsewhere the library builds without the network.
#pragma once

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#define NET_ISA 2
#elif defined(__AVX2__) && defined(__FMA__)
#define NET_ISA 1
#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#define NET_ISA 3
#else
#define NET_ISA 0
#endif
#define NET_NATIVE (NET_ISA > 0)

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

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

// a convolution's quantized input: 8 bits for VNNI, widened to 16 for AVX2, and for NEON, whose dot
// product takes signed bytes, less 128, which `Conv::offset` adds back
#if NET_ISA == 2
using Activation = uint8_t;
#elif NET_ISA == 3
using Activation = int8_t;
#else
using Activation = int16_t;
#endif
constexpr int ACTIVATION_SHIFT = NET_ISA == 3 ? 128 : 0;

struct Conv {
    int cin = 0, cout = 0, k = 3;
    // the weights for the kernel (see `pack`)
    std::vector<int8_t> packed;
    std::vector<int16_t> packed16;
    std::vector<float> scale, bias;  // of each output: acc * scale + bias
    std::vector<int32_t> offset;  // NEON: 128 times each output's weights' sum
    std::vector<float> inverse;  // of each input channel: its quantization is round(x * inverse) + zero_point
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
    // the heads' three 1x1 convolutions in one, to 3 * head_channels: policy, its pooled part, value
    Conv head;
    Norm policy_norm, pool_norm, value_norm;
    Linear pool_linear, pass_linear, value_hidden, value_out;
    std::vector<float> square_w;  // (head_channels)
};

// what a forward pass writes as it goes, one to a thread
struct Scratch {
    std::vector<Activation> padded;  // (100, max channels), a convolution's input
    std::vector<float> x, y, h;   // (64, max channels): the trunk, a block's middle, the heads
    std::vector<int16_t> winograd;  // (16, 16 tiles, max channels): AVX2's transformed input
    explicit Scratch(const Net& n) {
        int c = std::max(n.channels, 4);
        // and room for NEON's loads of 16 channels past the last square's
        padded.assign((size_t)PADDED * c + 16, 0);
        winograd.assign((size_t)16 * 16 * c, 0);
        x.assign((size_t)SQUARES * c, 0);
        y.assign((size_t)SQUARES * c, 0);
        h.assign((size_t)SQUARES * std::max(c, 3 * n.head_channels), 0);
    }
};

#if NET_NATIVE
inline int32_t load_u32(const void* p) {
    int32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

#if NET_ISA == 2
// One output row of a convolution, `RO` blocks of 16 outputs from `o`, for the 8 squares of row `y`.
// The tile is the board's row by two blocks: 16 sums, 8 broadcast inputs and 2 weight vectors, 26 of
// AVX-512's 32 registers, each weight used 8 times. It follows from the instruction set, not from
// any one CPU's caches or timings.
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
inline void convolve(const Conv& c, Scratch& s, float* out, int out_stride) {
    const Activation* in = s.padded.data();
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
#elif NET_ISA == 1
// AVX2: 3x3 convolutions by Winograd's F(2x2, 3x3) in integers, the 1x1 directly. Each 2x2 tile of
// outputs comes from its 4x4 patch of the padded input: the patch is transformed (B^T d B: sums and
// differences, values within +-1020), multiplied value by value with the weights transformed (G g G^T,
// G doubled to whole numbers: within +-1143), summed over the input channels, and transformed back (A^T
// m A), which gives 4 times VNNI's sums exactly, in 16 products where the direct convolution takes 36.
// The products are 16-bit pairs summed into 32 bits (vpmaddwd, or vpdpwssd with AVX-VNNI): exact,
// and wrapping 32-bit sums cancel out as the outputs fit in 32 bits.
inline __m256i multiply_add(__m256i acc, __m256i a, __m256i b) {
#ifdef __AVXVNNI__
    return _mm256_dpwssd_avx_epi32(acc, a, b);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
#endif
}

constexpr int TILES = 16;  // the board's 2x2 tiles of outputs, 4 by 4

// each tile's patch of `in`, the padded input, (100, cin), transformed, to `v`, (16 values, 16 tiles,
// cin)
inline void winograd_input(const int16_t* in, int cin, int16_t* v) {
    const size_t plane = (size_t)TILES * cin;
    for (int tile = 0; tile < TILES; tile++) {
        const int16_t* d = in + ((tile / 4) * 20 + (tile % 4) * 2) * cin;
        int16_t* out = v + (size_t)tile * cin;
#pragma omp simd
        for (int c = 0; c < cin; c++) {
            int16_t t[4][4];
            for (int j = 0; j < 4; j++) {
                const int16_t* col = d + j * cin + c;
                int16_t d0 = col[0], d1 = col[10 * cin], d2 = col[20 * cin], d3 = col[30 * cin];
                t[0][j] = d0 - d2;
                t[1][j] = d1 + d2;
                t[2][j] = d2 - d1;
                t[3][j] = d1 - d3;
            }
            for (int i = 0; i < 4; i++) {
                int16_t* row = out + (size_t)(4 * i) * plane + c;
                row[0] = t[i][0] - t[i][2];
                row[plane] = t[i][1] + t[i][2];
                row[2 * plane] = t[i][2] - t[i][1];
                row[3 * plane] = t[i][1] - t[i][3];
            }
        }
    }
}

// one transformed value's sums for 16 outputs: `v`, (16 tiles, cin), by `w`, (cin / 2, 2, 8 outputs,
// 2 inputs), to `m`, (16 tiles, 16 outputs). Tiles 4 at a time by the 16 outputs: 8 sums, 4 broadcast
// input pairs and 2 weight vectors, 14 of AVX2's 16 registers.
inline void winograd_multiply(const int16_t* v, const int16_t* w, int cin, int32_t* m) {
    const int c2n = cin / 2;
    for (int q = 0; q < TILES; q += 4) {
        __m256i acc[4][2];
        for (int i = 0; i < 4; i++) acc[i][0] = acc[i][1] = _mm256_setzero_si256();
        const int16_t* vq = v + (size_t)q * cin;
#pragma GCC unroll 2
        for (int c2 = 0; c2 < c2n; c2++) {
            __m256i a[4];
            for (int i = 0; i < 4; i++) a[i] = _mm256_set1_epi32(load_u32(vq + i * cin + 2 * c2));
            for (int r = 0; r < 2; r++) {
                __m256i wv = _mm256_loadu_si256((const __m256i*)(w + c2 * 32 + 16 * r));
                for (int i = 0; i < 4; i++) acc[i][r] = multiply_add(acc[i][r], a[i], wv);
            }
        }
        for (int i = 0; i < 4; i++)
            for (int r = 0; r < 2; r++) _mm256_storeu_si256((__m256i*)(m + (q + i) * 16 + 8 * r), acc[i][r]);
    }
}

// `m`, (16 values, 16 tiles, 16 outputs), transformed back, to outputs `o` to `o + 16` of `out`
inline void winograd_output(const Conv& c, const int32_t* m, int o, float* out, int out_stride) {
    const float* scale = c.scale.data() + o;
    const float* bias = c.bias.data() + o;
    for (int tile = 0; tile < TILES; tile++) {
        float* y = out + ((tile / 4) * 16 + (tile % 4) * 2) * out_stride + o;
#pragma omp simd
        for (int k = 0; k < 16; k++) {
            // unsigned: the sums wrap as the instructions' do
            uint32_t t[2][4];
            for (int j = 0; j < 4; j++) {
                uint32_t m0 = m[((0 + j) * TILES + tile) * 16 + k], m1 = m[((4 + j) * TILES + tile) * 16 + k];
                uint32_t m2 = m[((8 + j) * TILES + tile) * 16 + k], m3 = m[((12 + j) * TILES + tile) * 16 + k];
                t[0][j] = m0 + m1 + m2;
                t[1][j] = m1 - m2 - m3;
            }
            for (int i = 0; i < 2; i++) {
                y[(8 * i) * out_stride + k] =
                    std::fma((float)(int32_t)(t[i][0] + t[i][1] + t[i][2]), 0.25f * scale[k], bias[k]);
                y[(8 * i + 1) * out_stride + k] =
                    std::fma((float)(int32_t)(t[i][1] - t[i][2] - t[i][3]), 0.25f * scale[k], bias[k]);
            }
        }
    }
}

// a 1x1 convolution, half a row of 4 squares by 16 outputs as in `winograd_multiply`
inline void conv_quarter(const Conv& c, const int16_t* w, const int16_t* in, int y, int x0, int o, float* out,
                         int out_stride) {
    const int c2n = c.cin / 2;
    __m256i acc[4][2];
    for (int i = 0; i < 4; i++) acc[i][0] = acc[i][1] = _mm256_setzero_si256();
    const int16_t* row = in + ((y + 1) * 10 + x0 + 1) * c.cin;
#pragma GCC unroll 2
    for (int c2 = 0; c2 < c2n; c2++) {
        __m256i a[4];
        for (int i = 0; i < 4; i++) a[i] = _mm256_set1_epi32(load_u32(row + i * c.cin + 2 * c2));
        for (int r = 0; r < 2; r++) {
            __m256i wv = _mm256_loadu_si256((const __m256i*)(w + c2 * 32 + 16 * r));
            for (int i = 0; i < 4; i++) acc[i][r] = multiply_add(acc[i][r], a[i], wv);
        }
    }
    for (int r = 0; r < 2; r++) {
        __m256 m = _mm256_loadu_ps(c.scale.data() + o + 8 * r);
        __m256 b = _mm256_loadu_ps(c.bias.data() + o + 8 * r);
        for (int i = 0; i < 4; i++)
            _mm256_storeu_ps(out + (y * 8 + x0 + i) * out_stride + o + 8 * r,
                             _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[i][r]), m, b));
    }
}

inline void convolve(const Conv& c, Scratch& s, float* out, int out_stride) {
    const Activation* in = s.padded.data();
    if (c.k == 1) {
        const size_t group = (size_t)(c.cin / 2) * 32;
        for (int g = 0; g < c.cout / 16; g++)
            for (int y = 0; y < 8; y++)
                for (int x0 = 0; x0 < 8; x0 += 4) conv_quarter(c, c.packed16.data() + g * group, in, y, x0, 16 * g, out, out_stride);
        return;
    }
    int16_t* v = s.winograd.data();
    winograd_input(in, c.cin, v);
    alignas(32) int32_t m[16 * TILES * 16];
    const size_t chunk = (size_t)(c.cin / 2) * 32;
    for (int g = 0; g < c.cout / 16; g++) {
        const int16_t* w = c.packed16.data() + (size_t)g * 16 * chunk;
        for (int p = 0; p < 16; p++)
            winograd_multiply(v + (size_t)p * TILES * c.cin, w + p * chunk, c.cin, m + p * TILES * 16);
        winograd_output(c, m, 16 * g, out, out_stride);
    }
}
#else
// NEON with the dot product: the board's row of 8 squares by 8 outputs, 16 sums, 8 squares' inputs of 16
// channels and 2 weight vectors, 26 of its 32 registers, each weight used 8 times. An sdot multiplies 4
// outputs' weights for 4 channels by one square's 4 channels, picked from its vector by lane.
template <int K>
inline void conv_row(const Conv& c, const int8_t* w, const int8_t* in, int y, int o, float* out, int out_stride) {
    const int c4n = c.cin / 4;
    int32x4_t acc[8][2];
    for (int i = 0; i < 8; i++) acc[i][0] = acc[i][1] = vdupq_n_s32(0);
    for (int t = 0; t < K * K; t++) {
        int dy = K == 3 ? t / 3 : 1, dx = K == 3 ? t % 3 : 1;
        const int8_t* row = in + ((y + dy) * 10 + dx) * c.cin;
        const int8_t* wt = w + (size_t)t * c4n * 32;
        for (int c4 = 0; c4 < c4n; c4 += 4) {
            int8x16_t a[8];
            for (int i = 0; i < 8; i++) a[i] = vld1q_s8(row + i * c.cin + 4 * c4);
            // the lane must be a constant; channels come in fours, so the last group may be short
#define NET_LANE(L)                                                                  \
            if (c4 + L < c4n) {                                                      \
                int8x16_t w0 = vld1q_s8(wt + (c4 + L) * 32), w1 = vld1q_s8(wt + (c4 + L) * 32 + 16); \
                for (int i = 0; i < 8; i++) {                                        \
                    acc[i][0] = vdotq_laneq_s32(acc[i][0], w0, a[i], L);             \
                    acc[i][1] = vdotq_laneq_s32(acc[i][1], w1, a[i], L);             \
                }                                                                    \
            }
            NET_LANE(0) NET_LANE(1) NET_LANE(2) NET_LANE(3)
#undef NET_LANE
        }
    }
    for (int r = 0; r < 2; r++) {
        int32x4_t offset = vld1q_s32(c.offset.data() + o + 4 * r);
        float32x4_t m = vld1q_f32(c.scale.data() + o + 4 * r), b = vld1q_f32(c.bias.data() + o + 4 * r);
        for (int i = 0; i < 8; i++)
            vst1q_f32(out + (y * 8 + i) * out_stride + o + 4 * r,
                      vfmaq_f32(b, vcvtq_f32_s32(vaddq_s32(acc[i][r], offset)), m));
    }
}

inline void convolve(const Conv& c, Scratch& s, float* out, int out_stride) {
    const Activation* in = s.padded.data();
    const size_t group = (size_t)c.k * c.k * (c.cin / 4) * 32;
    for (int g = 0; g < c.cout / 8; g++)
        for (int y = 0; y < 8; y++) {
            if (c.k == 3) conv_row<3>(c, c.packed.data() + g * group, in, y, 8 * g, out, out_stride);
            else conv_row<1>(c, c.packed.data() + g * group, in, y, 8 * g, out, out_stride);
        }
}
#endif

#if NET_ISA == 2
// AVX-512's own, where the compiler's loops came out slower
// the first `n` lanes (all 16 when n >= 16), the rest zero
inline __mmask16 lanes(int n) { return n >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << n) - 1); }
inline __m512 load(const float* p, int n) { return _mm512_maskz_loadu_ps(lanes(n), p); }
inline void store(float* p, __m512 v, int n) { _mm512_mask_storeu_ps(p, lanes(n), v); }

// the first `channels` of `x`, (64, stride), quantized into `padded` for `c`, its border the zero point
inline void quantize(const Conv& c, const float* x, int stride, Activation* padded) {
    const int channels = c.cin;
    std::memset(padded, c.zero_point, (size_t)PADDED * channels);
    __m512i zp = _mm512_set1_epi32(c.zero_point);
    __m512i lo = _mm512_setzero_si512(), hi = _mm512_set1_epi32(255);
    for (int s = 0; s < SQUARES; s++) {
        const float* v = x + s * stride;
        uint8_t* q = padded + ((s / 8 + 1) * 10 + s % 8 + 1) * channels;
        for (int ch = 0; ch < channels; ch += 16) {
            int n = channels - ch;
            __m512 inv = load(c.inverse.data() + ch, n);
            __m512i i = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(load(v + ch, n), inv)), zp);
            i = _mm512_min_epi32(_mm512_max_epi32(i, lo), hi);
            _mm_mask_storeu_epi8(q + ch, lanes(n), _mm512_cvtepi32_epi8(i));
        }
    }
}

// `n` of `ch` values at `v`, in place, then a ReLU if `relu`
inline void normalize_square(const Norm& n, float* v, int ch, bool relu) {
    const __m512 zero = _mm512_setzero_ps();
    if (n.type == NORM_CHANNEL) {
        __m512 sum = zero;
        for (int i = 0; i < ch; i += 16) sum = _mm512_add_ps(sum, load(v + i, ch - i));
        __m512 mean = _mm512_set1_ps(_mm512_reduce_add_ps(sum) / ch);
        __m512 sq = zero;
        for (int i = 0; i < ch; i += 16) {
            __m512 d = _mm512_maskz_sub_ps(lanes(ch - i), load(v + i, ch - i), mean);
            sq = _mm512_fmadd_ps(d, d, sq);
        }
        __m512 rstd = _mm512_set1_ps(1.0f / std::sqrt(_mm512_reduce_add_ps(sq) / ch + n.eps));
        for (int i = 0; i < ch; i += 16) {
            int m = ch - i;
            __m512 d = _mm512_mul_ps(_mm512_sub_ps(load(v + i, m), mean), rstd);
            __m512 out = _mm512_fmadd_ps(d, load(n.a.data() + i, m), load(n.c.data() + i, m));
            store(v + i, relu ? _mm512_max_ps(out, zero) : out, m);
        }
    } else if (n.type == NORM_AFFINE) {
        for (int i = 0; i < ch; i += 16) {
            int m = ch - i;
            __m512 out = _mm512_fmadd_ps(load(v + i, m), load(n.a.data() + i, m), load(n.c.data() + i, m));
            store(v + i, relu ? _mm512_max_ps(out, zero) : out, m);
        }
    } else if (relu) {
        for (int i = 0; i < ch; i += 16) store(v + i, _mm512_max_ps(load(v + i, ch - i), zero), ch - i);
    }
}

// `n` of each square's first `n.channels` of `x`, (64, stride), in place, then a ReLU if `relu`
inline void normalize(const Norm& n, float* x, int stride, bool relu) {
    if (n.type == NORM_NONE && !relu) return;
    for (int s = 0; s < SQUARES; s++) normalize_square(n, x + s * stride, n.channels ? n.channels : stride, relu);
}

// each of `count` channels from `first`'s mean and maximum over the board, (2 * count)
inline void board_pool(const float* x, int stride, int first, int count, float* out) {
    for (int i = 0; i < count; i += 16) {
        int m = count - i;
        __m512 sum = _mm512_setzero_ps(), most = _mm512_set1_ps(-INFINITY);
        for (int s = 0; s < SQUARES; s++) {
            __m512 v = load(x + s * stride + first + i, m);
            sum = _mm512_add_ps(sum, v);
            most = _mm512_max_ps(most, v);
        }
        store(out + i, _mm512_mul_ps(sum, _mm512_set1_ps(1.0f / SQUARES)), m);
        store(out + count + i, most, m);
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
// `relu_after`
inline void add_rows(float* x, int stride, const float* b, int cols, bool relu_after) {
    const __m512 zero = _mm512_setzero_ps();
    for (int s = 0; s < SQUARES; s++)
        for (int i = 0; i < cols; i += 16) {
            int m = cols - i;
            __m512 v = _mm512_add_ps(load(x + s * stride + i, m), load(b + i, m));
            store(x + s * stride + i, relu_after ? _mm512_max_ps(v, zero) : v, m);
        }
}

// x += y, then a ReLU if `relu_after`, over `count` values, a multiple of 16
inline void residual(float* x, const float* y, int count, bool relu_after) {
    const __m512 zero = _mm512_setzero_ps();
    for (int i = 0; i < count; i += 16) {
        __m512 v = _mm512_add_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i));
        _mm512_storeu_ps(x + i, relu_after ? _mm512_max_ps(v, zero) : v);
    }
}

#else
#if NET_ISA == 1
// AVX2: the quantization and normalizations by hand, 8 lanes, as AVX-512's above; the rest, and NEON's,
// plain loops, which the compiler vectorizes
// the first `n` of 8 lanes (all when n >= 8)
inline __m256i lanes8(int n) {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(n), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}
inline __m256 load8(const float* p, int n) { return n >= 8 ? _mm256_loadu_ps(p) : _mm256_maskload_ps(p, lanes8(n)); }
inline void store8(float* p, __m256 v, int n) {
    if (n >= 8) _mm256_storeu_ps(p, v);
    else _mm256_maskstore_ps(p, lanes8(n), v);
}
inline float sum8(__m256 v) {
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    return _mm_cvtss_f32(_mm_add_ss(x, _mm_movehdup_ps(x)));
}

// the first `c.cin` channels of `x`, (64, stride), quantized into `padded` for `c`, its border the zero
// point
inline void quantize(const Conv& c, const float* x, int stride, Activation* padded) {
    const int channels = c.cin;
    std::fill(padded, padded + (size_t)PADDED * channels, (Activation)c.zero_point);
    const __m256 low = _mm256_set1_ps(-c.zero_point - 1.0f), high = _mm256_set1_ps(256.0f - c.zero_point);
    const __m256i zp = _mm256_set1_epi32(c.zero_point), lo = _mm256_setzero_si256(), hi = _mm256_set1_epi32(255);
    for (int s = 0; s < SQUARES; s++) {
        const float* v = x + s * stride;
        Activation* q = padded + ((s / 8 + 1) * 10 + s % 8 + 1) * channels;
        // channels come in fours: 8 at a time, and a last 4
        for (int ch = 0; ch < channels; ch += 8) {
            int n = channels - ch;
            __m256 t = _mm256_mul_ps(load8(v + ch, n), load8(c.inverse.data() + ch, n));
            t = _mm256_min_ps(_mm256_max_ps(t, low), high);
            // rounded to the nearest, ties to even, as `rint`
            __m256i i = _mm256_add_epi32(_mm256_cvtps_epi32(t), zp);
            i = _mm256_min_epi32(_mm256_max_epi32(i, lo), hi);
            __m128i packed = _mm_packs_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256(i, 1));
            if (n >= 8) _mm_storeu_si128((__m128i*)(q + ch), packed);
            else _mm_storel_epi64((__m128i*)(q + ch), packed);
        }
    }
}

// `n` of `ch` values at `v`, in place, then a ReLU if `relu`
inline void normalize_square(const Norm& n, float* v, int ch, bool relu) {
    const __m256 zero = _mm256_setzero_ps();
    if (n.type == NORM_CHANNEL) {
        __m256 sum = zero;
        for (int i = 0; i < ch; i += 8) sum = _mm256_add_ps(sum, load8(v + i, ch - i));
        __m256 mean = _mm256_set1_ps(sum8(sum) / ch);
        __m256 sq = zero;
        for (int i = 0; i < ch; i += 8) {
            __m256 d = _mm256_and_ps(_mm256_sub_ps(load8(v + i, ch - i), mean), _mm256_castsi256_ps(lanes8(ch - i)));
            sq = _mm256_fmadd_ps(d, d, sq);
        }
        __m256 rstd = _mm256_set1_ps(1.0f / std::sqrt(sum8(sq) / ch + n.eps));
        for (int i = 0; i < ch; i += 8) {
            int m = ch - i;
            __m256 d = _mm256_mul_ps(_mm256_sub_ps(load8(v + i, m), mean), rstd);
            __m256 out = _mm256_fmadd_ps(d, load8(n.a.data() + i, m), load8(n.c.data() + i, m));
            store8(v + i, relu ? _mm256_max_ps(out, zero) : out, m);
        }
    } else if (n.type == NORM_AFFINE) {
        for (int i = 0; i < ch; i += 8) {
            int m = ch - i;
            __m256 out = _mm256_fmadd_ps(load8(v + i, m), load8(n.a.data() + i, m), load8(n.c.data() + i, m));
            store8(v + i, relu ? _mm256_max_ps(out, zero) : out, m);
        }
    } else if (relu) {
        for (int i = 0; i < ch; i += 8) store8(v + i, _mm256_max_ps(load8(v + i, ch - i), zero), ch - i);
    }
}

#else
// NEON: the quantization and normalizations by hand, 4 lanes (GCC leaves the plain loops scalar)
// the first `channels` (a multiple of 4) of `x`, (64, stride), quantized into `padded` for `c`, less 128
// for the signed dot product, its border the zero point
inline void quantize(const Conv& c, const float* x, int stride, Activation* padded) {
    const int channels = c.cin;
    std::fill(padded, padded + (size_t)PADDED * channels, (Activation)(c.zero_point - ACTIVATION_SHIFT));
    const float32x4_t low = vdupq_n_f32(-c.zero_point - 1.0f), high = vdupq_n_f32(256.0f - c.zero_point);
    const int32x4_t zp = vdupq_n_s32(c.zero_point - ACTIVATION_SHIFT);
    const int32x4_t lo = vdupq_n_s32(-ACTIVATION_SHIFT), hi = vdupq_n_s32(255 - ACTIVATION_SHIFT);
    auto step = [&](const float* v, const float* inv) {
        float32x4_t t = vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(v), vld1q_f32(inv)), low), high);
        // rounded to the nearest, ties to even, as `rint`
        int32x4_t i = vaddq_s32(vcvtnq_s32_f32(t), zp);
        return vmovn_s32(vminq_s32(vmaxq_s32(i, lo), hi));
    };
    for (int s = 0; s < SQUARES; s++) {
        const float* v = x + s * stride;
        Activation* q = padded + ((s / 8 + 1) * 10 + s % 8 + 1) * channels;
        int ch = 0;
        for (; ch + 8 <= channels; ch += 8) {
            int16x8_t h = vcombine_s16(step(v + ch, c.inverse.data() + ch), step(v + ch + 4, c.inverse.data() + ch + 4));
            vst1_s8(q + ch, vmovn_s16(h));
        }
        if (ch < channels) {
            int8x8_t b = vmovn_s16(vcombine_s16(step(v + ch, c.inverse.data() + ch), vdup_n_s16(0)));
            vst1_lane_s32((int32_t*)(q + ch), vreinterpret_s32_s8(b), 0);
        }
    }
}

// `n` of `ch` values at `v`, in place, then a ReLU if `relu`
inline void normalize_square(const Norm& n, float* v, int ch, bool relu) {
    const float32x4_t zero = vdupq_n_f32(0);
    const float* a = n.a.data();
    const float* c = n.c.data();
    const int whole = ch & ~3;
    float mean = 0, rstd = 1;
    if (n.type == NORM_CHANNEL) {
        float32x4_t sum = zero;
        for (int i = 0; i < whole; i += 4) sum = vaddq_f32(sum, vld1q_f32(v + i));
        float total = vaddvq_f32(sum);
        for (int i = whole; i < ch; i++) total += v[i];
        mean = total / ch;
        float32x4_t sq = zero, m = vdupq_n_f32(mean);
        for (int i = 0; i < whole; i += 4) {
            float32x4_t d = vsubq_f32(vld1q_f32(v + i), m);
            sq = vfmaq_f32(sq, d, d);
        }
        float squares = vaddvq_f32(sq);
        for (int i = whole; i < ch; i++) squares += (v[i] - mean) * (v[i] - mean);
        rstd = 1.0f / std::sqrt(squares / ch + n.eps);
    }
    if (n.type == NORM_NONE && !relu) return;
    const float32x4_t m = vdupq_n_f32(mean), r = vdupq_n_f32(rstd);
    for (int i = 0; i < whole; i += 4) {
        float32x4_t x = vld1q_f32(v + i);
        if (n.type == NORM_CHANNEL) x = vfmaq_f32(vld1q_f32(c + i), vmulq_f32(vsubq_f32(x, m), r), vld1q_f32(a + i));
        else if (n.type == NORM_AFFINE) x = vfmaq_f32(vld1q_f32(c + i), x, vld1q_f32(a + i));
        vst1q_f32(v + i, relu ? vmaxq_f32(x, zero) : x);
    }
    for (int i = whole; i < ch; i++) {
        float x = v[i];
        if (n.type == NORM_CHANNEL) x = (x - mean) * rstd * a[i] + c[i];
        else if (n.type == NORM_AFFINE) x = x * a[i] + c[i];
        v[i] = relu ? std::max(x, 0.0f) : x;
    }
}
#endif

// `n` of each square's first `n.channels` of `x`, (64, stride), in place, then a ReLU if `relu`
inline void normalize(const Norm& n, float* x, int stride, bool relu) {
    if (n.type == NORM_NONE && !relu) return;
    for (int s = 0; s < SQUARES; s++) normalize_square(n, x + s * stride, n.channels ? n.channels : stride, relu);
}

// each of `count` channels from `first`'s mean and maximum over the board, (2 * count)
inline void board_pool(const float* x, int stride, int first, int count, float* out) {
    float* mean = out;
    float* most = out + count;
    for (int i = 0; i < count; i++) {
        mean[i] = 0;
        most[i] = -INFINITY;
    }
    for (int s = 0; s < SQUARES; s++) {
        const float* v = x + s * stride + first;
#pragma omp simd
        for (int i = 0; i < count; i++) {
            mean[i] += v[i];
            most[i] = v[i] > most[i] ? v[i] : most[i];
        }
    }
    for (int i = 0; i < count; i++) mean[i] *= 1.0f / SQUARES;
}

inline void linear(const Linear& l, const float* in, float* out) {
    for (int o = 0; o < l.out; o++) {
        const float* w = l.w.data() + (size_t)o * l.in;
        float sum = 0;
#pragma omp simd reduction(+ : sum)
        for (int i = 0; i < l.in; i++) sum += w[i] * in[i];
        out[o] = l.b[o] + sum;
    }
}

inline void relu(float* x, int count) {
#pragma omp simd
    for (int i = 0; i < count; i++) x[i] = std::max(x[i], 0.0f);
}

// `b`, (cols), added to each of the 64 squares' first `cols` of `x`, (64, stride), then a ReLU if
// `relu_after`
inline void add_rows(float* x, int stride, const float* b, int cols, bool relu_after) {
    for (int s = 0; s < SQUARES; s++) {
        float* v = x + s * stride;
#pragma omp simd
        for (int i = 0; i < cols; i++) {
            float out = v[i] + b[i];
            v[i] = relu_after ? std::max(out, 0.0f) : out;
        }
    }
}

// x += y, then a ReLU if `relu_after`, over `count` values
inline void residual(float* x, const float* y, int count, bool relu_after) {
#pragma omp simd
    for (int i = 0; i < count; i++) {
        float out = x[i] + y[i];
        x[i] = relu_after ? std::max(out, 0.0f) : out;
    }
}

#endif

// One position: `obs`, (2, 8, 8) channels-first, the mover's discs then the opponent's, to (65)
// policy logits and a value.
inline void forward(const Net& n, Scratch& s, const float* obs, float* logits, float* value) {
    const int C = n.channels;
    float* x = s.x.data();
    float* y = s.y.data();
    // the stem's input, its 2 planes padded to 4 channels
    std::fill(s.padded.begin(), s.padded.begin() + (size_t)PADDED * 4, (Activation)-ACTIVATION_SHIFT);
    for (int sq = 0; sq < SQUARES; sq++)
        for (int p = 0; p < 2; p++)
            s.padded[((sq / 8 + 1) * 10 + sq % 8 + 1) * 4 + p] =
                (Activation)((obs[p * SQUARES + sq] > 0.5f ? 1 : 0) - ACTIVATION_SHIFT);
    convolve(n.stem, s, x, C);
    if (!n.pre) normalize(n.stem_norm, x, C, true);
    for (const Block& b : n.blocks) {
        const int kept = C - b.pool_channels;
        if (n.pre) {
            // y = relu(norm1(x)), quantized for conv1
            std::memcpy(y, x, sizeof(float) * SQUARES * C);
            normalize(b.norm1, y, C, true);
            quantize(b.conv1, y, C, s.padded.data());
            convolve(b.conv1, s, y, C);
            normalize(b.norm2, y, C, true);
        } else {
            quantize(b.conv1, x, C, s.padded.data());
            convolve(b.conv1, s, y, C);
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
        convolve(b.conv2, s, y, C);
        if (!n.pre) normalize(b.norm2, y, C, false);
        residual(x, y, SQUARES * C, !n.pre);
    }
    if (n.pre) normalize(n.final_norm, x, C, true);

    // the heads: the three 1x1 convolutions at once, then in float
    const int H = n.head_channels, H3 = 3 * H;
    float* h = s.h.data();
    quantize(n.head, x, C, s.padded.data());
    convolve(n.head, s, h, H3);
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
    throw std::runtime_error("the native network needs AVX2, AVX-512 VNNI or NEON's dot product");
}
#endif

// Packs `q`, (cout, cin, k, k) signed 8-bit weights, for `convolve`: for VNNI, [cout / 32][k * k]
// [cin / 4][2][16 outputs][4 inputs], the last 16 outputs alone if cout / 16 is odd; for AVX2, in 16
// bits, [cout / 16][cin / 2][2][8 outputs][2 inputs] for 1x1, and for 3x3 the same of each of the 16
// transformed values, [cout / 16][16][cin / 2][2][8][2], 2G g 2G^T; for NEON, [cout / 8][k * k][cin / 4]
// [8 outputs][4 inputs].
inline void pack(Conv& c, const int8_t* q) {
    const int c4n = c.cin / 4, taps = c.k * c.k;
#if NET_ISA == 3
    c.packed.assign((size_t)c.cout * c.cin * taps, 0);
    size_t at8 = 0;
    for (int g = 0; g < c.cout / 8; g++)
        for (int t = 0; t < taps; t++)
            for (int c4 = 0; c4 < c4n; c4++)
                for (int o = 0; o < 8; o++)
                    for (int k = 0; k < 4; k++) c.packed[at8++] = q[((size_t)(8 * g + o) * c.cin + 4 * c4 + k) * taps + t];
    c.offset.assign(c.cout, 0);
    for (int o = 0; o < c.cout; o++)
        for (size_t i = 0; i < (size_t)c.cin * taps; i++) c.offset[o] += ACTIVATION_SHIFT * q[(size_t)o * c.cin * taps + i];
    return;
#endif
#if NET_ISA == 1
    // G of F(2x2, 3x3), doubled
    static const int G[4][3] = {{2, 0, 0}, {1, 1, 1}, {1, -1, 1}, {0, 0, 2}};
    const int values = c.k == 3 ? 16 : 1;
    c.packed16.assign((size_t)c.cout * c.cin * values, 0);
    size_t at16 = 0;
    for (int g = 0; g < c.cout / 16; g++)
        for (int p = 0; p < values; p++)
            for (int c2 = 0; c2 < c.cin / 2; c2++)
                for (int r = 0; r < 2; r++)
                    for (int o = 0; o < 8; o++)
                        for (int k = 0; k < 2; k++) {
                            const int8_t* w = q + ((size_t)(16 * g + 8 * r + o) * c.cin + 2 * c2 + k) * taps;
                            int u = w[0];
                            if (c.k == 3) {
                                u = 0;
                                for (int a = 0; a < 3; a++)
                                    for (int b = 0; b < 3; b++) u += G[p / 4][a] * w[a * 3 + b] * G[p % 4][b];
                            }
                            c.packed16[at16++] = (int16_t)u;
                        }
    return;
#endif
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
