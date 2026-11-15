// An Othello ResNet's forward pass on CPU, int8 in the convolutions and float everywhere else, one position at a time: what `cpu_net.py` builds from a trained network, for
// `cpu_search.cpp`'s engine. It covers basic blocks in either layout, with BatchNorm folded into the
// convolutions or normalizations between them, global pooling biases, and the pooled heads (see
// `core.networks.azresnet`).
//
// Each convolution takes 8-bit unsigned inputs (a scale and a zero point for the whole tensor, from
// calibration) and 8-bit signed weights (a scale for each output channel), sums in 32-bit integers, and
// writes float outputs, acc * scale + bias, which per-square normalizations, the residual sum and the
// global pooling then take in float, in as few passes over the board as they can: a normalization, its
// ReLU and the next convolution's quantized input in one, where OpenVINO runs separate ops (see
// `Lanes`). The stem's input is the discs, 0 or 1, so its sums are looked up (`stem`).
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

// a profiler's mark at the end of each stage of `forward` (see `profile_native.cpp`), by default none
#ifndef NET_STAGE
#define NET_STAGE(name)
#endif

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

// an allocator of memory 64-byte aligned, for vector loads that don't straddle cache lines
template <class T>
struct Aligned {
    using value_type = T;
    Aligned() = default;
    template <class U>
    Aligned(const Aligned<U>&) {}
    T* allocate(size_t n) { return (T*)::operator new(n * sizeof(T), std::align_val_t(64)); }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(64)); }
    template <class U>
    bool operator==(const Aligned<U>&) const { return true; }
    template <class U>
    bool operator!=(const Aligned<U>&) const { return false; }
};

struct Conv {
    int cin = 0, cout = 0, k = 3;
    // the weights for the kernel (see `pack`)
    std::vector<int8_t, Aligned<int8_t>> packed;
    std::vector<int16_t, Aligned<int16_t>> packed16;
    std::vector<float> scale, bias;  // of each output: acc * scale + bias
    std::vector<int32_t> offset;  // NEON: 128 times each output's weights' sum
    std::vector<int16_t, Aligned<int16_t>> table;  // the stem's, in place of the packed weights (see `stem_table`)
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
    // x86: `w`'s outputs in blocks of a vector's width V (16 or 8), each block's weights in the order
    // `linear` reads them, [in / V][V outputs][V inputs], so that it reads one stream (`set_linear`)
    std::vector<float, Aligned<float>> packed;
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

// `l` of `in` inputs and `out` outputs, `w` (out, in) and `b` (out), with its packed weights
inline void set_linear(Linear& l, int in, int out, const float* w, const float* b) {
    l.in = in;
    l.out = out;
    l.w.assign(w, w + (size_t)in * out);
    l.b.assign(b, b + out);
    l.packed.clear();
#if NET_ISA == 1 || NET_ISA == 2
    constexpr int V = NET_ISA == 2 ? 16 : 8;  // the x86 vector's width
    if (in % V) return;
    l.packed.reserve((size_t)in * (out - out % V));
    for (int o = 0; o + V <= out; o += V)
        for (int i = 0; i < in; i += V)
            for (int k = 0; k < V; k++) {
                const float* row = w + (size_t)(o + k) * in + i;
                l.packed.insert(l.packed.end(), row, row + V);
            }
#endif
}

// what a forward pass writes as it goes, one to a thread
struct Scratch {
    std::vector<Activation, Aligned<Activation>> padded;  // (100, max channels), a convolution's input
    std::vector<float, Aligned<float>> x, y, h;   // (64, max channels): the trunk, a block's middle, the heads
    std::vector<int16_t, Aligned<int16_t>> winograd;  // (16, 16 tiles, max channels): AVX2's transformed input
    // i8mm: the padded input in pairs of neighbouring squares, from even and from odd columns
    std::vector<int8_t, Aligned<int8_t>> pairs;
    explicit Scratch(const Net& n) {
        int c = std::max(n.channels, 4);
        // and room for NEON's loads of 16 channels past the last square's
        padded.assign((size_t)PADDED * c + 16, 0);
        winograd.assign((size_t)16 * 16 * c, 0);
        pairs.assign((size_t)10 * 9 * 16 * ((c + 7) / 8), 0);  // 10 rows of 9 pairs, by eights of channels
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
// any one CPU's caches or timings. A kernel row's three taps and all their channels are one run of
// both the padded input (3 squares of `cin`) and the packed weights, so each is one loop. Out of
// line: GCC's copies inlined into `forward` keep some sums on the stack.
template <int K, int RO>
__attribute__((noinline)) void conv_row(const Conv& c, const int8_t* w, const uint8_t* in, int y, int o, float* out,
                                        int out_stride, bool add) {
    const int cin = c.cin;
    const int steps = K * cin / 4;  // a kernel row's 4 channels at a time
    __m512i acc[8][RO];
    for (int i = 0; i < 8; i++)
        for (int r = 0; r < RO; r++) acc[i][r] = _mm512_setzero_si512();
    auto step = [&](const uint8_t* row, const int8_t* wt, int j) {
        __m512i a[8];
        for (int i = 0; i < 8; i++) a[i] = _mm512_set1_epi32(load_u32(row + i * cin + 4 * j));
        for (int r = 0; r < RO; r++) {
            __m512i wv = _mm512_loadu_si512(wt + (j * RO + r) * 64);
            for (int i = 0; i < 8; i++) acc[i][r] = _mm512_dpbusd_epi32(acc[i][r], a[i], wv);
        }
    };
    for (int dy = 0; dy < K; dy++) {
        // the 1x1's one tap is the 3x3's middle
        const uint8_t* row = in + ((y + (K == 3 ? dy : 1)) * 10 + (K == 3 ? 0 : 1)) * cin;
        const int8_t* wt = w + (size_t)dy * steps * RO * 64;
        int j = 0;
        for (; j + 2 <= steps; j += 2) {
            step(row, wt, j);
            step(row, wt, j + 1);
        }
        if (j < steps) step(row, wt, j);
    }
    for (int r = 0; r < RO; r++) {
        __m512 m = _mm512_loadu_ps(c.scale.data() + o + 16 * r);
        __m512 b = _mm512_loadu_ps(c.bias.data() + o + 16 * r);
        for (int i = 0; i < 8; i++) {
            float* dst = out + (y * 8 + i) * out_stride + o + 16 * r;
            __m512 v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[i][r]), m, b);
            _mm512_storeu_ps(dst, add ? _mm512_add_ps(_mm512_loadu_ps(dst), v) : v);
        }
    }
}

// `c` of the padded input `in`, written to `out`, (64, out_stride), its first `c.cout` columns, or
// added to them if `add` (a residual)
inline void convolve(const Conv& c, Scratch& s, float* out, int out_stride, bool add = false) {
    const Activation* in = s.padded.data();
    const int c4n = c.cin / 4;
    const int pairs = c.cout / 32;
    const size_t taps = (size_t)c.k * c.k;
    for (int g = 0; g < pairs; g++) {
        const int8_t* w = c.packed.data() + (size_t)g * taps * c4n * 128;
        for (int y = 0; y < 8; y++) {
            if (c.k == 3) conv_row<3, 2>(c, w, in, y, 32 * g, out, out_stride, add);
            else conv_row<1, 2>(c, w, in, y, 32 * g, out, out_stride, add);
        }
    }
    if (c.cout % 32) {
        const int8_t* w = c.packed.data() + (size_t)pairs * taps * c4n * 128;
        for (int y = 0; y < 8; y++) {
            if (c.k == 3) conv_row<3, 1>(c, w, in, y, 32 * pairs, out, out_stride, add);
            else conv_row<1, 1>(c, w, in, y, 32 * pairs, out, out_stride, add);
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
inline void winograd_output(const Conv& c, const int32_t* m, int o, float* out, int out_stride, bool add) {
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
                float v0 = std::fma((float)(int32_t)(t[i][0] + t[i][1] + t[i][2]), 0.25f * scale[k], bias[k]);
                float v1 = std::fma((float)(int32_t)(t[i][1] - t[i][2] - t[i][3]), 0.25f * scale[k], bias[k]);
                float* y0 = y + (8 * i) * out_stride + k;
                float* y1 = y0 + out_stride;
                *y0 = add ? *y0 + v0 : v0;
                *y1 = add ? *y1 + v1 : v1;
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

inline void convolve(const Conv& c, Scratch& s, float* out, int out_stride, bool add = false) {
    const Activation* in = s.padded.data();
    if (c.k == 1) {
        // (the heads', never a residual)
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
        winograd_output(c, m, 16 * g, out, out_stride, add);
    }
}
#elif NET_ISA == 3 && defined(__ARM_FEATURE_MATMUL_INT8)
// NEON with i8mm (ARMv8.6; Apple's M2 on, Graviton 3 on, Axion): smmla multiplies a 2x8 block of
// inputs, two neighbouring squares' 8 channels, by a 2x8 block of weights, two outputs' 8 channels,
// into a 2x2 block of sums, twice sdot's products an instruction. The input is laid out in such pairs
// first, from even columns and from odd, since a tap's column shift moves the pairs. The tile is a row's
// 4 pairs by 4 pairs of outputs: 16 sums, 4 input and 4 weight vectors.

// channels in eights, the last padded with zeros
inline int eights(int cin) { return (cin + 7) / 8; }

// `in`, the padded input, (100, cin), to (10 rows, 5 pairs from even columns then 4 from odd, eights,
// 2 squares, 8 channels)
inline void pair_up(const int8_t* in, int cin, int8_t* out) {
    const int e = eights(cin);
    if (cin % 8) std::memset(out, 0, (size_t)10 * 9 * e * 16);
    for (int y = 0; y < 10; y++)
        for (int k = 0; k < 9; k++) {
            int x = k < 5 ? 2 * k : 2 * (k - 5) + 1;
            int8_t* o = out + ((size_t)y * 9 + k) * e * 16;
            const int8_t* left = in + (y * 10 + x) * cin;
            const int8_t* right = left + cin;
            if (cin % 8 == 0) {
                for (int j = 0; j < e; j++) {
                    vst1_s8(o + j * 16, vld1_s8(left + j * 8));
                    vst1_s8(o + j * 16 + 8, vld1_s8(right + j * 8));
                }
            } else {
                for (int ch = 0; ch < cin; ch++) {
                    o[(ch / 8) * 16 + ch % 8] = left[ch];
                    o[(ch / 8) * 16 + 8 + ch % 8] = right[ch];
                }
            }
        }
}

template <int K>
inline void conv_row(const Conv& c, const int8_t* w, const int8_t* pairs, int y, int o, float* out, int out_stride,
                     bool add) {
    const int e = eights(c.cin);
    int32x4_t acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int r = 0; r < 4; r++) acc[i][r] = vdupq_n_s32(0);
    for (int t = 0; t < K * K; t++) {
        int dy = K == 3 ? t / 3 : 1, dx = K == 3 ? t % 3 : 1;
        // pairs from column dx: even columns' pair dx / 2 on, or odd columns' from the first
        const int first = dx % 2 == 0 ? dx / 2 : 5;
        const int8_t* row = pairs + ((size_t)(y + dy) * 9 + first) * e * 16;
        const int8_t* wt = w + (size_t)t * e * 64;
        for (int k = 0; k < e; k++) {
            int8x16_t a[4], b[4];
            for (int i = 0; i < 4; i++) a[i] = vld1q_s8(row + ((size_t)i * e + k) * 16);
            for (int r = 0; r < 4; r++) b[r] = vld1q_s8(wt + k * 64 + r * 16);
            for (int i = 0; i < 4; i++)
                for (int r = 0; r < 4; r++) acc[i][r] = vmmlaq_s32(acc[i][r], a[i], b[r]);
        }
    }
    // each sum block is (square 2i, outputs 2r and 2r + 1), then (square 2i + 1, the same)
    const float32x4_t m0 = vld1q_f32(c.scale.data() + o), m1 = vld1q_f32(c.scale.data() + o + 4);
    const float32x4_t b0 = vld1q_f32(c.bias.data() + o), b1 = vld1q_f32(c.bias.data() + o + 4);
    const int32x4_t f0 = vld1q_s32(c.offset.data() + o), f1 = vld1q_s32(c.offset.data() + o + 4);
    for (int i = 0; i < 4; i++)
        for (int sq = 0; sq < 2; sq++) {
            int32x4_t lo, hi;
            if (sq == 0) {
                lo = vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(acc[i][0]), vreinterpretq_s64_s32(acc[i][1])));
                hi = vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(acc[i][2]), vreinterpretq_s64_s32(acc[i][3])));
            } else {
                lo = vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(acc[i][0]), vreinterpretq_s64_s32(acc[i][1])));
                hi = vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(acc[i][2]), vreinterpretq_s64_s32(acc[i][3])));
            }
            float* dst = out + (y * 8 + 2 * i + sq) * out_stride + o;
            float32x4_t v0 = vfmaq_f32(b0, vcvtq_f32_s32(vaddq_s32(lo, f0)), m0);
            float32x4_t v1 = vfmaq_f32(b1, vcvtq_f32_s32(vaddq_s32(hi, f1)), m1);
            vst1q_f32(dst, add ? vaddq_f32(vld1q_f32(dst), v0) : v0);
            vst1q_f32(dst + 4, add ? vaddq_f32(vld1q_f32(dst + 4), v1) : v1);
        }
}

inline void convolve(const Conv& c, Scratch& s, float* out, int out_stride, bool add = false) {
    int8_t* pairs = s.pairs.data();
    pair_up(s.padded.data(), c.cin, pairs);
    const size_t group = (size_t)c.k * c.k * eights(c.cin) * 64;
    for (int g = 0; g < c.cout / 8; g++)
        for (int y = 0; y < 8; y++) {
            if (c.k == 3) conv_row<3>(c, c.packed.data() + g * group, pairs, y, 8 * g, out, out_stride, add);
            else conv_row<1>(c, c.packed.data() + g * group, pairs, y, 8 * g, out, out_stride, add);
        }
}
#else
// NEON with the dot product: the board's row of 8 squares by 8 outputs, 16 sums, 8 squares' inputs of 16
// channels and 2 weight vectors, 26 of its 32 registers, each weight used 8 times. An sdot multiplies 4
// outputs' weights for 4 channels by one square's 4 channels, picked from its vector by lane.
template <int K>
inline void conv_row(const Conv& c, const int8_t* w, const int8_t* in, int y, int o, float* out, int out_stride,
                     bool add) {
    const int cin = c.cin;
    // a kernel row's three taps and all their channels are one run of both the padded input (3
    // squares of `cin`) and the packed weights, so each is one loop, 4 channels a step
    const int steps = K * cin / 4;
    int32x4_t acc[8][2];
    for (int i = 0; i < 8; i++) acc[i][0] = acc[i][1] = vdupq_n_s32(0);
    for (int dy = 0; dy < K; dy++) {
        // the 1x1's one tap is the 3x3's middle
        const int8_t* row = in + ((y + (K == 3 ? dy : 1)) * 10 + (K == 3 ? 0 : 1)) * cin;
        const int8_t* wt = w + (size_t)dy * steps * 32;
        for (int j = 0; j < steps; j += 4) {
            int8x16_t a[8];
            for (int i = 0; i < 8; i++) a[i] = vld1q_s8(row + i * cin + 4 * j);
            // the lane must be a constant; channels come in fours, so the last group may be short
#define NET_LANE(L)                                                                  \
            if (j + L < steps) {                                                     \
                int8x16_t w0 = vld1q_s8(wt + (j + L) * 32), w1 = vld1q_s8(wt + (j + L) * 32 + 16); \
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
        for (int i = 0; i < 8; i++) {
            float* dst = out + (y * 8 + i) * out_stride + o + 4 * r;
            float32x4_t v = vfmaq_f32(b, vcvtq_f32_s32(vaddq_s32(acc[i][r], offset)), m);
            vst1q_f32(dst, add ? vaddq_f32(vld1q_f32(dst), v) : v);
        }
    }
}

inline void convolve(const Conv& c, Scratch& s, float* out, int out_stride, bool add = false) {
    const Activation* in = s.padded.data();
    const size_t group = (size_t)c.k * c.k * (c.cin / 4) * 32;
    for (int g = 0; g < c.cout / 8; g++)
        for (int y = 0; y < 8; y++) {
            if (c.k == 3) conv_row<3>(c, c.packed.data() + g * group, in, y, 8 * g, out, out_stride, add);
            else conv_row<1>(c, c.packed.data() + g * group, in, y, 8 * g, out, out_stride, add);
        }
}
#endif

// The float work between the convolutions, a vector of `Lanes::W` floats at a time: `Lanes` is each
// level's vector, and the passes below are written once for all three. Each value comes from the same
// operations in the same order as a square at a time, one operation over the board at a time, would
// give it, bit for bit (`tests/cpu_net_checks.cpp`): sums over many squares' channels, or many outputs,
// are reduced together in each level's own order of a vector's lanes (`Lanes::reduce`).
#if NET_ISA == 2
// the first `n` lanes (all 16 when n >= 16), the rest zero
inline __mmask16 lanes(int n) { return n >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << n) - 1); }
inline __m512 load(const float* p, int n) { return _mm512_maskz_loadu_ps(lanes(n), p); }
inline void store(float* p, __m512 v, int n) { _mm512_mask_storeu_ps(p, lanes(n), v); }

struct Lanes {
    using F = __m512;
    static constexpr int W = 16;
    static F zero() { return _mm512_setzero_ps(); }
    static F set1(float v) { return _mm512_set1_ps(v); }
    static F load(const float* p) { return _mm512_loadu_ps(p); }
    static void store(float* p, F v) { _mm512_storeu_ps(p, v); }
    // the first `n` (all when n >= W), the rest zero
    static F load(const float* p, int n) { return net::load(p, n); }
    static void store(float* p, F v, int n) { net::store(p, v, n); }
    static F add(F a, F b) { return _mm512_add_ps(a, b); }
    static F sub(F a, F b) { return _mm512_sub_ps(a, b); }
    static F mul(F a, F b) { return _mm512_mul_ps(a, b); }
    static F div(F a, F b) { return _mm512_div_ps(a, b); }
    static F sqrt(F a) { return _mm512_sqrt_ps(a); }
    static F fma(F a, F b, F c) { return _mm512_fmadd_ps(a, b, c); }  // a * b + c, rounded once
    // the sums of 3 rows of W 16-bit integers, as floats
    static F widen(const int16_t* a, const int16_t* b, const int16_t* c) {
        __m256i s = _mm256_add_epi16(_mm256_add_epi16(_mm256_loadu_si256((const __m256i*)a), _mm256_loadu_si256((const __m256i*)b)),
                                     _mm256_loadu_si256((const __m256i*)c));
        return _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(s));
    }
    // the first 16 of `p` as bits: p[i] > 0.5
    static uint64_t bits(const float* p) { return _mm512_cmp_ps_mask(_mm512_loadu_ps(p), set1(0.5f), _CMP_GT_OQ); }
    // a * b, rounded on its own: GCC's intrinsics are its vector operators, and fuses an add of a
    // product into an fma where it can
    static F product(F a, F b) {
        F p = _mm512_mul_ps(a, b);
        asm("" : "+v"(p));
        return p;
    }
    // a normalization's ReLU, and a sum's (the residual's, a pooling bias's)
    static F relu(F v) { return _mm512_max_ps(v, zero()); }
    static F relu_sum(F v) { return _mm512_max_ps(v, zero()); }
    // the sums of 16 vectors' lanes, lane s v[s]'s, each added as `_mm512_reduce_add_ps` adds them:
    // lane j and lane j + 8, those sums' j and j + 4, then j + 2, then j + 1. Pairs of vectors are
    // shuffled so that each addition adds those same pairs, for several vectors at once.
    static F reduce(const F* v) {
        F r1[8], r2[4], r3[2];
        for (int k = 0; k < 8; k++)  // [v[2k] lanes 0-7, v[2k + 1] lanes 0-7]
            r1[k] = add(_mm512_shuffle_f32x4(v[2 * k], v[2 * k + 1], _MM_SHUFFLE(1, 0, 1, 0)),
                        _mm512_shuffle_f32x4(v[2 * k], v[2 * k + 1], _MM_SHUFFLE(3, 2, 3, 2)));
        for (int k = 0; k < 4; k++)  // 4 lanes each of vectors 4k to 4k + 3
            r2[k] = add(_mm512_shuffle_f32x4(r1[2 * k], r1[2 * k + 1], _MM_SHUFFLE(2, 0, 2, 0)),
                        _mm512_shuffle_f32x4(r1[2 * k], r1[2 * k + 1], _MM_SHUFFLE(3, 1, 3, 1)));
        for (int k = 0; k < 2; k++)  // in each 4 lanes b, 2 of vector 8k + b, then 2 of 8k + 4 + b
            r3[k] = add(_mm512_shuffle_ps(r2[2 * k], r2[2 * k + 1], _MM_SHUFFLE(1, 0, 1, 0)),
                        _mm512_shuffle_ps(r2[2 * k], r2[2 * k + 1], _MM_SHUFFLE(3, 2, 3, 2)));
        // lane 4b + i is vector b + 4i's
        F r4 = add(_mm512_shuffle_ps(r3[0], r3[1], _MM_SHUFFLE(2, 0, 2, 0)),
                   _mm512_shuffle_ps(r3[0], r3[1], _MM_SHUFFLE(3, 1, 3, 1)));
        return _mm512_permutexvar_ps(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), r4);
    }
    // 16 vectors as a 16 by 16 matrix, transposed in place: r[j] lane i to r[i] lane j
    static void transpose(F* r) {
        F a[16], b[16];
        for (int k = 0; k < 16; k += 2) {
            a[k] = _mm512_unpacklo_ps(r[k], r[k + 1]);
            a[k + 1] = _mm512_unpackhi_ps(r[k], r[k + 1]);
        }
        for (int k = 0; k < 16; k += 4) {  // b[k + e]: in each 4 lanes q, lane 4q + e of r[k] to r[k + 3]
            b[k] = _mm512_shuffle_ps(a[k], a[k + 2], _MM_SHUFFLE(1, 0, 1, 0));
            b[k + 1] = _mm512_shuffle_ps(a[k], a[k + 2], _MM_SHUFFLE(3, 2, 3, 2));
            b[k + 2] = _mm512_shuffle_ps(a[k + 1], a[k + 3], _MM_SHUFFLE(1, 0, 1, 0));
            b[k + 3] = _mm512_shuffle_ps(a[k + 1], a[k + 3], _MM_SHUFFLE(3, 2, 3, 2));
        }
        for (int e = 0; e < 4; e++) {
            F c1 = _mm512_shuffle_f32x4(b[e], b[4 + e], _MM_SHUFFLE(2, 0, 2, 0));
            F c2 = _mm512_shuffle_f32x4(b[e], b[4 + e], _MM_SHUFFLE(3, 1, 3, 1));
            F d1 = _mm512_shuffle_f32x4(b[8 + e], b[12 + e], _MM_SHUFFLE(2, 0, 2, 0));
            F d2 = _mm512_shuffle_f32x4(b[8 + e], b[12 + e], _MM_SHUFFLE(3, 1, 3, 1));
            r[e] = _mm512_shuffle_f32x4(c1, d1, _MM_SHUFFLE(2, 0, 2, 0));
            r[4 + e] = _mm512_shuffle_f32x4(c2, d2, _MM_SHUFFLE(2, 0, 2, 0));
            r[8 + e] = _mm512_shuffle_f32x4(c1, d1, _MM_SHUFFLE(3, 1, 3, 1));
            r[12 + e] = _mm512_shuffle_f32x4(c2, d2, _MM_SHUFFLE(3, 1, 3, 1));
        }
    }
    // std::max(v, 0.0f), as the compiler's loops take it (a -0 kept)
    static F max0(F v) { return _mm512_max_ps(zero(), v); }
    // a board pool's maximum, so far `most`, with `v`
    static F most(F most, F v) { return _mm512_max_ps(most, v); }
};

// a square's channels quantized for `c`'s input
struct Quantizer {
    const float* inverse;
    __m512i zero_point;
    explicit Quantizer(const Conv& c) : inverse(c.inverse.data()), zero_point(_mm512_set1_epi32(c.zero_point)) {}
    // the first `n` lanes of `v`, channels `ch` on, into the square's place `q`
    void operator()(__m512 v, int ch, Activation* q, int n = 16) const {
        __m512i i = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v, load(inverse + ch, n))), zero_point);
        // at least 0, then at most 255 as it narrows
        __m128i b = _mm512_cvtusepi32_epi8(_mm512_max_epi32(i, _mm512_setzero_si512()));
        if (n >= 16) _mm_storeu_si128((__m128i*)(q + ch), b);
        else _mm_mask_storeu_epi8(q + ch, lanes(n), b);
    }
    // 16 channels from `ch`, `v[1]`
    void group(const __m512* v, int ch, Activation* q) const { (*this)(v[0], ch, q); }
};

inline void linear(const Linear& l, const float* in, float* out) {
    int o = 0;
    // 16 outputs at a time, their sums reduced together, from the packed weights (a line a load,
    // fetched 2 KB ahead); the inputs outside the outputs, so that the 16 sums stay in registers
    if (l.in % 16 == 0)
        for (const float* w = l.packed.data(); o + 16 <= l.out; o += 16) {
            __m512 acc[16];
            for (int k = 0; k < 16; k++) acc[k] = _mm512_setzero_ps();
            for (int i = 0; i < l.in; i += 16, w += 256) {
                const __m512 x = _mm512_loadu_ps(in + i);
                for (int k = 0; k < 16; k++) {
                    _mm_prefetch((const char*)(w + 16 * k + 512), _MM_HINT_T0);
                    acc[k] = _mm512_fmadd_ps(_mm512_load_ps(w + 16 * k), x, acc[k]);
                }
            }
            _mm512_storeu_ps(out + o, _mm512_add_ps(_mm512_loadu_ps(l.b.data() + o), Lanes::reduce(acc)));
        }
    for (; o < l.out; o++) {
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

#elif NET_ISA == 1
// AVX2: the first `n` of 8 lanes (all when n >= 8)
inline __m256i lanes8(int n) {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(n), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}
inline __m256 load8(const float* p, int n) { return n >= 8 ? _mm256_loadu_ps(p) : _mm256_maskload_ps(p, lanes8(n)); }
inline void store8(float* p, __m256 v, int n) {
    if (n >= 8) _mm256_storeu_ps(p, v);
    else _mm256_maskstore_ps(p, lanes8(n), v);
}

struct Lanes {
    using F = __m256;
    static constexpr int W = 8;
    static F zero() { return _mm256_setzero_ps(); }
    static F set1(float v) { return _mm256_set1_ps(v); }
    static F load(const float* p) { return _mm256_loadu_ps(p); }
    static void store(float* p, F v) { _mm256_storeu_ps(p, v); }
    static F load(const float* p, int n) { return load8(p, n); }
    static void store(float* p, F v, int n) { store8(p, v, n); }
    static F add(F a, F b) { return _mm256_add_ps(a, b); }
    static F sub(F a, F b) { return _mm256_sub_ps(a, b); }
    static F mul(F a, F b) { return _mm256_mul_ps(a, b); }
    static F div(F a, F b) { return _mm256_div_ps(a, b); }
    static F sqrt(F a) { return _mm256_sqrt_ps(a); }
    static F fma(F a, F b, F c) { return _mm256_fmadd_ps(a, b, c); }
    static F widen(const int16_t* a, const int16_t* b, const int16_t* c) {
        __m128i s = _mm_add_epi16(_mm_add_epi16(_mm_loadu_si128((const __m128i*)a), _mm_loadu_si128((const __m128i*)b)),
                                  _mm_loadu_si128((const __m128i*)c));
        return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(s));
    }
    static uint64_t bits(const float* p) {
        return (uint64_t)_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(p), set1(0.5f), _CMP_GT_OQ));
    }
    // a * b, rounded on its own (see AVX-512's)
    static F product(F a, F b) {
        F p = _mm256_mul_ps(a, b);
        asm("" : "+x"(p));
        return p;
    }
    // a normalization's ReLU, max(v, 0); and a sum's, std::max(v, 0.0f) as the compiler's loops took
    // it, which keeps a -0
    static F relu(F v) { return _mm256_max_ps(v, zero()); }
    static F relu_sum(F v) { return _mm256_max_ps(zero(), v); }
    // the sums of 8 vectors' lanes, lane s v[s]'s, each added as a vector's lanes were added on their
    // own: lane j and j + 4, then j + 2, then j + 1
    static F reduce(const F* v) {
        F r1[4], r2[2];
        for (int k = 0; k < 4; k++)  // [v[2k] lanes 0-3, v[2k + 1] lanes 0-3]
            r1[k] = add(_mm256_permute2f128_ps(v[2 * k], v[2 * k + 1], 0x20),
                        _mm256_permute2f128_ps(v[2 * k], v[2 * k + 1], 0x31));
        for (int k = 0; k < 2; k++)  // in each half b, 2 of vector 4k + b, then 2 of 4k + 2 + b
            r2[k] = add(_mm256_shuffle_ps(r1[2 * k], r1[2 * k + 1], _MM_SHUFFLE(1, 0, 1, 0)),
                        _mm256_shuffle_ps(r1[2 * k], r1[2 * k + 1], _MM_SHUFFLE(3, 2, 3, 2)));
        // lanes: vectors 0, 2, 4, 6, 1, 3, 5, 7
        F r3 = add(_mm256_shuffle_ps(r2[0], r2[1], _MM_SHUFFLE(2, 0, 2, 0)),
                   _mm256_shuffle_ps(r2[0], r2[1], _MM_SHUFFLE(3, 1, 3, 1)));
        return _mm256_permutevar8x32_ps(r3, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    }
    // 8 vectors as an 8 by 8 matrix, transposed in place: r[j] lane i to r[i] lane j
    static void transpose(F* r) {
        F t[8], u[8];
        for (int k = 0; k < 8; k += 2) {
            t[k] = _mm256_unpacklo_ps(r[k], r[k + 1]);
            t[k + 1] = _mm256_unpackhi_ps(r[k], r[k + 1]);
        }
        for (int k = 0; k < 8; k += 4)  // u[k + e]: lanes e and e + 4 of r[k] to r[k + 3]
            for (int j = 0; j < 2; j++) {
                u[k + 2 * j] = _mm256_shuffle_ps(t[k + j], t[k + j + 2], _MM_SHUFFLE(1, 0, 1, 0));
                u[k + 2 * j + 1] = _mm256_shuffle_ps(t[k + j], t[k + j + 2], _MM_SHUFFLE(3, 2, 3, 2));
            }
        for (int e = 0; e < 4; e++) {
            r[e] = _mm256_permute2f128_ps(u[e], u[e + 4], 0x20);
            r[e + 4] = _mm256_permute2f128_ps(u[e], u[e + 4], 0x31);
        }
    }
    static F max0(F v) { return _mm256_max_ps(zero(), v); }
    // a board pool's maximum, so far `most`, with `v`: v > most ? v : most, as the compiler's loop took it
    static F most(F most, F v) { return _mm256_max_ps(v, most); }
};

// a square's channels quantized for `c`'s input
struct Quantizer {
    const float* inverse;
    __m256 low, high;
    __m256i zero_point;
    // x * inverse clamped to -zero point to 255 - zero point before it's rounded, which gives what
    // rounding and then clamping to 0 to 255 does (rounding keeps the order, and the bounds are whole),
    // infinities included; a NaN, which max takes as its second operand, to 0
    explicit Quantizer(const Conv& c)
        : inverse(c.inverse.data()), low(_mm256_set1_ps((float)-c.zero_point)), high(_mm256_set1_ps(255.0f - c.zero_point)),
          zero_point(_mm256_set1_epi32(c.zero_point)) {}
    __m256i integers(__m256 v, int ch, int n = 8) const {
        __m256 t = _mm256_mul_ps(v, load8(inverse + ch, n));
        t = _mm256_min_ps(_mm256_max_ps(t, low), high);
        // rounded to the nearest, ties to even, as `rint`
        return _mm256_add_epi32(_mm256_cvtps_epi32(t), zero_point);
    }
    // the first `n` lanes of `v` (8, or 4: channels come in fours), channels `ch` on, into the square's
    // place `q`
    void operator()(__m256 v, int ch, Activation* q, int n = 8) const {
        __m256i i = integers(v, ch, n);
        __m128i packed = _mm_packs_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256(i, 1));
        if (n >= 8) _mm_storeu_si128((__m128i*)(q + ch), packed);
        else _mm_storel_epi64((__m128i*)(q + ch), packed);
    }
    // 16 channels from `ch`, `v[2]`: both vectors narrowed at once, each 128-bit lane's 4 of each,
    // then put in order
    void group(const __m256* v, int ch, Activation* q) const {
        __m256i words = _mm256_packs_epi32(integers(v[0], ch), integers(v[1], ch + 8));
        _mm256_storeu_si256((__m256i*)(q + ch), _mm256_permute4x64_epi64(words, _MM_SHUFFLE(3, 1, 2, 0)));
    }
};

#else
// NEON: 4 lanes
struct Lanes {
    using F = float32x4_t;
    static constexpr int W = 4;
    static F zero() { return vdupq_n_f32(0); }
    static F set1(float v) { return vdupq_n_f32(v); }
    static F load(const float* p) { return vld1q_f32(p); }
    static void store(float* p, F v) { vst1q_f32(p, v); }
    // channels come in fours: always all 4
    static F load(const float* p, int) { return vld1q_f32(p); }
    static void store(float* p, F v, int) { vst1q_f32(p, v); }
    static F add(F a, F b) { return vaddq_f32(a, b); }
    static F sub(F a, F b) { return vsubq_f32(a, b); }
    static F mul(F a, F b) { return vmulq_f32(a, b); }
    static F div(F a, F b) { return vdivq_f32(a, b); }
    static F sqrt(F a) { return vsqrtq_f32(a); }
    static F fma(F a, F b, F c) { return vfmaq_f32(c, a, b); }
    static F widen(const int16_t* a, const int16_t* b, const int16_t* c) {
        return vcvtq_f32_s32(vmovl_s16(vadd_s16(vadd_s16(vld1_s16(a), vld1_s16(b)), vld1_s16(c))));
    }
    static uint64_t bits(const float* p) {
        uint64_t b = 0;
        for (int i = 0; i < W; i++) b |= (uint64_t)(p[i] > 0.5f) << i;
        return b;
    }
    // a normalization's ReLU, max(v, 0); and a sum's, std::max(v, 0.0f) as the compiler's loops took
    // it, which keeps a -0
    static F relu(F v) { return vmaxq_f32(v, zero()); }
    static F relu_sum(F v) { return vbslq_f32(vcltq_f32(v, zero()), zero(), v); }
    // the sums of 4 vectors' lanes, lane s v[s]'s, each added as `vaddvq_f32` adds them, in pairs:
    // (lane 0 + lane 1) + (lane 2 + lane 3)
    static F reduce(const F* v) { return vpaddq_f32(vpaddq_f32(v[0], v[1]), vpaddq_f32(v[2], v[3])); }
    // a board pool's maximum, so far `most`, with `v`: v > most ? v : most, as the compiler's loop took it
    static F most(F most, F v) { return vbslq_f32(vcgtq_f32(v, most), v, most); }
};

// a square's channels quantized for `c`'s input, less 128 for the signed dot product: round(x *
// inverse) + zero point - 128 within -128 to 127. The conversion, the addition and the narrowing all
// saturate, which clamps every value, infinities included, as clamping the float to the zero point's
// range and then the integer did (and a NaN converts to 0 either way).
struct Quantizer {
    const float* inverse;
    int32x4_t zero_point;
    explicit Quantizer(const Conv& c) : inverse(c.inverse.data()), zero_point(vdupq_n_s32(c.zero_point - ACTIVATION_SHIFT)) {}
    int32x4_t integers(float32x4_t v, int ch) const {
        // rounded to the nearest, ties to even, as `rint`
        return vqaddq_s32(vcvtnq_s32_f32(vmulq_f32(v, vld1q_f32(inverse + ch))), zero_point);
    }
    // `v`, channels `ch` to `ch + 4`, into the square's place `q`
    void operator()(float32x4_t v, int ch, Activation* q, int = 4) const {
        int8x8_t b = vqmovn_s16(vcombine_s16(vqmovn_s32(integers(v, ch)), vdup_n_s16(0)));
        vst1_lane_s32((int32_t*)(q + ch), vreinterpret_s32_s8(b), 0);
    }
    // 16 channels from `ch`, `v[4]`
    void group(const float32x4_t* v, int ch, Activation* q) const {
        int16x8_t lo = vqmovn_high_s32(vqmovn_s32(integers(v[0], ch)), integers(v[1], ch + 4));
        int16x8_t hi = vqmovn_high_s32(vqmovn_s32(integers(v[2], ch + 8)), integers(v[3], ch + 12));
        vst1q_s8(q + ch, vqmovn_high_s16(vqmovn_s16(lo), hi));
    }
};
#endif

#if NET_ISA != 2
// AVX2's and NEON's: plain loops, which the compiler vectorizes
inline void linear(const Linear& l, const float* in, float* out) {
    int o = 0;
#if NET_ISA == 1
    // 8 outputs at a time, summed as the compiler's loop below sums one: products, unfused, added up
    // 8 lanes at a time, then the lanes in order onto 0. From the packed weights; the inputs outside
    // the outputs, so that the 8 sums stay in registers.
    if (l.in % 8 == 0)
        for (const float* w = l.packed.data(); o + 8 <= l.out; o += 8) {
            __m256 acc[8];
            for (int k = 0; k < 8; k++) acc[k] = _mm256_setzero_ps();
            for (int i = 0; i < l.in; i += 8, w += 64) {
                const __m256 x = _mm256_loadu_ps(in + i);
                for (int k = 0; k < 8; k++) {
                    // a line every two loads, fetched 2 KB ahead
                    if (k % 2 == 0) _mm_prefetch((const char*)(w + 8 * k + 512), _MM_HINT_T0);
                    acc[k] = _mm256_add_ps(acc[k], Lanes::product(_mm256_load_ps(w + 8 * k), x));
                }
            }
            // lane j of each output's sum to acc[j]
            Lanes::transpose(acc);
            __m256 total = _mm256_setzero_ps();
            for (int j = 0; j < 8; j++) total = _mm256_add_ps(total, acc[j]);
            _mm256_storeu_ps(out + o, _mm256_add_ps(total, _mm256_loadu_ps(l.b.data() + o)));
        }
#endif
    for (; o < l.out; o++) {
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
#endif

using F = Lanes::F;
constexpr int W = Lanes::W;
// the passes take channels 16 at a time, G vectors: a fixed number, which the compiler keeps in
// registers (channels come in 16s, but for a convolution's input after a block's pooled channels)
constexpr int G = 16 / W;

// where square `s` goes in a convolution's padded input of `cin` channels
inline Activation* padded_square(Activation* padded, int s, int cin) {
    return padded + ((s / 8 + 1) * 10 + s % 8 + 1) * cin;
}

// the border all round the board of `c`'s padded input, its zero point; the board is written by the
// passes below
inline void border(const Conv& c, Activation* padded) {
    const Activation zero = (Activation)(c.zero_point - ACTIVATION_SHIFT);
    const int cin = c.cin;
    std::fill(padded, padded + 11 * cin, zero);  // the first row, and the second's first square
    for (int r = 1; r < 8; r++)  // a row's last square and the next's first
        std::fill(padded + (r * 10 + 9) * cin, padded + (r * 10 + 11) * cin, zero);
    std::fill(padded + 89 * cin, padded + PADDED * cin, zero);  // the ninth row's last, and the last row
}

// a channel normalization's mean and 1 / std for each square of `x`, (64, stride), over its first `ch`
// (a multiple of W) channels: W squares at a time, so that their sums don't wait on each other. The
// channels' loop is the outer one, so that the W sums stay in registers (under a loop of a runtime
// length the compiler keeps them in memory); each square's sum still adds its channels in order.
inline void norm_stats(const float* x, int stride, int ch, float eps, float* mean, float* rstd) {
    const F count = Lanes::set1((float)ch);
    for (int s0 = 0; s0 < SQUARES; s0 += W) {
        const float* v = x + (size_t)s0 * stride;
        F acc[W];
        for (int s = 0; s < W; s++) acc[s] = Lanes::zero();
        for (int i = 0; i < ch; i += W)
            for (int s = 0; s < W; s++) acc[s] = Lanes::add(acc[s], Lanes::load(v + (size_t)s * stride + i));
        Lanes::store(mean + s0, Lanes::div(Lanes::reduce(acc), count));
        F m[W];
        for (int s = 0; s < W; s++) {
            m[s] = Lanes::set1(mean[s0 + s]);
            acc[s] = Lanes::zero();
        }
        for (int i = 0; i < ch; i += W)
            for (int s = 0; s < W; s++) {
                F d = Lanes::sub(Lanes::load(v + (size_t)s * stride + i), m[s]);
                acc[s] = Lanes::fma(d, d, acc[s]);
            }
        Lanes::store(rstd + s0, Lanes::div(Lanes::set1(1.0f), Lanes::sqrt(Lanes::add(Lanes::div(Lanes::reduce(acc), count),
                                                                                       Lanes::set1(eps)))));
    }
}

// a normalization of type `TYPE` of every square, with its statistics for a channel normalization
template <int TYPE>
struct Normalizer {
    const float* a;
    const float* c;
    alignas(64) float mean[SQUARES], rstd[SQUARES];
    Normalizer(const Norm& n, const float* x, int stride, int ch) : a(n.a.data()), c(n.c.data()) {
        if (TYPE == NORM_CHANNEL) norm_stats(x, stride, ch, n.eps, mean, rstd);
    }
    // square `s`'s
    struct Square {
        const float* a;
        const float* c;
        F mean, rstd;
        // its channels `i` to `i + W`, `v`, normalized
        F operator()(F v, int i) const {
            if (TYPE == NORM_CHANNEL)
                return Lanes::fma(Lanes::mul(Lanes::sub(v, mean), rstd), Lanes::load(a + i), Lanes::load(c + i));
            if (TYPE == NORM_AFFINE) return Lanes::fma(v, Lanes::load(a + i), Lanes::load(c + i));
            return v;
        }
    };
    Square square(int s) const {
        if (TYPE == NORM_CHANNEL) return {a, c, Lanes::set1(mean[s]), Lanes::set1(rstd[s])};
        return {a, c, Lanes::zero(), Lanes::zero()};
    }
    // channel `i` of squares `s` to `s + W`, `v`, normalized
    F column(F v, int s, int i) const {
        if (TYPE == NORM_CHANNEL)
            return Lanes::fma(Lanes::mul(Lanes::sub(v, Lanes::load(mean + s)), Lanes::load(rstd + s)), Lanes::set1(a[i]),
                              Lanes::set1(c[i]));
        if (TYPE == NORM_AFFINE) return Lanes::fma(v, Lanes::set1(a[i]), Lanes::set1(c[i]));
        return v;
    }
};

// `f` called with the `Normalizer` of `n`, for `x`, (64, stride), its first `ch` (a multiple of W)
// channels
template <class Fn>
inline void with_norm(const Norm& n, const float* x, int stride, int ch, Fn f) {
    if (n.type == NORM_CHANNEL) f(Normalizer<NORM_CHANNEL>(n, x, stride, ch));
    else if (n.type == NORM_AFFINE) f(Normalizer<NORM_AFFINE>(n, x, stride, ch));
    else f(Normalizer<NORM_NONE>(n, x, stride, ch));
}

// `n` of each square's first `ch` (a multiple of W) channels of `x`, (64, stride), in place, then a
// ReLU if `relu`
inline void normalize(const Norm& n, float* x, int stride, int ch, bool relu) {
    if (n.type == NORM_NONE && !relu) return;
    with_norm(n, x, stride, ch, [&](const auto& norm) {
        for (int s = 0; s < SQUARES; s++) {
            const auto square = norm.square(s);
            float* v = x + s * stride;
            for (int i = 0; i < ch; i += W) {
                F out = square(Lanes::load(v + i), i);
                Lanes::store(v + i, relu ? Lanes::relu(out) : out);
            }
        }
    });
}

// a square's `cin` (a multiple of 16) channels quantized for `quantize`'s convolution into its place
// `q`, 16 at a time, each W from `i` given by `value(i)`. (`quantize` by value: the float stores could
// alias it otherwise, and the compiler would load it again after each. Always inlined: Apple's clang
// otherwise calls it, a square at a time.)
template <class Fn>
__attribute__((always_inline)) inline void quantize_square(const Quantizer quantize, const int cin, Activation* q, Fn value) {
    for (int i = 0; i < cin; i += 16) {
        F out[G];
        for (int g = 0; g < G; g++) out[g] = value(i + g * W);
        quantize.group(out, i, q);
    }
}

// relu(n(x)), each square's `c.cin` (a multiple of 16) channels of `x`, (64, stride), quantized for `c`
// into `padded`; and kept in `keep` too if given
inline void normalize_quantize(const Norm& n, const float* x, int stride, const Conv& c, Activation* padded,
                               float* keep = nullptr) {
    border(c, padded);
    const Quantizer quantize(c);
    const int cin = c.cin;
    with_norm(n, x, stride, cin, [&](const auto& norm) {
        for (int s = 0; s < SQUARES; s++) {
            const auto square = norm.square(s);
            const float* v = x + s * stride;
            quantize_square(quantize, cin, padded_square(padded, s, cin), [&](int i) {
                F out = Lanes::relu(square(Lanes::load(v + i), i));
                if (keep) Lanes::store(keep + s * stride + i, out);
                return out;
            });
        }
    });
}

// the post layout's residual: x = max(x + n(y), 0), each square's `c.cin` channels of `x` and `y`,
// (64, stride), quantized for `c`, the next convolution, into `padded`
inline void residual_quantize(const Norm& n, const float* y, float* x, int stride, const Conv& c, Activation* padded) {
    border(c, padded);
    const Quantizer quantize(c);
    const int cin = c.cin;
    with_norm(n, y, stride, cin, [&](const auto& norm) {
        for (int s = 0; s < SQUARES; s++) {
            const auto square = norm.square(s);
            float* v = x + s * stride;
            quantize_square(quantize, cin, padded_square(padded, s, cin), [&](int j) {
                F out = Lanes::relu_sum(Lanes::add(Lanes::load(v + j), square(Lanes::load(y + s * stride + j), j)));
                Lanes::store(v + j, out);
                return out;
            });
        }
    });
}

// a block's global pooling bias, and its second convolution's input: of `y`, (64, stride), the first
// convolution's output, n of each square's `ch` channels; the last `ch - c.cin` after a ReLU pooled
// (each one's mean and maximum over the board) and through `pool`, the bias; the first `c.cin` plus the
// bias, after a ReLU (`pre`) or with one after the sum, quantized for `c` into `padded`. y stays as it
// was.
inline void pool_bias_quantize(const Norm& n, const Linear& pool, const float* y, int stride, int ch, bool pre,
                               const Conv& c, Activation* padded) {
    border(c, padded);
    const Quantizer quantize(c);
    const int kept = c.cin, count = ch - kept;
    with_norm(n, y, stride, ch, [&](const auto& norm) {
        float pooled[2 * 512], bias[1024];
        // 16 channels at a time from the 16 the first pooled one is in (whole vectors, all within the
        // `ch`); only the pooled ones' kept
        for (int first = kept & ~15; first < ch; first += 16) {
            F sum[G], most[G];
            for (int g = 0; g < G; g++) {
                sum[g] = Lanes::zero();
                most[g] = Lanes::set1(-INFINITY);
            }
            for (int s = 0; s < SQUARES; s++) {
                const auto square = norm.square(s);
                for (int g = 0; g < G; g++) {
                    const int i = first + g * W;
                    const F v = Lanes::relu(square(Lanes::load(y + s * stride + i), i));
                    sum[g] = Lanes::add(sum[g], v);
                    most[g] = Lanes::most(most[g], v);
                }
            }
            alignas(64) float mean[16], top[16];
            for (int g = 0; g < G; g++) {
                Lanes::store(mean + g * W, Lanes::mul(sum[g], Lanes::set1(1.0f / SQUARES)));
                Lanes::store(top + g * W, most[g]);
            }
            for (int j = std::max(kept - first, 0); j < 16; j++) {
                pooled[first + j - kept] = mean[j];
                pooled[count + first + j - kept] = top[j];
            }
        }
        linear(pool, pooled, bias);
        const int whole = kept & ~15;
        for (int s = 0; s < SQUARES; s++) {
            const auto square = norm.square(s);
            const float* v = y + s * stride;
            Activation* q = padded_square(padded, s, kept);
            quantize_square(quantize, whole, q, [&](int i) {
                const F x = square(Lanes::load(v + i), i), b = Lanes::load(bias + i);
                return pre ? Lanes::add(Lanes::relu(x), b) : Lanes::relu_sum(Lanes::add(x, b));
            });
            // channels come in fours: the norm's whole vector is within `ch`, its last lanes unused
            for (int i = whole; i < kept; i += W) {
                const int m = kept - i;
                const F x = square(Lanes::load(v + i), i), b = Lanes::load(bias + i, m);
                quantize(pre ? Lanes::add(Lanes::relu(x), b) : Lanes::relu_sum(Lanes::add(x, b)), i, q, m);
            }
        }
    });
}

// relu(n(x)) of each square's `ch` (a multiple of 16) channels of `x`, (64, stride), pooled:
// each one's mean and maximum over the board, to (2 * ch); x as it was
inline void normalize_pool(const Norm& n, const float* x, int stride, int ch, float* out) {
    with_norm(n, x, stride, ch, [&](const auto& norm) {
        for (int first = 0; first < ch; first += 16) {
            F sum[G], most[G];
            for (int g = 0; g < G; g++) {
                sum[g] = Lanes::zero();
                most[g] = Lanes::set1(-INFINITY);
            }
            for (int s = 0; s < SQUARES; s++) {
                const auto square = norm.square(s);
                for (int g = 0; g < G; g++) {
                    const int i = first + g * W;
                    F v = Lanes::relu(square(Lanes::load(x + s * stride + i), i));
                    sum[g] = Lanes::add(sum[g], v);
                    most[g] = Lanes::most(most[g], v);
                }
            }
            for (int g = 0; g < G; g++) {
                Lanes::store(out + first + g * W, Lanes::mul(sum[g], Lanes::set1(1.0f / SQUARES)));
                Lanes::store(out + ch + first + g * W, most[g]);
            }
        }
    });
}

#if NET_ISA != 3
// the policy's logit of each square: the sum, in order onto 0, of std::max(p + bias, 0) * square_w, the
// products rounded on their own, over p = n(h), the square's first H (a multiple of W) channels of h,
// (64, stride); as the compiler's loop summed one square, here W squares a lane each, their channels
// transposed. K groups of W squares at a time, so that their sums don't wait on each other (AVX-512's
// 32 registers hold 4 groups' columns; AVX2's 16 only one group's).
inline void policy_logits(const Norm& n, const float* h, int stride, int H, const float* bias, const float* square_w,
                          float* logits) {
    constexpr int K = NET_ISA == 2 ? 4 : 1;
    with_norm(n, h, stride, H, [&](const auto& norm) {
        for (int s0 = 0; s0 < SQUARES; s0 += K * W) {
            F logit[K];
            for (int k = 0; k < K; k++) logit[k] = Lanes::zero();
            for (int i0 = 0; i0 < H; i0 += W) {
                F column[K][W];
                for (int k = 0; k < K; k++) {
                    for (int s = 0; s < W; s++) column[k][s] = Lanes::load(h + (s0 + k * W + s) * stride + i0);
                    Lanes::transpose(column[k]);
                }
                for (int j = 0; j < W; j++) {
                    const int i = i0 + j;
                    const F b = Lanes::set1(bias[i]), w = Lanes::set1(square_w[i]);
                    for (int k = 0; k < K; k++) {
                        F p = Lanes::add(norm.column(column[k][j], s0 + k * W, i), b);
                        logit[k] = Lanes::add(logit[k], Lanes::product(Lanes::max0(p), w));
                    }
                }
            }
            for (int k = 0; k < K; k++) Lanes::store(logits + s0 + k * W, logit[k]);
        }
    });
}
#endif

// The stem's convolution of the board, `obs`, (2, 8, 8), its discs 0 or 1, to `out`, (64, c.cout): each
// square's sums are those of its 3 rows of 3 squares' discs' weights, looked up (`stem_table`). The
// same integers as the convolution kernels sum, scaled the same way, so the same floats.
inline void stem(const Conv& c, const float* obs, float* out) {
    // each side's discs, a bit a square
    uint64_t discs[2] = {0, 0};
    for (int p = 0; p < 2; p++)
        for (int sq = 0; sq < SQUARES; sq += W) discs[p] |= Lanes::bits(obs + p * SQUARES + sq) << sq;
    // each square's 3 squares of its row, the one it's centred on: 6 bits as `stem_table` takes them;
    // and rows of none above and below the board
    int patterns[10][8] = {};
    for (int y = 0; y < 8; y++) {
        // the row's discs, from bit 1 (bit 0 and 9 off the board)
        const int mine = (int)((discs[0] >> (8 * y)) & 0xff) << 1, theirs = (int)((discs[1] >> (8 * y)) & 0xff) << 1;
        for (int x = 0; x < 8; x++) patterns[y + 1][x] = ((mine >> x) & 7) | ((theirs >> x) & 7) << 3;
    }
    const int C = c.cout;
    const float* scale = c.scale.data();
    const float* bias = c.bias.data();
    const int16_t* table = c.table.data();
    for (int sq = 0; sq < SQUARES; sq++) {
        const int y = sq / 8, x = sq % 8;
        const int16_t* rows[3];
        for (int dy = 0; dy < 3; dy++) rows[dy] = table + ((size_t)dy * 64 + patterns[y + dy][x]) * C;
        for (int i = 0; i < C; i += W)
            Lanes::store(out + sq * C + i, Lanes::fma(Lanes::widen(rows[0] + i, rows[1] + i, rows[2] + i),
                                                      Lanes::load(scale + i), Lanes::load(bias + i)));
    }
}

// the heads' float work, from the head convolution's output `h`, (64, 3 * head_channels)
inline void heads(const Net& n, float* h, float* logits, float* value) {
    const int H = n.head_channels, H3 = 3 * H;
    // each part normalized, of its own channels: the pooled part's and the value's pooled as they go
    // (out: a vector of outputs, as `linear` writes them, though it takes 1)
    alignas(64) float pooled[256], bias[128], hidden[1024], out[16];
    normalize_pool(n.pool_norm, h + H, H3, H, pooled);
    linear(n.pool_linear, pooled, bias);
    NET_STAGE("head policy pool");
#if NET_ISA != 3
    policy_logits(n.policy_norm, h, H3, H, bias, n.square_w.data(), logits);
#else
    normalize(n.policy_norm, h, H3, H, false);
    for (int sq = 0; sq < SQUARES; sq++) {
        float* v = h + sq * H3;
        float logit = 0;
        for (int i = 0; i < H; i++) logit += std::max(v[i] + bias[i], 0.0f) * n.square_w[i];
        logits[sq] = logit;
    }
#endif
    NET_STAGE("head square logits");
    linear(n.pass_linear, pooled, out);
    logits[SQUARES] = out[0];
    normalize_pool(n.value_norm, h + 2 * H, H3, H, pooled);
    linear(n.value_hidden, pooled, hidden);
    relu(hidden, n.value_hidden.out);
    linear(n.value_out, hidden, out);
    *value = std::tanh(out[0]);
    NET_STAGE("head pass and value");
}

// One position: `obs`, (2, 8, 8) channels-first, the mover's discs then the opponent's, to (65)
// policy logits and a value.
inline void forward(const Net& n, Scratch& s, const float* obs, float* logits, float* value) {
    const int C = n.channels;
    float* x = s.x.data();
    float* y = s.y.data();
    Activation* padded = s.padded.data();
    stem(n.stem, obs, x);
    NET_STAGE("stem");
    // what takes the trunk's output next: the next block's first convolution, or the heads'
    auto next = [&](size_t i) -> const Conv& { return i < n.blocks.size() ? n.blocks[i].conv1 : n.head; };
    if (!n.pre) normalize_quantize(n.stem_norm, x, C, next(0), padded, x);
    NET_STAGE("stem norm");
    for (size_t i = 0; i < n.blocks.size(); i++) {
        const Block& b = n.blocks[i];
        // the post layout's x is quantized for conv1 already
        if (n.pre) normalize_quantize(b.norm1, x, C, b.conv1, padded);
        NET_STAGE("norm1 quantize");
        convolve(b.conv1, s, y, C);
        NET_STAGE("conv1");
        if (b.pool_channels) {
            pool_bias_quantize(n.pre ? b.norm2 : b.norm1, b.pool, y, C, C, n.pre, b.conv2, padded);
        } else {
            normalize_quantize(n.pre ? b.norm2 : b.norm1, y, C, b.conv2, padded);
        }
        NET_STAGE("norm2 pool quantize");
        if (n.pre) {
            convolve(b.conv2, s, x, C, true);
            NET_STAGE("conv2");
        } else {
            convolve(b.conv2, s, y, C);
            NET_STAGE("conv2");
            residual_quantize(b.norm2, y, x, C, next(i + 1), padded);
            NET_STAGE("residual quantize");
        }
    }
    if (n.pre) normalize_quantize(n.final_norm, x, C, n.head, padded);
    NET_STAGE("final norm quantize");

    // the heads: the three 1x1 convolutions at once, then in float
    float* h = s.h.data();
    convolve(n.head, s, h, 3 * n.head_channels);
    NET_STAGE("head conv");
    heads(n, h, logits, value);
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
// The stem's sums, for `stem`, from its weights `q`, (cout, 4, 3, 3), the last 2 input channels padding:
// for each row of the 3 rows of 3 squares a square's sums take in (above, its own, below) and each
// pattern of discs on that row's squares (6 bits: the mover's from the left, then the opponent's), the
// sum of those discs' weights, (3, 64, cout), within 16 bits (and 3 rows' too: 18 weights of at most 127).
inline void stem_table(Conv& c, const int8_t* q) {
    c.table.assign((size_t)3 * 64 * c.cout, 0);
    for (int dy = 0; dy < 3; dy++)
        for (int pattern = 0; pattern < 64; pattern++)
            for (int o = 0; o < c.cout; o++) {
                int sum = 0;
                for (int k = 0; k < 6; k++)
                    if (pattern >> k & 1) sum += q[(((size_t)o * c.cin + k / 3) * 3 + dy) * 3 + k % 3];
                c.table[((size_t)dy * 64 + pattern) * c.cout + o] = (int16_t)sum;
            }
}

inline void pack(Conv& c, const int8_t* q) {
    const int c4n = c.cin / 4, taps = c.k * c.k;
#if NET_ISA == 3 && defined(__ARM_FEATURE_MATMUL_INT8)
    // for i8mm, [cout / 8][k * k][cin / 8, rounded up][4 pairs of outputs][2 outputs][8 inputs]
    const int e = (c.cin + 7) / 8;
    c.packed.assign((size_t)c.cout * e * 8 * taps, 0);
    for (int g = 0; g < c.cout / 8; g++)
        for (int t = 0; t < taps; t++)
            for (int k = 0; k < e; k++)
                for (int o = 0; o < 8; o++)
                    for (int ch = 0; ch < 8; ch++)
                        if (8 * k + ch < c.cin)
                            c.packed[(((size_t)g * taps + t) * e + k) * 64 + o * 8 + ch] =
                                q[((size_t)(8 * g + o) * c.cin + 8 * k + ch) * taps + t];
#elif NET_ISA == 3
    c.packed.assign((size_t)c.cout * c.cin * taps, 0);
    size_t at8 = 0;
    for (int g = 0; g < c.cout / 8; g++)
        for (int t = 0; t < taps; t++)
            for (int c4 = 0; c4 < c4n; c4++)
                for (int o = 0; o < 8; o++)
                    for (int k = 0; k < 4; k++) c.packed[at8++] = q[((size_t)(8 * g + o) * c.cin + 4 * c4 + k) * taps + t];
#endif
#if NET_ISA == 3
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
