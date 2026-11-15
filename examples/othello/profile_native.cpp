// Times the native network (`cpu_net.h`) at batch 1 on one thread, whole and stage by stage, on a
// network of random weights of the given shape (its time doesn't depend on them).
//
//     g++ -std=c++17 -O3 -march=x86-64-v4 -mavx512vnni -mtune=generic -fopenmp-simd profile_native.cpp
//         -o /tmp/profile_native  (or another level's flags, `native_target.LEVELS`)
//     taskset -c 9 /tmp/profile_native 4 64 pre channel
//
// prints one JSON line: the shape, ns a forward pass without the marks (the median of rounds), and ns
// a pass in each stage with them (each mark costs a few ns of its own).
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace profile {
// a fine clock: the cycle counter on x86, elsewhere the steady clock's ns
#if defined(__x86_64__)
inline unsigned long long now() { return __rdtsc(); }
#else
inline unsigned long long now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#endif
inline std::vector<std::string> names;
inline std::vector<unsigned long long> ticks;
inline unsigned long long last = 0;
inline bool on = false;
inline int id(const char* name) {
    for (size_t i = 0; i < names.size(); i++)
        if (names[i] == name) return (int)i;
    names.push_back(name);
    ticks.push_back(0);
    return (int)names.size() - 1;
}
inline void mark(int i) {
    if (!on) return;
    unsigned long long t = now();
    ticks[i] += t - last;
    last = t;
}
}  // namespace profile

#define NET_STAGE(name)                            \
    do {                                           \
        static const int stage_ = profile::id(name); \
        profile::mark(stage_);                     \
    } while (0)

#include "cpu_net.h"

using namespace net;

std::mt19937 rng(1);
float normal() { return std::normal_distribution<float>(0, 1)(rng); }

Conv make_conv(int cin, int cout, int k, int zero_point) {
    Conv c;
    c.cin = cin;
    c.cout = cout;
    c.k = k;
    std::vector<int8_t> q((size_t)cout * cin * k * k);
    for (auto& v : q) v = (int8_t)std::uniform_int_distribution<int>(-127, 127)(rng);
    if (cin == 4) stem_table(c, q.data());  // the stem
    else pack(c, q.data());
    for (int o = 0; o < cout; o++) {
        c.scale.push_back(1.0f / (40.0f * cin * k * k));
        c.bias.push_back(0.1f * normal());
    }
    for (int i = 0; i < cin; i++) c.inverse.push_back(40.0f);
    c.zero_point = zero_point;
    return c;
}

Norm make_norm(int type, int channels) {
    Norm n;
    n.type = type;
    n.channels = channels;
    if (type != NORM_NONE)
        for (int i = 0; i < channels; i++) {
            n.a.push_back(1 + 0.1f * normal());
            n.c.push_back(0.1f * normal());
        }
    return n;
}

Linear make_linear(int in, int out) {
    Linear l;
    l.in = in;
    l.out = out;
    for (int i = 0; i < in * out; i++) l.w.push_back(normal() / std::sqrt((float)in));
    for (int o = 0; o < out; o++) l.b.push_back(0.1f * normal());
    return l;
}

// as `core.networks.azresnet` builds a ResNet of pooled heads, pool_every=2
Net make_net(int blocks, int channels, bool pre, int norm, int pool_channels, int head_channels, int hidden) {
    Net n;
    n.channels = channels;
    n.pre = pre;
    n.head_channels = head_channels;
    // channel norms aren't folded; BatchNorm is, before (affine) or into the convolution after
    const int after = norm == NORM_CHANNEL ? NORM_CHANNEL : NORM_NONE;
    const int before = norm == NORM_CHANNEL ? NORM_CHANNEL : NORM_AFFINE;
    n.stem = make_conv(4, channels, 3, 0);
    n.stem_norm = make_norm(pre ? NORM_NONE : after, channels);
    for (int i = 0; i < blocks; i++) {
        Block b;
        b.conv1 = make_conv(channels, channels, 3, pre ? 0 : 0);
        b.conv2 = make_conv(channels, channels, 3, pre ? 40 : 0);
        b.norm1 = make_norm(pre ? before : after, channels);
        b.norm2 = make_norm(pre ? before : after, channels);
        if ((i + 1) % 2 == 0) {
            b.pool_channels = pool_channels;
            b.pool = make_linear(2 * pool_channels, channels - pool_channels);
        }
        n.blocks.push_back(b);
    }
    n.final_norm = make_norm(pre ? before : NORM_NONE, channels);
    n.head = make_conv(channels, 3 * head_channels, 1, pre ? 0 : 0);
    n.policy_norm = make_norm(after, head_channels);
    n.pool_norm = make_norm(after, head_channels);
    n.value_norm = make_norm(after, head_channels);
    n.pool_linear = make_linear(2 * head_channels, head_channels);
    n.pass_linear = make_linear(2 * head_channels, 1);
    n.value_hidden = make_linear(2 * head_channels, hidden);
    n.value_out = make_linear(hidden, 1);
    for (int i = 0; i < head_channels; i++) n.square_w.push_back(normal());
    return n;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: profile_native BLOCKS CHANNELS pre|post channel|batch [ROUNDS] [VALUE_HIDDEN]\n");
        return 2;
    }
    int blocks = std::atoi(argv[1]), channels = std::atoi(argv[2]);
    bool pre = std::string(argv[3]) == "pre";
    int norm = std::string(argv[4]) == "channel" ? NORM_CHANNEL : NORM_AFFINE;
    int rounds = argc > 5 ? std::atoi(argv[5]) : 15;
    int hidden = argc > 6 ? std::atoi(argv[6]) : 256;
    Net n = make_net(blocks, channels, pre, norm, channels >= 96 ? 32 : 16, 32, hidden);
    Scratch s(n);
    // positions of random discs, the same every run
    const int P = 1024;
    std::vector<float> obs((size_t)P * 128);
    for (int p = 0; p < P; p++)
        for (int sq = 0; sq < 64; sq++) {
            int r = std::uniform_int_distribution<int>(0, 2)(rng);
            obs[(size_t)p * 128 + sq] = r == 1;
            obs[(size_t)p * 128 + 64 + sq] = r == 2;
        }
    float logits[65], value, check = 0;
    // passes a round: about 20 ms of them
    int reps = std::max(200, (int)(2e6 / (1.0 * blocks * channels * channels / 64 + 2000)));
    for (int i = 0; i < reps; i++) forward(n, s, obs.data() + (size_t)(i % P) * 128, logits, &value);
    std::vector<double> times;
    for (int r = 0; r < rounds; r++) {
        auto t = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) {
            forward(n, s, obs.data() + (size_t)(i % P) * 128, logits, &value);
            check += value;
        }
        times.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t).count() / reps);
    }
    std::sort(times.begin(), times.end());
    // the marks, and the clock's ticks a ns
    auto t0 = std::chrono::steady_clock::now();
    unsigned long long c0 = profile::now();
    profile::on = true;
    const int marked = reps * 5;
    for (int i = 0; i < marked; i++) {
        profile::last = profile::now();
        forward(n, s, obs.data() + (size_t)(i % P) * 128, logits, &value);
    }
    double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    double per_ns = (profile::now() - c0) / ns;
    std::printf("{\"shape\": \"%dx%d\", \"pre\": %d, \"norm\": \"%s\", \"ns\": %.1f, \"min_ns\": %.1f, \"marked_ns\": %.1f, \"stages\": {",
                blocks, channels, (int)pre, argv[4], times[times.size() / 2], times[0], ns / marked);
    for (size_t i = 0; i < profile::names.size(); i++)
        std::printf("%s\"%s\": %.1f", i ? ", " : "", profile::names[i].c_str(), profile::ticks[i] / per_ns / marked);
    std::printf("}, \"check\": %g}\n", (double)check);
}
