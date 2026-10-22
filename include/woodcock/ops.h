// CPU compute kernels (fp32). Matrix products use Apple Accelerate (BLAS) when
// available, a portable loop otherwise. Activations are row-major [rows][cols].
#pragma once

#include <cstdint>

#include "woodcock/weights.h"

namespace woodcock::ops {

// y[L][out] (+)= x[L][in] . W[out][in]^T
void linear(const float* x, int L, int in, const Tensor& W, float* y, bool accumulate = false);
// y = x * rsqrt(mean(x^2) + eps) * w, per row of length d; x and y may alias
void rmsnorm(const float* x, float* y, int rows, int d, const float* w, float eps);
// rotate-half RoPE (Qwen3 / MLX traditional=false) on x[L][heads][D] at given positions
void rope(float* x, int L, int heads, int D, const int* positions, float base);
// gate = silu(gate) * up
void silu_mul(float* gate, const float* up, int64_t n);
// q[L][H][D]; K, V: [T][Hk][D]; mask[L][T] (nullptr = all visible); out[L][H][D]
void attention(const float* q, int L, int H, int Hk, int D, const float* K, const float* V, int T,
               const uint8_t* mask, float* out, float scale);
void softmax(float* x, int n);

}  // namespace woodcock::ops
