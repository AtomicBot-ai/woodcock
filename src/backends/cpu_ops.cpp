#include "woodcock/ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef WOODCOCK_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace woodcock::ops {

namespace {
// C[M][N] = alpha * A[M][K] . B^T (B is [N][K]) + beta * C ; leading dims given
void gemm_nt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
             int ldc) {
#ifdef WOODCOCK_ACCELERATE
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) s += A[m * lda + k] * B[n * ldb + k];
      C[m * ldc + n] = alpha * s + (beta == 0.f ? 0.f : beta * C[m * ldc + n]);
    }
#endif
}

// C[M][N] = A[M][K] . B[K][N] + beta * C
void gemm_nn(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
#ifdef WOODCOCK_ACCELERATE
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.f, A, lda, B, ldb, beta, C, ldc);
#else
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) s += A[m * lda + k] * B[k * ldb + n];
      C[m * ldc + n] = s + (beta == 0.f ? 0.f : beta * C[m * ldc + n]);
    }
#endif
}
}  // namespace

void linear(const float* x, int L, int in, const Tensor& W, float* y, bool accumulate) {
  const int out = int(W.rows());
  gemm_nt(L, out, in, 1.f, x, in, W.data.data(), in, accumulate ? 1.f : 0.f, y, out);
}

void rmsnorm(const float* x, float* y, int rows, int d, const float* w, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + int64_t(r) * d;
    float* yr = y + int64_t(r) * d;
    double ss = 0.0;
    for (int i = 0; i < d; ++i) ss += double(xr[i]) * xr[i];
    const float inv = 1.f / std::sqrt(float(ss / d) + eps);
    for (int i = 0; i < d; ++i) yr[i] = xr[i] * inv * w[i];
  }
}

void rope(float* x, int L, int heads, int D, const int* positions, float base) {
  const int half = D / 2;
  std::vector<float> inv(half);
  for (int i = 0; i < half; ++i) inv[i] = std::pow(base, -2.f * i / D);
  for (int l = 0; l < L; ++l) {
    for (int i = 0; i < half; ++i) {
      const float a = float(positions[l]) * inv[i];
      const float c = std::cos(a), s = std::sin(a);
      for (int h = 0; h < heads; ++h) {
        float* v = x + (int64_t(l) * heads + h) * D;
        const float x1 = v[i], x2 = v[i + half];
        v[i] = x1 * c - x2 * s;
        v[i + half] = x2 * c + x1 * s;
      }
    }
  }
}

void silu_mul(float* gate, const float* up, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    const float g = gate[i];
    gate[i] = g / (1.f + std::exp(-g)) * up[i];
  }
}

void softmax(float* x, int n) {
  float m = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i) m = std::max(m, x[i]);
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    x[i] = std::isinf(x[i]) && x[i] < 0 ? 0.f : std::exp(x[i] - m);
    s += x[i];
  }
  const float inv = float(1.0 / s);
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

void attention(const float* q, int L, int H, int Hk, int D, const float* K, const float* V, int T, const uint8_t* mask,
               float* out, float scale) {
  const int rep = H / Hk;
  std::vector<float> S(static_cast<size_t>(L) * T);
  const float ninf = -std::numeric_limits<float>::infinity();
  for (int h = 0; h < H; ++h) {
    const int kh = h / rep;
    gemm_nt(L, T, D, scale, q + h * D, H * D, K + kh * D, Hk * D, 0.f, S.data(), T);
    for (int l = 0; l < L; ++l) {
      float* row = S.data() + int64_t(l) * T;
      if (mask)
        for (int t = 0; t < T; ++t)
          if (!mask[int64_t(l) * T + t]) row[t] = ninf;
      softmax(row, T);
    }
    gemm_nn(L, D, T, S.data(), T, V + kh * D, Hk * D, 0.f, out + h * D, H * D);
  }
}

}  // namespace woodcock::ops
