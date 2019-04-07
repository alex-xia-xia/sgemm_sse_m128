# sgemm_sse_m128
we create a sgemm example with sse m128 after learning how-to-optimize-gemm(https://github.com/flame/how-to-optimize-gemm).

we donot use avx512, because avx512 is not supoortted with gcc4.8.

you must change the parameters (mr, nr, mc, nc, kc) to get best performance according to your architecture.
