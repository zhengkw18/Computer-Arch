import jittor as jt
from jittor import init, Module
import numpy as np

header = """
                #include <ops/binary_op_defs.h>
                #include <misc/cuda_limits.h>
                #include <stdio.h>

                #define loop9__(j,k,max_v,cache,out0_type) max_v = maximum(out0_type, max_v, cache[j*3+k])
                #define loop9_(j,max_v,cache,out0_type) loop9__(j,0,max_v,cache,out0_type); loop9__(j,1,max_v,cache,out0_type); loop9__(j,2,max_v,cache,out0_type)
                #define loop9(max_v,cache,out0_type) loop9_(0,max_v,cache,out0_type); loop9_(1,max_v,cache,out0_type); loop9_(2,max_v,cache,out0_type)

                #define loop8___(i,j,k,max_v,cache,out0_type) max_v = maximum(out0_type, max_v, cache[4*i+2*j+k])
                #define loop8__(i,j,max_v,cache,out0_type) loop8___(i,j,0,max_v,cache,out0_type); loop8___(i,j,1,max_v,cache,out0_type)
                #define loop8_(i,max_v,cache,out0_type) loop8__(i,0,max_v,cache,out0_type); loop8__(i,1,max_v,cache,out0_type)
                #define loop8(max_v,cache,out0_type) loop8_(0,max_v,cache,out0_type); loop8_(1,max_v,cache,out0_type)
            """


class Pool3D(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None
        assert op == "maximum"
        self.kernel_size = kernel_size
        self.op = op
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad and padding != 0

    def execute(self, x):
        N, C, T, H, W = x.shape
        if self.ceil_mode == False:
            t = (T + self.padding * 2 - self.kernel_size) // self.stride + 1
            h = (H + self.padding * 2 - self.kernel_size) // self.stride + 1
            w = (W + self.padding * 2 - self.kernel_size) // self.stride + 1
        else:
            t = (T + self.padding * 2 - self.kernel_size + self.stride - 1) // self.stride + 1
            h = (H + self.padding * 2 - self.kernel_size + self.stride - 1) // self.stride + 1
            w = (W + self.padding * 2 - self.kernel_size + self.stride - 1) // self.stride + 1
        # x = jt.reshape(x, (-1, T, H, W))
        dim1, dim2 = min(32, h), min(32, w)
        shared_sz = ((dim1 - 1) * self.stride + self.kernel_size) * ((dim2 - 1) * self.stride + self.kernel_size) * self.kernel_size
        # if self.padding == 0 and shared_sz <= 12200:
        #     out = jt.code(
        #         [N * C, t, h, w],
        #         x.dtype,
        #         [x],
        #         cuda_header=header,
        #         cuda_src=f"""
        #         __host__ __device__ __forceinline__ int CeilDiv(int a, int b) {{
        #             return (a + b - 1) / b;
        #         }}
        #         __global__ static void kernel1(
        #             in0_type* __restrict__ in0_p,
        #             index_t in0_shape0, index_t in0_shape1, index_t in0_shape2, index_t in0_shape3,
        #             out0_type* __restrict__ out0_p,
        #             index_t out0_shape0, index_t out0_shape1, index_t out0_shape2, index_t out0_shape3
        #         ) {{
        #             int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
        #             int oRow = blockIdx.y * blockDim.y + threadIdx.y;
        #             int oFrame = blockIdx.z % out0_shape1;
        #             int slice = blockIdx.z / out0_shape1;
        #             if (oRow < out0_shape2 && oColumn < out0_shape3){{
        #                 int tStart = oFrame  * {self.stride} - {self.padding};
        #                 int hStart = oRow    * {self.stride} - {self.padding};
        #                 int wStart = oColumn * {self.stride} - {self.padding};
        #                 int iStart = 0;
        #                 int jStart = threadIdx.y * {self.stride};
        #                 int kStart = threadIdx.x * {self.stride};
        #                 int iEnd = iStart + {self.kernel_size};
        #                 int jEnd = jStart + {self.kernel_size};
        #                 int kEnd = kStart + {self.kernel_size};
        #                 auto in0_stride2 = in0_stride3 * in0_shape3; auto in0_stride1 = in0_stride2 * in0_shape2; auto in0_stride0 = in0_stride1 * in0_shape1;
        #                 auto out0_stride2 = out0_stride3 * out0_shape3; auto out0_stride1 = out0_stride2 * out0_shape2; auto out0_stride0 = out0_stride1 * out0_shape1;

        #                 auto max_v = init_maximum(out0_type);
        #                 in0_p += slice * in0_stride0;

        #                 __shared__ in0_type buffer[12200];
        #                 auto stride3 = 1; auto stride2 = stride3 * ((blockDim.x - 1) * {self.stride} + {self.kernel_size}); auto stride1 = stride2 * ((blockDim.y - 1) * {self.stride} + {self.kernel_size});
        #                 int jMax = (threadIdx.y == blockDim.y - 1 || oRow == out0_shape2 - 1) ? {self.kernel_size} : {self.stride};
        #                 int kMax = (threadIdx.x == blockDim.x - 1 || oColumn == out0_shape3 - 1) ? {self.kernel_size} : {self.stride};
        #                 for(int i = 0, i1 = iStart * stride1, i2 = tStart * in0_stride1; i < {self.kernel_size}; i++, i1 += stride1, i2 += in0_stride1)
        #                 for(int j = 0, j1 = jStart * stride2, j2 = hStart * in0_stride2; j < jMax; j++, j1 += stride2, j2 += in0_stride2)
        #                 for(int k = 0, k1 = kStart * stride3, k2 = wStart * in0_stride3; k < kMax; k++, k1 += stride3, k2 += in0_stride3)
        #                 {{
        #                     buffer[i1 + j1 + k1] = in0_p[i2 + j2 + k2];
        #                 }}
        #                 __syncthreads();
        #                 if({self.kernel_size} == 2)
        #                 {{
        #                     in0_type cache[8];
        #                     int t0 = iStart, t1 = iStart + 1, h0 = jStart, h1 = jStart + 1, w0 = kStart, w1 = kStart + 1;
        #                     int t0_s = t0 * stride1, t1_s = t1 * stride1;
        #                     int h0_s = h0 * stride2, h1_s = h1 * stride2;
        #                     int w0_s = w0 * stride3, w1_s = w1 * stride3;
        #                     cache[0] = buffer[t0_s + h0_s + w0_s];
        #                     cache[1] = buffer[t0_s + h0_s + w1_s];
        #                     cache[2] = buffer[t0_s + h1_s + w0_s];
        #                     cache[3] = buffer[t0_s + h1_s + w1_s];
        #                     cache[4] = buffer[t1_s + h0_s + w0_s];
        #                     cache[5] = buffer[t1_s + h0_s + w1_s];
        #                     cache[6] = buffer[t1_s + h1_s + w0_s];
        #                     cache[7] = buffer[t1_s + h1_s + w1_s];
        #                     loop8(max_v,cache,out0_type);
        #                 }}
        #                 else if({self.kernel_size} == 3)
        #                 {{
        #                     in0_type cache[9];
        #                     int h0 = jStart, h1 = jStart + 1, h2 = jStart + 2, w0 = kStart, w1 = kStart + 1, w2 = kStart + 2;
        #                     int h0_s = h0 * stride2, h1_s = h1 * stride2, h2_s = h2 * stride2;
        #                     int w0_s = w0 * stride3, w1_s = w1 * stride3, w2_s = w2 * stride3;
        #                     for (int t = iStart * stride1; t < iEnd * stride1; t += stride1)
        #                     {{
        #                         cache[0] = buffer[t + h0_s + w0_s];
        #                         cache[1] = buffer[t + h0_s + w1_s];
        #                         cache[2] = buffer[t + h0_s + w2_s];
        #                         cache[3] = buffer[t + h1_s + w0_s];
        #                         cache[4] = buffer[t + h1_s + w1_s];
        #                         cache[5] = buffer[t + h1_s + w2_s];
        #                         cache[6] = buffer[t + h2_s + w0_s];
        #                         cache[7] = buffer[t + h2_s + w1_s];
        #                         cache[8] = buffer[t + h2_s + w2_s];
        #                         loop9(max_v,cache,out0_type);
        #                     }}
        #                 }}
        #                 else if({self.kernel_size} == 8)
        #                 {{
        #                     in0_type cache[8];
        #                     for (int t = iStart * stride1; t < iEnd * stride1; t += stride1)
        #                     for (int h = jStart * stride2; h < jEnd * stride2; h += stride2)
        #                     {{
        #                         int w = kStart * stride3;
        #                         in0_type* _in0_p = &buffer[t + h + w];
        #                         cache[0] = _in0_p[0];
        #                         cache[1] = _in0_p[1];
        #                         cache[2] = _in0_p[2];
        #                         cache[3] = _in0_p[3];
        #                         cache[4] = _in0_p[4];
        #                         cache[5] = _in0_p[5];
        #                         cache[6] = _in0_p[6];
        #                         cache[7] = _in0_p[7];
        #                         loop8(max_v,cache,out0_type);
        #                     }}
        #                 }}
        #                 else
        #                 {{
        #                     for (int t = iStart * stride1; t < iEnd * stride1; t += stride1)
        #                     for (int h = jStart * stride2; h < jEnd * stride2; h += stride2)
        #                     for (int w = kStart * stride3; w < kEnd * stride3; w += stride3)
        #                     {{
        #                         max_v = maximum(out0_type, max_v, buffer[t + h + w]);
        #                     }}
        #                 }}
        #                 out0_p[slice * out0_stride0 + oFrame * out0_stride1 + oRow * out0_stride2 + oColumn] = max_v;
        #             }}
        #         }}
        #         int tx = min(32, out0_shape3);
        #         int ty = min(32, out0_shape2);
        #         dim3 block(tx, ty);
        #         int bx = CeilDiv(out0_shape3, static_cast<int>(block.x));
        #         int by = CeilDiv(out0_shape2, static_cast<int>(block.y));
        #         int bz = in0_shape0 * out0_shape1;
        #         dim3 grid(bx, by, bz);
        #         kernel1<<<grid, block>>>(
        #                 in0_p,
        #                 in0_shape0, in0_shape1, in0_shape2, in0_shape3,
        #                 out0_p,
        #                 out0_shape0, out0_shape1, out0_shape2, out0_shape3
        #         );
        #     """,
        #     )
        if self.padding == 0:
            out = jt.code(
                [N, C, t, h, w],
                x.dtype,
                [x],
                cuda_header=header,
                cuda_src=f"""
                __host__ __device__ __forceinline__ int CeilDiv(int a, int b) {{
                    return (a + b - 1) / b;
                }}
                __global__ static void kernel1(
                    in0_type* __restrict__ in0_p, 
                    index_t in0_shape0, index_t in0_shape1, index_t in0_shape2, index_t in0_shape3,
                    out0_type* __restrict__ out0_p, 
                    index_t out0_shape0, index_t out0_shape1, index_t out0_shape2, index_t out0_shape3
                ) {{
                    int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
                    int oRow = blockIdx.y * blockDim.y + threadIdx.y;
                    int oNum = blockIdx.z * blockDim.z + threadIdx.z;
                    int oFrame = oNum % out0_shape1;
                    int slice = oNum / out0_shape1;
                    if (oRow < out0_shape2 && oColumn < out0_shape3 && slice < out0_shape0){{
                        int tStart = oFrame  * {self.stride} - {self.padding};
                        int hStart = oRow    * {self.stride} - {self.padding};
                        int wStart = oColumn * {self.stride} - {self.padding};
                        int tEnd = tStart + {self.kernel_size};
                        int hEnd = hStart + {self.kernel_size};
                        int wEnd = wStart + {self.kernel_size};
                        auto in0_stride3 = 1; auto out0_stride3 = 1;
                        auto in0_stride2 = in0_stride3 * in0_shape3; auto in0_stride1 = in0_stride2 * in0_shape2; auto in0_stride0 = in0_stride1 * in0_shape1; 
                        auto out0_stride2 = out0_stride3 * out0_shape3; auto out0_stride1 = out0_stride2 * out0_shape2; auto out0_stride0 = out0_stride1 * out0_shape1; 

                        auto max_v = init_maximum(out0_type);
                        in0_p += slice * in0_stride0;
                        if({self.kernel_size} == 2)
                        {{
                            in0_type cache[8];
                            int t0_s = tStart * in0_stride1, t1_s = t0_s + in0_stride1;
                            int h0_s = hStart * in0_stride2, h1_s = h0_s + in0_stride2;
                            int w0_s = wStart * in0_stride3, w1_s = w0_s + in0_stride3;
                            cache[0] = in0_p[t0_s + h0_s + w0_s];
                            cache[1] = in0_p[t0_s + h0_s + w1_s];
                            cache[2] = in0_p[t0_s + h1_s + w0_s];
                            cache[3] = in0_p[t0_s + h1_s + w1_s];
                            cache[4] = in0_p[t1_s + h0_s + w0_s];
                            cache[5] = in0_p[t1_s + h0_s + w1_s];
                            cache[6] = in0_p[t1_s + h1_s + w0_s];
                            cache[7] = in0_p[t1_s + h1_s + w1_s];
                            loop8(max_v,cache,out0_type);
                        }}
                        else if({self.kernel_size} == 3)
                        {{
                            in0_type cache[9];
                            int h0_s = hStart * in0_stride2, h1_s = h0_s + in0_stride2, h2_s = h1_s + in0_stride2;
                            int w0_s = wStart * in0_stride3, w1_s = w0_s + in0_stride3, w2_s = w1_s + in0_stride3;
                            #pragma unroll 3
                            for (int t = tStart * in0_stride1; t < tEnd * in0_stride1; t += in0_stride1)
                            {{
                                cache[0] = in0_p[t + h0_s + w0_s];
                                cache[1] = in0_p[t + h0_s + w1_s];
                                cache[2] = in0_p[t + h0_s + w2_s];
                                cache[3] = in0_p[t + h1_s + w0_s];
                                cache[4] = in0_p[t + h1_s + w1_s];
                                cache[5] = in0_p[t + h1_s + w2_s];
                                cache[6] = in0_p[t + h2_s + w0_s];
                                cache[7] = in0_p[t + h2_s + w1_s];
                                cache[8] = in0_p[t + h2_s + w2_s];
                                loop9(max_v,cache,out0_type);
                            }}
                        }}
                        else if({self.kernel_size} == 8)
                        {{
                            in0_type cache[8];
                            in0_type* _in0_p_ = &in0_p[wStart * in0_stride3];
                            for (int t = tStart * in0_stride1; t < tEnd * in0_stride1; t += in0_stride1){{
                            #pragma unroll 8
                            for (int h = hStart * in0_stride2; h < hEnd * in0_stride2; h += in0_stride2)
                            {{
                                in0_type* _in0_p = &_in0_p_[t + h];
                                cache[0] = _in0_p[0];
                                cache[1] = _in0_p[1];
                                cache[2] = _in0_p[2];
                                cache[3] = _in0_p[3];
                                cache[4] = _in0_p[4];
                                cache[5] = _in0_p[5];
                                cache[6] = _in0_p[6];
                                cache[7] = _in0_p[7];
                                loop8(max_v,cache,out0_type);
                            }}
                            }}
                        }}
                        else
                        {{
                            for (int t = tStart * in0_stride1; t < tEnd * in0_stride1; t += in0_stride1)
                            for (int h = hStart * in0_stride2; h < hEnd * in0_stride2; h += in0_stride2)
                            for (int w = wStart * in0_stride3; w < wEnd * in0_stride3; w += in0_stride3)
                            {{
                                max_v = maximum(out0_type, max_v, in0_p[t + h + w]);
                            }}
                        }}
                        out0_p[slice * out0_stride0 + oFrame * out0_stride1 + oRow * out0_stride2 + oColumn] = max_v;
                    }}
                }}
                int tx = min(1024, out0_shape4);
                int ty = min(1024 / tx, out0_shape3);
                int tz = 1024 / (tx * ty);
                tz = 1;
                dim3 block(tx, ty, tz);
                int bx = CeilDiv(out0_shape4, tx);
                int by = CeilDiv(out0_shape3, ty);
                int bz = CeilDiv(in0_shape0 * out0_shape1 * out0_shape2, tz);
                dim3 grid(bx, by, bz);
                kernel1<<<grid, block>>>(
                        in0_p, 
                        in0_shape0* in0_shape1, in0_shape2, in0_shape3,in0_shape4,
                        out0_p, 
                        out0_shape0* out0_shape1, out0_shape2, out0_shape3,out0_shape4
                );
            """,
            )
        # else:
        #     out = jt.code(
        #         [N * C, t, h, w],
        #         x.dtype,
        #         [x],
        #         cuda_header=header,
        #         cuda_src=f"""
        #         __host__ __device__ __forceinline__ int CeilDiv(int a, int b) {{
        #             return (a + b - 1) / b;
        #         }}
        #         __global__ static void kernel1(
        #             in0_type* __restrict__ in0_p,
        #             index_t in0_shape0, index_t in0_shape1, index_t in0_shape2, index_t in0_shape3,
        #             out0_type* __restrict__ out0_p,
        #             index_t out0_shape0, index_t out0_shape1, index_t out0_shape2, index_t out0_shape3
        #         ) {{
        #             int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
        #             int oRow = blockIdx.y * blockDim.y + threadIdx.y;
        #             int oFrame = blockIdx.z % out0_shape1;
        #             int slice = blockIdx.z / out0_shape1;
        #             if (oRow < out0_shape2 && oColumn < out0_shape3){{
        #                 int tStart = oFrame  * {self.stride} - {self.padding};
        #                 int hStart = oRow    * {self.stride} - {self.padding};
        #                 int wStart = oColumn * {self.stride} - {self.padding};
        #                 int tEnd = min(tStart + {self.kernel_size}, in0_shape1);
        #                 int hEnd = min(hStart + {self.kernel_size}, in0_shape2);
        #                 int wEnd = min(wStart + {self.kernel_size}, in0_shape3);
        #                 tStart = max(tStart, 0);
        #                 hStart = max(hStart, 0);
        #                 wStart = max(wStart, 0);
        #                 auto in0_stride2 = in0_stride3 * in0_shape3; auto in0_stride1 = in0_stride2 * in0_shape2; auto in0_stride0 = in0_stride1 * in0_shape1;
        #                 auto out0_stride2 = out0_stride3 * out0_shape3; auto out0_stride1 = out0_stride2 * out0_shape2; auto out0_stride0 = out0_stride1 * out0_shape1;
        #                 auto max_v = init_maximum(out0_type);
        #                 in0_p += slice * in0_stride0;
        #                 if({self.kernel_size} == 2)
        #                 {{
        #                     in0_type cache[8];
        #                     int t0 = tStart, t1 = tStart + 1, h0 = hStart, h1 = hStart + 1, w0 = wStart, w1 = wStart + 1;
        #                     int t0_s = t0 * in0_stride1, t1_s = t1 * in0_stride1;
        #                     int h0_s = h0 * in0_stride2, h1_s = h1 * in0_stride2;
        #                     int w0_s = w0 * in0_stride3, w1_s = w1 * in0_stride3;
        #                     cache[0] = (t0 < tEnd && h0 < hEnd && w0 < wEnd) ? in0_p[t0_s + h0_s + w0_s] : init_maximum(out0_type);
        #                     cache[1] = (t0 < tEnd && h0 < hEnd && w1 < wEnd) ? in0_p[t0_s + h0_s + w1_s] : init_maximum(out0_type);
        #                     cache[2] = (t0 < tEnd && h1 < hEnd && w0 < wEnd) ? in0_p[t0_s + h1_s + w0_s] : init_maximum(out0_type);
        #                     cache[3] = (t0 < tEnd && h1 < hEnd && w1 < wEnd) ? in0_p[t0_s + h1_s + w1_s] : init_maximum(out0_type);
        #                     cache[4] = (t1 < tEnd && h0 < hEnd && w0 < wEnd) ? in0_p[t1_s + h0_s + w0_s] : init_maximum(out0_type);
        #                     cache[5] = (t1 < tEnd && h0 < hEnd && w1 < wEnd) ? in0_p[t1_s + h0_s + w1_s] : init_maximum(out0_type);
        #                     cache[6] = (t1 < tEnd && h1 < hEnd && w0 < wEnd) ? in0_p[t1_s + h1_s + w0_s] : init_maximum(out0_type);
        #                     cache[7] = (t1 < tEnd && h1 < hEnd && w1 < wEnd) ? in0_p[t1_s + h1_s + w1_s] : init_maximum(out0_type);
        #                     loop8(max_v,cache,out0_type);
        #                 }}
        #                 else if({self.kernel_size} == 3)
        #                 {{
        #                     in0_type cache[9];
        #                     int h0 = hStart, h1 = hStart + 1, h2 = hStart + 2, w0 = wStart, w1 = wStart + 1, w2 = wStart + 2;
        #                     int h0_s = h0 * in0_stride2, h1_s = h1 * in0_stride2, h2_s = h2 * in0_stride2;
        #                     int w0_s = w0 * in0_stride3, w1_s = w1 * in0_stride3, w2_s = w2 * in0_stride3;
        #                     for (int t = tStart * in0_stride1; t < tEnd * in0_stride1; t += in0_stride1)
        #                     {{
        #                         cache[0] = (h0 < hEnd && w0 < wEnd) ? in0_p[t + h0_s + w0_s] : init_maximum(out0_type);
        #                         cache[1] = (h0 < hEnd && w1 < wEnd) ? in0_p[t + h0_s + w1_s] : init_maximum(out0_type);
        #                         cache[2] = (h0 < hEnd && w2 < wEnd) ? in0_p[t + h0_s + w2_s] : init_maximum(out0_type);
        #                         cache[3] = (h1 < hEnd && w0 < wEnd) ? in0_p[t + h1_s + w0_s] : init_maximum(out0_type);
        #                         cache[4] = (h1 < hEnd && w1 < wEnd) ? in0_p[t + h1_s + w1_s] : init_maximum(out0_type);
        #                         cache[5] = (h1 < hEnd && w2 < wEnd) ? in0_p[t + h1_s + w2_s] : init_maximum(out0_type);
        #                         cache[6] = (h2 < hEnd && w0 < wEnd) ? in0_p[t + h2_s + w0_s] : init_maximum(out0_type);
        #                         cache[7] = (h2 < hEnd && w1 < wEnd) ? in0_p[t + h2_s + w1_s] : init_maximum(out0_type);
        #                         cache[8] = (h2 < hEnd && w2 < wEnd) ? in0_p[t + h2_s + w2_s] : init_maximum(out0_type);
        #                         loop9(max_v,cache,out0_type);
        #                     }}
        #                 }}
        #                 else if({self.kernel_size} == 8)
        #                 {{
        #                     in0_type cache[8];
        #                     for (int t = tStart * in0_stride1; t < tEnd * in0_stride1; t += in0_stride1)
        #                     for (int h = hStart * in0_stride2; h < hEnd * in0_stride2; h += in0_stride2)
        #                     {{
        #                         int w = wStart * in0_stride3;
        #                         in0_type* _in0_p = &in0_p[t + h + w];
        #                         cache[0] = (w + 0 < wEnd) ? _in0_p[0] : init_maximum(out0_type);
        #                         cache[1] = (w + 1 < wEnd) ? _in0_p[1] : init_maximum(out0_type);
        #                         cache[2] = (w + 2 < wEnd) ? _in0_p[2] : init_maximum(out0_type);
        #                         cache[3] = (w + 3 < wEnd) ? _in0_p[3] : init_maximum(out0_type);
        #                         cache[4] = (w + 4 < wEnd) ? _in0_p[4] : init_maximum(out0_type);
        #                         cache[5] = (w + 5 < wEnd) ? _in0_p[5] : init_maximum(out0_type);
        #                         cache[6] = (w + 6 < wEnd) ? _in0_p[6] : init_maximum(out0_type);
        #                         cache[7] = (w + 7 < wEnd) ? _in0_p[7] : init_maximum(out0_type);
        #                         loop8(max_v,cache,out0_type);
        #                     }}
        #                 }}
        #                 else
        #                 {{
        #                     for (int t = tStart * in0_stride1; t < tEnd * in0_stride1; t += in0_stride1)
        #                     for (int h = hStart * in0_stride2; h < hEnd * in0_stride2; h += in0_stride2)
        #                     for (int w = wStart * in0_stride3; w < wEnd * in0_stride3; w += in0_stride3)
        #                     {{
        #                         max_v = maximum(out0_type, max_v, in0_p[t + h + w]);
        #                     }}
        #                 }}
        #                 out0_p[slice * out0_stride0 + oFrame * out0_stride1 + oRow * out0_stride2 + oColumn] = max_v;
        #             }}
        #         }}
        #         int tx = min(32, out0_shape3);
        #         int ty = min(32, out0_shape2);
        #         dim3 block(tx, ty);
        #         int bx = CeilDiv(out0_shape3, static_cast<int>(block.x));
        #         int by = CeilDiv(out0_shape2, static_cast<int>(block.y));
        #         int bz = in0_shape0 * out0_shape1;
        #         dim3 grid(bx, by, bz);
        #         kernel1<<<grid, block>>>(
        #                 in0_p,
        #                 in0_shape0, in0_shape1, in0_shape2, in0_shape3,
        #                 out0_p,
        #                 out0_shape0, out0_shape1, out0_shape2, out0_shape3
        #         );
        #     """,
        #     )
        return out
        return jt.reshape(out, (N, C, t, h, w))
