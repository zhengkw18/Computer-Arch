import jittor as jt
from jittor import init, Module

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
        if self.kernel_size == 8 and (self.stride == 1 or (T == 8 and H == 8 or W == 8) or (C == 64 and T == 32 and H == 32 or W == 32)):
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
                        auto in0_stride3 = 1; auto out0_stride3 = 1;
                        auto in0_stride2 = in0_stride3 * in0_shape3; auto in0_stride1 = in0_stride2 * in0_shape2; auto in0_stride0 = in0_stride1 * in0_shape1; 
                        auto out0_stride2 = out0_stride3 * out0_shape3; auto out0_stride1 = out0_stride2 * out0_shape2; auto out0_stride0 = out0_stride1 * out0_shape1; 

                        auto max_v = init_maximum(out0_type);
                        in0_p += slice * in0_stride0;
                        if({self.kernel_size} == 2)
                        {{
                            #pragma unroll 2
                            for(int i = tStart; i < tStart + 2; i++){{
                            #pragma unroll 2
                            for(int j = hStart; j < hStart + 2; j++){{
                            #pragma unroll 2
                            for(int k = wStart; k < wStart + 2; k++){{
                                max_v = maximum(out0_type, max_v, in0_p[i * in0_stride1 + j * in0_stride2 + k]);
                            }}
                            }}
                            }}
                        }}
                        else if({self.kernel_size} == 3)
                        {{
                            int tEnd = tStart + {self.kernel_size};
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
                            int hEnd = hStart + {self.kernel_size};
                            int tEnd = tStart + {self.kernel_size};
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
                            int hEnd = hStart + {self.kernel_size};
                            int wEnd = wStart + {self.kernel_size};
                            int tEnd = tStart + {self.kernel_size};
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
                int tz = min(64, 1024 / tx / ty);
                dim3 block(tx, ty, tz);
                int bx = CeilDiv(out0_shape4, tx);
                int by = CeilDiv(out0_shape3, ty);
                int bz = CeilDiv(in0_shape0 * out0_shape1 * out0_shape2, tz);
                dim3 grid(bx, by, bz);
                // printf("%d %d %d %d %d %d\\n", block.x, block.y, block.z, bx, by, bz);
                kernel1<<<grid, block>>>(
                        in0_p, 
                        in0_shape0 * in0_shape1, in0_shape2, in0_shape3, in0_shape4,
                        out0_p, 
                        out0_shape0 * out0_shape1, out0_shape2, out0_shape3, out0_shape4
                );
                cudaError_t cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess)
                    printf("addKernel launch failed: %s\\n", cudaGetErrorString(cudaStatus));
            """,
            )
        else:
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
                    int oColumn = threadIdx.x;
                    int oRow = blockIdx.x * blockDim.y + threadIdx.y;
                    int oFrame = blockIdx.y * blockDim.z + threadIdx.z;
                    int slice = blockIdx.z;
                    if (oRow < out0_shape2 && oColumn < out0_shape3 && slice < out0_shape0 && oFrame < out0_shape1){{
                        int hStart = oRow    * {self.stride} - {self.padding};
                        int wStart = oColumn * {self.stride} - {self.padding};
                        int tStart = oFrame  * {self.stride} - {self.padding};
                        auto in0_stride3 = 1; auto out0_stride3 = 1;
                        auto in0_stride2 = in0_stride3 * in0_shape3; auto in0_stride1 = in0_stride2 * in0_shape2; auto in0_stride0 = in0_stride1 * in0_shape1; 
                        auto out0_stride2 = out0_stride3 * out0_shape3; auto out0_stride1 = out0_stride2 * out0_shape2; auto out0_stride0 = out0_stride1 * out0_shape1; 
                        in0_p += slice * in0_stride0;
                        auto max_v = init_maximum(out0_type);
                        
                        if({self.kernel_size} == 2)
                        {{
                            #pragma unroll 2
                            for(int i = tStart; i < tStart + 2; i++){{
                            #pragma unroll 2
                            for(int j = hStart; j < hStart + 2; j++){{
                            #pragma unroll 2
                            for(int k = wStart; k < wStart + 2; k++){{
                                max_v = maximum(out0_type, max_v, in0_p[i * in0_stride1 + j * in0_stride2 + k]);
                            }}
                            }}
                            }}
                        }}
                        else if({self.kernel_size} == 3)
                        {{
                            int tEnd = tStart + {self.kernel_size};
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
                            int hEnd = hStart + {self.kernel_size};
                            int tEnd = tStart + {self.kernel_size};
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
                            int hEnd = hStart + {self.kernel_size};
                            int wEnd = wStart + {self.kernel_size};
                            int tEnd = tStart + {self.kernel_size};
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
                int tz = min(1024 / tx / ty, out0_shape2);
                dim3 block(tx, ty, tz);
                int bx = CeilDiv(out0_shape3, ty);
                int by = CeilDiv(out0_shape2, tz);
                int bz = in0_shape0 * out0_shape1;
                dim3 grid(bx, by, bz);
                // printf("%d %d %d %d %d %d\\n", block.x, block.y, block.z, bx, by, bz);
                kernel1<<<grid, block>>>(
                        in0_p, 
                        in0_shape0 * in0_shape1, in0_shape2, in0_shape3, in0_shape4,
                        out0_p, 
                        out0_shape0 * out0_shape1, out0_shape2, out0_shape3, out0_shape4
                );
                cudaError_t cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess)
                    printf("addKernel launch failed: %s\\n", cudaGetErrorString(cudaStatus));
            """,
            )
        return out
