from jittor.nn import Pool
from jittor import init, Module
import numpy as np
import jittor as jt


class Pool2D(Module):
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
        N, C, H, W = x.shape
        if self.ceil_mode == False:
            h = (H + self.padding * 2 - self.kernel_size) // self.stride + 1
            w = (W + self.padding * 2 - self.kernel_size) // self.stride + 1
        else:
            h = (H + self.padding * 2 - self.kernel_size + self.stride - 1) // self.stride + 1
            w = (W + self.padding * 2 - self.kernel_size + self.stride - 1) // self.stride + 1
        out = jt.code(
            [N, C, h, w],
            x.dtype,
            [x],
            cuda_header="""
                #include <ops/binary_op_defs.h>
                #include <misc/cuda_limits.h>
            """,
            cuda_src=f"""
                __global__ static void kernel1(
                    in0_type* __restrict__ in0_p, 
                    index_t in0_shape0, index_t in0_shape1, index_t in0_shape2, index_t in0_shape3, 
                    out0_type* __restrict__ out0_p, 
                    index_t out0_shape0, index_t out0_shape1, index_t out0_shape2, index_t out0_shape3
                ) {{
                    
                    auto in0_stride2 = in0_stride3 * in0_shape3; auto in0_stride1 = in0_stride2 * in0_shape2; auto in0_stride0 = in0_stride1 * in0_shape1; 
                    auto out0_stride2 = out0_stride3 * out0_shape3; auto out0_stride1 = out0_stride2 * out0_shape2; auto out0_stride0 = out0_stride1 * out0_shape1; 

                    int p3 = threadIdx.x;
                    int s3 = blockDim.x;
                    int p2 = threadIdx.y + blockIdx.x * blockDim.y;
                    int s2 = blockDim.y * gridDim.x;
                    int i1 = blockIdx.y;
                    int i0 = blockIdx.z;
                    for (int i3 = p3; i3 < out0_shape3; i3 += s3)
                    for (int i2 = p2; i2 < out0_shape2; i2 += s2)
                    {{
                        int k3 = i3*{self.stride}-{self.padding};
                        int k2 = i2*{self.stride}-{self.padding};
                        int k3_ = min(k3 + {self.kernel_size}, in0_shape3);
                        int k2_ = min(k2 + {self.kernel_size}, in0_shape2);
                        k3 = max(0, k3);
                        k2 = max(0, k2);
                        out0_p[(i0)*out0_stride0+( i1)*out0_stride1+( i2)*out0_stride2+( i3)*out0_stride3] = init_maximum(out0_type);
                        for (int p = k2; p < k2_; ++p)
                            for (int q = k3; q < k3_; ++q)
                                out0_p[(i0)*out0_stride0+( i1)*out0_stride1+( i2)*out0_stride2+( i3)*out0_stride3] = maximum(out0_type, out0_p[(i0)*out0_stride0+( i1)*out0_stride1+( i2)*out0_stride2+( i3)*out0_stride3], in0_p[(i0)*in0_stride0+( i1)*in0_stride1+( p)*in0_stride2+( q)*in0_stride3]);
                    }}
                }}
                int tx = min(1024, out0_shape3);
                int ty = min(1024 / tx, out0_shape2);
                int bx = (out0_shape2 - 1) / ty + 1;
                int by = out0_shape1;
                int bz = out0_shape0;
                dim3 s1(bx, by, bz);
                dim3 s2(tx, ty);
                kernel1<<<s1, s2>>>(
                    in0_p, 
                    in0_shape0, in0_shape1, in0_shape2, in0_shape3, 
                    out0_p, 
                    out0_shape0, out0_shape1, out0_shape2, out0_shape3
                );
            """,
        )
        return out

        out = jt.code([N, C, h, w], x.dtype, [x], cuda_header="...", cuda_src="...")


jt.flags.use_cuda = 1
x = jt.random([256, 64, 256, 256])
m = Pool2D(3)
m_std = Pool(3)
y = m(x)
y_std = m_std(x)
assert np.allclose(y.data, y_std.data)
print("success!")