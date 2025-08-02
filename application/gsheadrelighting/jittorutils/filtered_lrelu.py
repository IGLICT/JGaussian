# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Custom Pyjt ops for efficient bias and activation."""

import os
import numpy as np
import jittor as jt
from jittor import nn
from pathlib import Path
import sys
parent_dir = str(Path(__file__).parent.parent.absolute())
sys.path.append(parent_dir)
# import dnnlib
import bias_act
import dnnlib
import warnings
import upfirdn2d
# from . import custom_ops
# from . import misc

#----------------------------------------------------------------------------

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, jt.Var)
    assert 1 <= f.ndim <= 2
    return f.shape[-1], f.shape[0] # width, height

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, (int, np.integer)) for x in padding)
    padding = [int(x) for x in padding]
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    px0, px1, py0, py1 = padding
    return px0, px1, py0, py1

#----------------------------------------------------------------------------


_null_tensor = jt.array([])




cuda_header = """
#include "filtered_lrelu.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
cudaStream_t get_current_cuda_stream() {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA stream");
        exit(EXIT_FAILURE);
    }
    return stream;
}

std::vector<int> compute_strides(const jittor::NanoVector& shape, int itemsize) {
    std::vector<int> strides;
    int stride = itemsize;  // 初始 stride 是 itemsize（字节数）
    // 逆序遍历 shape（从最后一个维度开始）
    for (int i=shape.size()-1;i>=0;i--) {
        strides.insert(strides.begin(), stride);  // 在开头插入当前 stride
        stride *= shape[i];  
    }
    return strides;
}
"""
header_path = os.path.join(os.path.dirname(__file__), 'ops')
# glm_path = os.path.join(os.path.dirname(__file__),'third_party','glm')
proj_options = {f'FLAGS: -I"{header_path}" -l"NetworkOps" -L"{os.path.dirname(__file__)}"':1}
jt.flags.use_cuda = 1
#----------------------------------------------------------------------------

def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl='cuda'):
    r"""Filtered leaky ReLU for a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Add channel-specific bias if provided (`b`).

    2. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    3. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    4. Convolve the image with the specified upsampling FIR filter (`fu`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    5. Multiply each value by the provided gain factor (`gain`).

    6. Apply leaky ReLU activation function to each value.

    7. Clamp each value between -clamp and +clamp, if `clamp` parameter is provided.

    8. Convolve the image with the specified downsampling FIR filter (`fd`), shrinking
       it so that the footprint of all output pixels lies within the input image.

    9. Downsample the image by keeping every Nth pixel (`down`).

    The fused op is considerably more efficient than performing the same calculation
    using standard Pyjt ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float16/float64 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        fu:          Float32 upsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        fd:          Float32 downsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        b:           Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                     as `x`. The length of vector must must match the channel dimension of `x`.
        up:          Integer upsampling factor (default: 1).
        down:        Integer downsampling factor. (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        gain:        Overall scaling factor for signal magnitude (default: sqrt(2)).
        slope:       Slope on the negative side of leaky ReLU (default: 0.2).
        clamp:       Maximum magnitude for leaky ReLU output (default: None).
        flip_filter: False = convolution, True = correlation (default: False).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, jt.Var)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda':
        return _filtered_lrelu_cuda(up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter).apply(x, fu, fd, b, None, 0, 0)
    return _filtered_lrelu_ref(x, fu=fu, fd=fd, b=b, up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter)
#----------------------------------------------------------------------------

# @misc.profiled_function
def _filtered_lrelu_ref(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Slow and memory-inefficient reference implementation of `filtered_lrelu()` using
    existing `upfirdn2n()` and `bias_act()` ops.
    """
    assert isinstance(x, jt.Var) and x.ndim == 4
    fu_w, fu_h = _get_filter_size(fu)
    fd_w, fd_h = _get_filter_size(fd)
    if b is not None:
        assert isinstance(b, jt.Var) and b.dtype == x.dtype
        # misc.assert_shape(b, [x.shape[1]])
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)

    # Calculate output size.
    batch_size, channels, in_h, in_w = x.shape
    in_dtype = x.dtype
    out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) // down
    out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) // down

    # Compute using existing ops.
    x = bias_act.bias_act(x=x, b=b) # Apply bias.
    x = upfirdn2d.upfirdn2d(x=x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter) # Upsample.
    x = bias_act.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp) # Bias, leaky ReLU, clamp.
    x = upfirdn2d.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter) # Downsample.

    # Check output shape & dtype.
    # misc.assert_shape(x, [batch_size, channels, out_h, out_w])
    # assert x.dtype == in_dtype
    return x

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

_filtered_lrelu_cuda_cache = dict()

def _filtered_lrelu_cuda(up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Fast CUDA implementation of `filtered_lrelu()` using custom ops.
    """
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    clamp = float(clamp if clamp is not None else 'inf')

    # Lookup from cache.
    key = (up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter)
    if key in _filtered_lrelu_cuda_cache:
        return _filtered_lrelu_cuda_cache[key]

    # Forward op.
    class FilteredLReluCuda(jt.Function):
        def save_for_backward(self,*args):
            self.saved_tensors = args

        def execute(self, x, fu, fd, b, si, sx, sy): # pylint: disable=arguments-differ
            assert isinstance(x, jt.Var) and x.ndim == 4

            # Replace empty up/downsample kernels with full 1x1 kernels (faster than separable).
            if fu is None:
                fu = jt.ones([1, 1], dtype=jt.float32)
            if fd is None:
                fd = jt.ones([1, 1], dtype=jt.float32)
            assert 1 <= fu.ndim <= 2
            assert 1 <= fd.ndim <= 2

            # Replace separable 1x1 kernels with full 1x1 kernels when scale factor is 1.
            if up == 1 and fu.ndim == 1 and fu.shape[0] == 1:
                fu = fu.square()[None]
            if down == 1 and fd.ndim == 1 and fd.shape[0] == 1:
                fd = fd.square()[None]

            # Missing sign input tensor.
            if si is None:
                si = jt.array([])

            # Missing bias tensor.
            if b is None:
                b = jt.zeros([x.shape[1]], dtype=x.dtype)

            # Construct internal sign tensor only if gradients are needed.
            write_signs = (si.numel() == 0) and (x.requires_grad or b.requires_grad)

            # Warn if input storage strides are not in decreasing order due to e.g. channels-last layout.
            # strides = [x.stride(i) for i in range(x.ndim) if x.size(i) > 1]
            # if any(a < b for a, b in zip(strides[:-1], strides[1:])):
            #     warnings.warn("low-performance memory layout detected in filtered_lrelu input", RuntimeWarning)
            strides = []
            stride = 1  # 初始 stride 是元素的大小（字节数）
            for i in reversed(range(x.ndim)):
                strides.insert(0, stride)  # 在列表开头插入当前 stride
                stride *= x.shape[i] if x.shape[i] > 1 else 0  # 如果维度大小为 1，不增加 stride
            strides = [s for i, s in enumerate(strides) if x.shape[i] > 1]  # 过滤掉 shape=1 的维度
            # strides = [x.stride for i in range(x.ndim) if x.shape[i] > 1]
            if any(a < b for a, b in zip(strides[:-1], strides[1:])):
                warnings.warn("low-performance memory layout detected in filtered_lrelu input", RuntimeWarning)

            # Call C++/Cuda plugin if datatype is supported.
            if x.dtype in [jt.float16, jt.float32]:
                # if jt.cuda.current_stream(x.device) != jt.cuda.default_stream(x.device):
                #     warnings.warn("filtered_lrelu called with non-default cuda stream but concurrent execution is not supported", RuntimeWarning)
                y, so, return_code = filtered_lrelu_ops(x, fu, fd, b, si, up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp, flip_filter, write_signs)
            else:
                return_code = -1

            # No Cuda kernel found? Fall back to generic implementation. Still more memory efficient than the reference implementation because
            # only the bit-packed sign tensor is retained for gradient computation.
            if return_code < 0:
                warnings.warn("filtered_lrelu called with parameters that have no optimized CUDA kernel, using generic fallback", RuntimeWarning)

                y = x.add(b.unsqueeze(-1).unsqueeze(-1)) # Add bias.
                y = upfirdn2d.upfirdn2d(x=y, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter) # Upsample.
                so = filtered_lrelu_act_ops(y, si, sx, sy, gain, slope, clamp, write_signs) # Activation function and sign handling. Modifies y in-place.
                y = upfirdn2d.upfirdn2d(x=y, f=fd, down=down, flip_filter=flip_filter) # Downsample.

            # Prepare for gradient computation.
            self.save_for_backward(x, b, fu, fd, (si if si.numel() else so))
            self.x_shape = x.shape
            self.y_shape = y.shape
            self.s_ofs = sx, sy
            return y

        def backward(self, dy): # pylint: disable=arguments-differ
            x, b, fu, fd, si = self.saved_tensors
            _, _, xh, xw = self.x_shape
            _, _, yh, yw = self.y_shape
            sx, sy = self.s_ofs
            # dx  = None # 0
            # dfu = None; assert not self.needs_input_grad[1]
            # dfd = None; assert not self.needs_input_grad[2]
            # db  = None # 3
            # dsi = None; assert not self.needs_input_grad[4]
            # dsx = None; assert not self.needs_input_grad[5]
            # dsy = None; assert not self.needs_input_grad[6]
            dx  = jt.zeros_like(x) # 0
            dfu = jt.zeros_like(fu)
            dfd = jt.zeros_like(fd)
            db  = jt.zeros_like(b) # 3
            dsi = jt.zeros_like(si)
            dsx = jt.zeros_like(sx)
            dsy = jt.zeros_like(sy)

            if x.requires_grad or b.requires_grad:
                pp = [
                    (fu.shape[-1] - 1) + (fd.shape[-1] - 1) - px0,
                    xw * up - yw * down + px0 - (up - 1),
                    (fu.shape[0] - 1) + (fd.shape[0] - 1) - py0,
                    xh * up - yh * down + py0 - (up - 1),
                ]
                gg = gain * (up ** 2) / (down ** 2)
                ff = (not flip_filter)
                sx = sx - (fu.shape[-1] - 1) + px0
                sy = sy - (fu.shape[0]  - 1) + py0
                dx = _filtered_lrelu_cuda(up=down, down=up, padding=pp, gain=gg, slope=slope, clamp=None, flip_filter=ff).apply(dy, fd, fu, None, si, sx, sy)

            if b.requires_grad:
                db = dx.sum([0, 2, 3])

            return dx, dfu, dfd, db, dsi, dsx, dsy

    # Add to cache.
    _filtered_lrelu_cuda_cache[key] = FilteredLReluCuda
    return FilteredLReluCuda

#----------------------------------------------------------------------------

# static std::tuple<torch::Tensor, torch::Tensor, int> filtered_lrelu(
#     torch::Tensor x, torch::Tensor fu, torch::Tensor fd, torch::Tensor b, torch::Tensor si,
#     int up, int down, int px0, int px1, int py0, int py1, int sx, int sy, float gain, float slope, float clamp, bool flip_filters, bool writeSigns)
def filtered_lrelu_ops(x, fu, fd, b, si,
    up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp, flip_filters, writeSigns):
    # // Input sizes.
    # int64_t xw = (int)x->shape[3];
    # int64_t xh = (int)x->shape[2];
    # int64_t fut_w = (int)fu->shape[(fu->shape).size()-1] - 1;
    # int64_t fut_h = (int)fu->shape[0]  - 1;
    # int64_t fdt_w = (int)fd->shape[(fd->shape).size()-1] - 1;
    # int64_t fdt_h = (int)fd->shape[0]  - 1;

    # // Logical size of upsampled buffer.
    # int64_t cw = xw * up + (px0 + px1) - fut_w;
    # int64_t ch = xh * up + (py0 + py1) - fut_h;

    # // Compute output size and allocate.
    # int64_t yw = (cw - fdt_w + (down - 1)) / down;
    # int64_t yh = (ch - fdt_h + (down - 1)) / down;

    # // torch::Tensor y = torch::empty({x.size(0), x.size(1), yh, yw}, x.options(), x.suggest_memory_format());
    # // Allocate sign tensor.
    # //torch::Tensor so;
    # //torch::Tensor s = si;
    # bool readSigns = !!s.numel();
    # int64_t sw_active = 0; // Active width of sign tensor.
    # if (writeSigns)
    # {
    #     sw_active = yw * down - (down - 1) + fdt_w;     // Active width in elements.
    #     int64_t sh = yh * down - (down - 1) + fdt_h;    // Height = active height.
    #     int64_t sw = (sw_active + 15) & ~15;            // Width  = active width in elements, rounded up to multiple of 16.
    #     TORCH_CHECK(sh <= INT_MAX && (sw >> 2) <= INT_MAX, "signs is too large");
    #     s = so = torch::empty({x.size(0), x.size(1), sh, sw >> 2}, x.options().dtype(torch::kUInt8), at::MemoryFormat::Contiguous);
    # }
    # else if (readSigns)
    #     sw_active = s.size(3) << 2;
    # 获取输入张量形状
    xw = x.shape[3]
    xh = x.shape[2]
    # 获取上采样滤波器尺寸（最后两个维度）
    fut_w = fu.shape[-1] - 1
    fut_h = fu.shape[0] - 1
    # 获取下采样滤波器尺寸（最后两个维度）
    fdt_w = fd.shape[-1] - 1
    fdt_h = fd.shape[0] - 1
    # 计算上采样后的逻辑尺寸
    cw = xw * up + (px0 + px1) - fut_w
    ch = xh * up + (py0 + py1) - fut_h
    # 计算输出尺寸（考虑下采样）
    yw = (cw - fdt_w + (down - 1)) // down
    yh = (ch - fdt_h + (down - 1)) // down
    # 在Jittor中分配输出张量（假设输出是4D张量）
    y = jt.zeros((x.shape[0], x.shape[1], yh, yw), dtype=x.dtype)
    so = jt.array([])
    s = si
    readSigns = bool(s.numel())
    sw_active = 0
    if writeSigns:
        # 计算激活宽度和高度
        sw_active = yw * down - (down - 1) + fdt_w
        sh = yh * down - (down - 1) + fdt_h
        # 宽度向上取整到16的倍数
        sw = (sw_active + 15) & ~15
        # # 检查尺寸是否在int范围内
        # assert sh <= 0x7fffffff and (sw >> 2) <= 0x7fffffff, "signs is too large"
        # 创建空的uint8张量
        so = s = jt.empty((x.shape[0], x.shape[1], sh, sw >> 2), dtype='uint8').contiguous()
        # Jittor默认使用连续内存布局，无需特别指定
    elif readSigns:
        sw_active = s.shape[3] << 2  # 从存储宽度计算实际宽度 


    with jt.flag_scope(compile_options=proj_options):
        y, s = jt.code(
                        inputs=[x, fu, fd, b],
                        outputs=[y, s],
                        data={
                                'up':up,
                                'down':down,
                                'px0':px0,
                                'px1':px1,
                                'px0':px0,
                                'py1':py1,
                                'py0':py0,
                                'sx':sx,
                                'sy':sy,
                                'gain':gain,
                                'slope':slope,
                                'clamp':clamp,
                                'flip_filters':int(flip_filters),
                                'writeSigns':int(writeSigns),
                                'readSigns':int(readSigns),
                                'sw_active': sw_active,
                            },
                        cuda_header=cuda_header,
                        cuda_src='''
                        
    // Figure out how much shared memory is available on the device.
    int maxSharedBytes = 0;
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&maxSharedBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    int sharedKB = maxSharedBytes >> 10;

    // Populate enough launch parameters to check if a CUDA kernel exists.
    filtered_lrelu_kernel_params p;
    p.up      = data["up"];
    p.down    = data["down"];
    // p.fuShape = make_int2((int)fu.size(-1), fu.dim() == 2 ? (int)fu.size(0) : 0); // shape [n, 0] indicates separable filter.
    p.fuShape = make_int2(in1->shape[(in1->shape).size()-1], (in1->shape).size() == 2 ? in1->shape[0] : 0); // shape [n, 0] indicates separable filter.
    p.fdShape = make_int2(in2->shape[(in2->shape).size()-1], (in2->shape).size() == 2 ? in2->shape[0] : 0);
    filtered_lrelu_kernel_spec test_spec = choose_filtered_lrelu_kernel<float, int32_t, false, false>(p, sharedKB);

    // Input/output element size.
    //int64_t sz = (x.dtype() == torch::kHalf) ? 2 : 4;
    int64_t sz = 4;
    


    // Populate rest of CUDA kernel parameters.
    bool readSigns = data["readSigns"];
    bool writeSigns = data["writeSigns"];
    p.x         = in0_p;
    p.y         = out0_p;
    p.b         = in3_p;
    // p.s         = (data["readSigns"] || data["writeSigns"]) ? out1_p : 0;
    p.s         = (data["readSigns"] || data["writeSigns"]) ? out1_p : nullptr;
    p.fu        = in1_p;
    p.fd        = in2_p;
    p.pad0      = make_int2(data["px0"], data["py0"]);
    p.gain      = data["gain"];
    p.slope     = data["slope"];
    p.clamp     = data["clamp"];
    p.flip      = (data["flip_filters"]) ? 1 : 0;
    p.xShape    = make_int4((int)in0_shape3, (int)in0_shape2, (int)in0_shape1, (int)in0_shape0);
    p.yShape    = make_int4((int)out0_shape3, (int)out0_shape2, (int)out0_shape1, (int)out0_shape0);
    p.sShape    = (data["readSigns"] || data["writeSigns"]) ? make_int2((int)out1_shape3, (int)out1_shape2) : make_int2(0, 0); // Width is in bytes. Contiguous.
    p.sOfs      = make_int2(data["sx"], data["sy"]);
    p.swLimit   = ((int)data["sw_active"] + 3) >> 2; // Rounded up to bytes.
    
    // x, y, b strides are in bytes.
    p.xStride   = make_longlong4(sz * in0_stride3, sz * in0_stride2, sz * in0_stride1, sz * in0_stride0);
    p.yStride   = make_longlong4(sz * out0_stride3, sz * out0_stride2, sz * out0_stride1, sz * out0_stride0);
    p.bStride   = sz * in3_stride0;

    // fu, fd strides are in elements.
    std::vector<int> in1_strides = compute_strides(in1->shape, 1);

    std::vector<int> in2_strides = compute_strides(in2->shape, 1);

    p.fuStride  = make_longlong3(in1_strides[(in1->shape).size()-1], (in1->shape).size() == 2 ? in1_stride0 : 0, 0);

    p.fdStride  = make_longlong3(in2_strides[(in2->shape).size()-1], (in2->shape).size() == 2 ? in2_stride0 : 0, 0);
    //p.fdStride  = make_longlong3(in2->stride[(in2->stride).size()-1], (in2->shape).size() == 2 ? in2_stride0 : 0, 0);

    // Determine if indices don't fit in int32. Support negative strides although Torch currently never produces those.
    bool index64b = false;
    if (std::abs(p.bStride * in0_shape1) > INT_MAX) index64b = true;
    if (std::min(in0_shape0 * p.xStride.w, 0ll) + std::min(in0_shape1 * p.xStride.z, 0ll) + std::min(in0_shape2 * p.xStride.y, 0ll) + std::min(in0_shape3 * p.xStride.x, 0ll) < -INT_MAX) index64b = true;
    if (std::max(in0_shape0 * p.xStride.w, 0ll) + std::max(in0_shape1 * p.xStride.z, 0ll) + std::max(in0_shape2 * p.xStride.y, 0ll) + std::max(in0_shape3 * p.xStride.x, 0ll) >  INT_MAX) index64b = true;
    if (std::min(out0_shape0 * p.yStride.w, 0ll) + std::min(out0_shape1 * p.yStride.z, 0ll) + std::min(out0_shape2 * p.yStride.y, 0ll) + std::min(out0_shape3 * p.yStride.x, 0ll) < -INT_MAX) index64b = true;
    if (std::max(out0_shape0 * p.yStride.w, 0ll) + std::max(out0_shape1 * p.yStride.z, 0ll) + std::max(out0_shape2 * p.yStride.y, 0ll) + std::max(out0_shape3 * p.yStride.x, 0ll) >  INT_MAX) index64b = true;
    if (out1->numel() > INT_MAX) index64b = true;

    // Choose CUDA kernel.
    filtered_lrelu_kernel_spec spec;

    //filtered_lrelu_kernel_spec spec;
    typedef float scalar_t;
    if constexpr (sizeof(scalar_t) <= 4) // Exclude doubles. constexpr prevents template instantiation.
    {
            // Choose kernel based on index type, datatype and sign read/write modes.
            
            if      (!index64b &&  writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int32_t, true,  false>(p, sharedKB);
            else if (!index64b && !writeSigns &&  readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int32_t, false, true >(p, sharedKB);
            else if (!index64b && !writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int32_t, false, false>(p, sharedKB);
            else if ( index64b &&  writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int64_t, true,  false>(p, sharedKB);
            else if ( index64b && !writeSigns &&  readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int64_t, false, true >(p, sharedKB);
            else if ( index64b && !writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int64_t, false, false>(p, sharedKB);
    }



    // Launch CUDA kernel.
    void* args[] = {&p};
    int bx = spec.numWarps * 32;

    int gx = (p.yShape.x - 1) / spec.tileOut.x + 1;

    int gy = (p.yShape.y - 1) / spec.tileOut.y + 1;

    int gz = p.yShape.z * p.yShape.w;


    // Repeat multiple horizontal tiles in a CTA?
    if (spec.xrep)
    {
        p.tilesXrep = spec.xrep;
        p.tilesXdim = gx;
        gx = (gx + p.tilesXrep - 1) / p.tilesXrep;
        std::swap(gx, gy);
    }
    else
    {
        p.tilesXrep = 0;
        p.tilesXdim = 0;
    }

    // Launch filter setup kernel.
    cudaLaunchKernel(spec.setup, 1, 1024, args, 0, get_current_cuda_stream());

    // Copy kernels to constant memory.
    if      ( writeSigns && !readSigns) copy_filters<true,  false>(get_current_cuda_stream());
    else if (!writeSigns &&  readSigns) copy_filters<false, true >(get_current_cuda_stream());
    else if (!writeSigns && !readSigns) copy_filters<false, false>(get_current_cuda_stream());

    // Set cache and shared memory configurations for main kernel.
    cudaFuncSetCacheConfig(spec.exec, cudaFuncCachePreferShared);
    if (spec.dynamicSharedKB) // Need dynamically allocated shared memory?
        cudaFuncSetAttribute(spec.exec, cudaFuncAttributeMaxDynamicSharedMemorySize, spec.dynamicSharedKB << 10);
    cudaFuncSetSharedMemConfig(spec.exec, cudaSharedMemBankSizeFourByte);

    // Launch main kernel.
    const int maxSubGz = 65535; // CUDA maximum for block z dimension.
    for (int zofs=0; zofs < gz; zofs += maxSubGz) // Do multiple launches if gz is too big.
    {
        p.blockZofs = zofs;
        int subGz = std::min(maxSubGz, gz - zofs);
        cudaLaunchKernel(spec.exec, dim3(gx, gy, subGz), bx, args, spec.dynamicSharedKB << 10, get_current_cuda_stream());
    }
                  '''
                    )
    if writeSigns:
        return y, s, 0
    return y, so, 0
    
# static torch::Tensor filtered_lrelu_act(torch::Tensor x, torch::Tensor si, int sx, int sy, float gain, float slope, float clamp, bool writeSigns)
def filtered_lrelu_act_ops(x, si, sx, sy, gain, slope, clamp, writeSigns):
    so = jt.array([])
    s = si
    readSigns = bool(s.numel())
    if writeSigns:
        sw = x.shape[3]  # 获取 x 的第 4 维大小
        sw = (sw + 15) & ~15  # 向上取整到 16 的倍数（优化内存访问）
        # 创建一个新的张量，数据类型为 uint8，内存连续
        s = so = jt.empty((x.shape[0], x.shape[1], x.shape[2], sw >> 2), dtype='uint8')
    with jt.flag_scope(compile_options=proj_options):
        (s,) = jt.code(
                        inputs=[x],
                        outputs=[s],
                        data={
                                'sx':sx,
                                'sy':sy,
                                'gain':gain,
                                'slope':slope,
                                'clamp':clamp,
                                'writeSigns':int(writeSigns),
                                'readSigns':int(readSigns),
                            },
                        cuda_header=cuda_header,
                        cuda_src='''
    filtered_lrelu_act_kernel_params p;
    p.x         = in0_p;
    p.s         = (readSigns || writeSigns) ? out0_p : 0;
    p.gain      = data["gain"];
    p.slope     = data["slope"];
    p.clamp     = data["clamp"];
    p.xShape    = make_int4(in0_shape3, in0_shape2, in0_shape1, in0_shape0);
    p.xStride   = make_longlong4(in0_stride3, in0_stride2, in0_stride1, in0_stride0);
    p.sShape    = (readSigns || writeSigns) ? make_int2(out0_shape3 << 2, out0_shape2) : make_int2(0, 0); // Width is in elements. Contiguous.
    p.sOfs      = make_int2(data["sx"], data["sy"]);
    void* func = 0;

    if (writeSigns)
        func = choose_filtered_lrelu_act_kernel<scalar_t, true, false>();
    else if (readSigns)
        func = choose_filtered_lrelu_act_kernel<scalar_t, false, true>();
    else
        func = choose_filtered_lrelu_act_kernel<scalar_t, false, false>();

    // Launch CUDA kernel.
    void* args[] = {&p};
    int bx = 128; // 4 warps per block.

    // Logical size of launch = writeSigns ? p.s : p.x
    uint32_t gx = writeSigns ? p.sShape.x : p.xShape.x;
    uint32_t gy = writeSigns ? p.sShape.y : p.xShape.y;
    uint32_t gz = p.xShape.z * p.xShape.w; // Same as in p.sShape if signs are in use.
    gx = (gx - 1) / bx + 1;

    // Make sure grid y and z dimensions are within CUDA launch limits. Kernel loops internally to do the rest.
    const uint32_t gmax = 65535;
    gy = std::min(gy, gmax);
    gz = std::min(gz, gmax);
    // Launch.
    cudaLaunchKernel(func, dim3(gx, gy, gz), bx, args, 0, get_current_cuda_stream());
    ''')
    if writeSigns:
        return s
    return so

if __name__ == "__main__":
    import time
    jt.flags.use_cuda = 1
    x = jt.rand([2,3,512,512]).float()
    fu = jt.rand([1,1]).float()
    fd = jt.rand([1,1]).float()
    time1 = time.time()
    y1 = filtered_lrelu(x,fu,fd,impl='ref')
    time2 = time.time()
    y2 = filtered_lrelu(x,fu,fd)
    time3 = time.time()
    print(y1-y2)
    print(time2-time1)
    print(time3-time2)