# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Custom PyTorch ops for efficient resampling of 2D images."""

import os
import numpy as np
import jittor as jt

# from .. import custom_ops
# from .. import misc
from . import conv2d_gradfix

#----------------------------------------------------------------------------

cuda_header = """
#include "upfirdn2d.cuh"
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
"""
header_path = os.path.join(os.path.dirname(__file__), 'ops')
# glm_path = os.path.join(os.path.dirname(__file__),'third_party','glm')
proj_options = {f'FLAGS: -I"{header_path}" -l"NetworkOps" -L"{os.path.dirname(__file__)}"':1}
jt.flags.use_cuda = 1
# _plugin = None

# def _init():
#     global _plugin
#     if _plugin is None:
#         _plugin = custom_ops.get_plugin(
#             module_name='upfirdn2d_plugin',
#             sources=['upfirdn2d.cpp', 'upfirdn2d.cu'],
#             headers=['upfirdn2d.h'],
#             source_dir=os.path.dirname(__file__),
#             extra_cuda_cflags=['--use_fast_math'],
#         )
#     return True

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, jt.Var) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    # with misc.suppress_tracer_warnings():
    #     fw = int(fw)
    #     fh = int(fh)
    # misc.assert_shape(f, [fh, fw][:f.ndim])
    # assert fw >= 1 and fh >= 1
    return fw, fh

#----------------------------------------------------------------------------

def setup_filter(f, device='cuda', normalize=True, flip_filter=False, gain=1, separable=None):
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f:           jt tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = jt.array(f, dtype=jt.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.unsqueeze(1) @ f.unsqueeze(0) # f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    # f = f.to(device=device)
    return f

#----------------------------------------------------------------------------

def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, jt.Var)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda':
        return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

#----------------------------------------------------------------------------

# @misc.profiled_function
def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    # Validate arguments.
    assert isinstance(x, jt.Var) and x.ndim == 4
    if f is None:
        f = jt.ones([1, 1], dtype=jt.float32, device=x.device)
    assert isinstance(f, jt.Var) and f.ndim in [1, 2]
    f.stop_grad()
    assert f.dtype == jt.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Check that upsampled buffer is not smaller than the filter.
    upW = in_width * upx + padx0 + padx1
    upH = in_height * upy + pady0 + pady1
    assert upW >= f.shape[-1] and upH >= f.shape[0]

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = jt.nn.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = jt.nn.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = conv2d_gradfix.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = conv2d_gradfix.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = conv2d_gradfix.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x

#----------------------------------------------------------------------------

_upfirdn2d_cuda_cache = dict()

def _upfirdn2d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast CUDA implementation of `upfirdn2d()` using custom ops.
    """
    # Parse arguments.
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Lookup from cache.
    key = (upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
    if key in _upfirdn2d_cuda_cache:
        return _upfirdn2d_cuda_cache[key]

    # Forward op.
    class Upfirdn2dCuda(jt.Function):
        def save_for_backward(self,*args):
            self.saved_tensors = args

        def execute(self, x, f): # pylint: disable=arguments-differ
            assert isinstance(x, jt.Var) and x.ndim == 4
            if f is None:
                f = jt.ones([1, 1], dtype=jt.float32, device=x.device)
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(0) # Convert separable-1 into full-1x1.
            assert isinstance(f, jt.Var) and f.ndim in [1, 2]
            # f.stop_grad()
            y = x
            if f.ndim == 2:
                y = upfirdn2d_ops(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = upfirdn2d_ops(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, 1.0)
                y = upfirdn2d_ops(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, gain)
            self.save_for_backward(f)
            self.x_shape = x.shape
            return y


        def grad(self, dy): # pylint: disable=arguments-differ
            f, = self.saved_tensors
            _, _, ih, iw = self.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [
                fw - padx0 - 1,
                iw * upx - ow * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            ]
            dx = None
            df = None

            if self.needs_input_grad[0]:
                dx = _upfirdn2d_cuda(up=down, down=up, padding=p, flip_filter=(not flip_filter), gain=gain).apply(dy, f)

            assert not self.needs_input_grad[1]
            return dx, df

    # Add to cache.
    _upfirdn2d_cuda_cache[key] = Upfirdn2dCuda
    return Upfirdn2dCuda

#----------------------------------------------------------------------------

def filter2d(x, f, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Filter a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape matches the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + fw // 2,
        padx1 + (fw - 1) // 2,
        pady0 + fh // 2,
        pady1 + (fh - 1) // 2,
    ]
    return upfirdn2d(x, f, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------

def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx*upy, impl=impl)

#----------------------------------------------------------------------------

def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Downsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the input. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------
# static torch::Tensor upfirdn2d(torch::Tensor x, torch::Tensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain)
def upfirdn2d_ops(x, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip, gain):
    # Calculate output dimensions
    outW = (x.shape[3] * upx + padx0 + padx1 - f.shape[1] + downx) // downx
    outH = (x.shape[2] * upy + pady0 + pady1 - f.shape[0] + downy) // downy
    # Check output dimensions
    assert outW >= 1 and outH >= 1, "output must be at least 1x1"
    y = jt.empty((x.shape[0], x.shape[1], outH, outW), dtype=x.dtype)
    with jt.flag_scope(compile_options=proj_options):
        (y,) = jt.code(
                        inputs=[x, f],
                        outputs=[y],
                        data={
                                'upx':upx,
                                'upy':upy,
                                'downx':downx,
                                'downy':downy,
                                'padx0':padx0,
                                'padx1':padx1,
                                'pady0':pady0,
                                'pady1':pady1,
                                'flip':int(flip),
                                'gain':int(gain),                                    
                            },
                        cuda_header=cuda_header,
                        cuda_src='''
    upfirdn2d_kernel_params p;
    p.x             = in0_p;
    p.f             = in1_p;
    p.y             = out0_p;
    p.up            = make_int2(data["upx"], data["upy"]);
    p.down          = make_int2(data["downx"], data["downy"]);
    p.pad0          = make_int2(data["padx0"], data["pady0"]);
    p.flip          = (data["flip"]) ? 1 : 0;
    p.gain          = data["gain"];
    p.inSize = make_int4(in0_shape3, in0_shape2, in0_shape1, in0_shape0);
    p.inStride = make_int4(in0_stride3, in0_stride2, in0_stride1, in0_stride0);
    p.filterSize = make_int2(in1_shape1, in1_shape0);
    p.filterStride = make_int2(in1_stride1, in1_stride0);
    p.outSize = make_int4(out0_shape3, out0_shape2, out0_shape1, out0_shape0);
    p.outStride = make_int4(out0_stride3, out0_stride2, out0_stride1, out0_stride0);
    p.sizeMajor     = (p.inStride.z == 1) ? p.inSize.w : p.inSize.w * p.inSize.z;
    p.sizeMinor     = (p.inStride.z == 1) ? p.inSize.z : 1;
    upfirdn2d_kernel_spec spec;
    // spec = choose_upfirdn2d_kernel<scalar_t>(p);
    spec = choose_upfirdn2d_kernel<float>(p);
    p.loopMajor     = (p.sizeMajor - 1) / 16384 + 1;
    p.loopMinor     = spec.loopMinor;
    p.loopX         = spec.loopX;
    p.launchMinor   = (p.sizeMinor - 1) / p.loopMinor + 1;
    p.launchMajor   = (p.sizeMajor - 1) / p.loopMajor + 1;



    dim3 blockSize, gridSize;
    if (spec.tileOutW < 0) // large
    {
        blockSize = dim3(4, 32, 1);
        gridSize = dim3(
            ((p.outSize.y - 1) / blockSize.x + 1) * p.launchMinor,
            (p.outSize.x - 1) / (blockSize.y * p.loopX) + 1,
            p.launchMajor);
    }
    else // small
    {
        blockSize = dim3(256, 1, 1);
        gridSize = dim3(
            ((p.outSize.y - 1) / spec.tileOutH + 1) * p.launchMinor,
            (p.outSize.x - 1) / (spec.tileOutW * p.loopX) + 1,
            p.launchMajor);
    }


    void* args[] = {&p};
    cudaLaunchKernel(spec.kernel, gridSize, blockSize, args, 0, get_current_cuda_stream());
                        '''
                    )
    return y

if __name__ == "__main__":
    x = jt.rand([2,4,256,256]).float()
    f = jt.rand([3,3]).float()
    f.stop_grad()
    up = 2
    down=2
    # y = bias_act(x,b=b,act='sigmoid',impl='ref')
    # print(y)
    y1 = upfirdn2d(x,f,up,down,impl='ref')
    
    y2 = upfirdn2d(x,f,up,down)
    print(y1-y2)