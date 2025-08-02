# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Custom PyTorch ops for efficient bias and activation."""

import os
import numpy as np
import jittor as jt
from jittor import nn
from pathlib import Path
import sys
parent_dir = str(Path(__file__).parent.parent.absolute())
sys.path.append(parent_dir)
# import dnnlib
import dnnlib
# from . import custom_ops
# from . import misc

#----------------------------------------------------------------------------

activation_funcs = {
    'linear':   dnnlib.EasyDict(func=lambda x, **_:         x,                                          def_alpha=0,    def_gain=1,             cuda_idx=1, ref='',  has_2nd_grad=False),
    'relu':     dnnlib.EasyDict(func=lambda x, **_:         nn.relu(x),                def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=2, ref='y', has_2nd_grad=False),
    'lrelu':    dnnlib.EasyDict(func=lambda x, alpha, **_:  nn.leaky_relu(x, alpha),   def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', has_2nd_grad=False),
    'tanh':     dnnlib.EasyDict(func=lambda x, **_:         jt.tanh(x),                              def_alpha=0,    def_gain=1,             cuda_idx=4, ref='y', has_2nd_grad=True),
    'sigmoid':  dnnlib.EasyDict(func=lambda x, **_:         jt.sigmoid(x),                           def_alpha=0,    def_gain=1,             cuda_idx=5, ref='y', has_2nd_grad=True),
    'elu':      dnnlib.EasyDict(func=lambda x, **_:         jt.nn.elu(x),                 def_alpha=0,    def_gain=1,             cuda_idx=6, ref='y', has_2nd_grad=True),
    'selu':     dnnlib.EasyDict(func=lambda x, **_:         jt.nn.selu(x),                def_alpha=0,    def_gain=1,             cuda_idx=7, ref='y', has_2nd_grad=True),
    'softplus': dnnlib.EasyDict(func=lambda x, **_:         jt.nn.softplus(x),            def_alpha=0,    def_gain=1,             cuda_idx=8, ref='y', has_2nd_grad=True),
    'swish':    dnnlib.EasyDict(func=lambda x, **_:         jt.sigmoid(x) * x,                       def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=9, ref='x', has_2nd_grad=True),
}

#----------------------------------------------------------------------------

_plugin = None
_null_tensor = jt.array([])

# def _init():
#     global _plugin
#     if _plugin is None:
#         _plugin = custom_ops.get_plugin(
#             module_name='bias_act_plugin',
#             sources=['bias_act.cpp', 'bias_act.cu'],
#             headers=['bias_act.h'],
#             source_dir=os.path.dirname(__file__),
#             extra_cuda_cflags=['--use_fast_math'],
#         )
#     return True
cuda_header = """
#include "bias_act.cuh"
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
#----------------------------------------------------------------------------

def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None, impl='cuda'):
    r"""Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x:      Input activation tensor. Can be of any shape.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `dim`.
        dim:    The dimension in `x` corresponding to the elements of `b`.
                The value of `dim` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying 1.
        clamp:  Clamp the output values to `[-clamp, +clamp]`, or `None` to disable
                the clamping (default).
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the same shape and datatype as `x`.
    """
    assert isinstance(x, jt.Var)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda':
        return _bias_act_cuda(dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp).apply(x, b)
    else:
        return _bias_act_ref(x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)

#----------------------------------------------------------------------------

# @misc.profiled_function
def _bias_act_ref(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """Slow reference implementation of `bias_act()` using standard TensorFlow ops.
    """
    assert isinstance(x, jt.Var)
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Add bias.
    if b is not None:
        assert isinstance(b, jt.Var) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    # Evaluate activation function.
    alpha = float(alpha)
    x = spec.func(x, alpha=alpha)

    # Scale by gain.
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        x = x.clamp(-clamp, clamp) # pylint: disable=invalid-unary-operand-type
    return x

#----------------------------------------------------------------------------

_bias_act_cuda_cache = dict()

def _bias_act_cuda(dim=1, act='linear', alpha=None, gain=None, clamp=None):
    """Fast CUDA implementation of `bias_act()` using custom ops.
    """
    # Parse arguments.
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Lookup from cache.
    key = (dim, act, alpha, gain, clamp)
    if key in _bias_act_cuda_cache:
        return _bias_act_cuda_cache[key]

    # Forward op.
    class BiasActCuda(jt.Function):
        def save_for_backward(self,*args):
            self.saved_tensors = args

        def execute(self, x, b): # pylint: disable=arguments-differ
            # self.memory_format = torch.channels_last if x.ndim > 2 and x.stride(1) == 1 else torch.contiguous_format
            # x = x.contiguous(memory_format=self.memory_format)
            x = x.contiguous()
            b = b.contiguous() if b is not None else _null_tensor
            y = x
            if act != 'linear' or gain != 1 or clamp >= 0 or b is not _null_tensor:
                y = bias_act_ops(x, b, _null_tensor, _null_tensor, _null_tensor, 0, dim, spec.cuda_idx, alpha, gain, clamp)

            self.save_for_backward(
                x if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor,
                b if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor,
                y if 'y' in spec.ref else _null_tensor)
            return y


        def grad(self, dy): # pylint: disable=arguments-differ
            # dy = dy.contiguous(memory_format=self.memory_format)
            x, b, y = self.saved_tensors
            dx = None
            db = None

            # if self.needs_input_grad[0] or self.needs_input_grad[1]:
            #     dx = dy
            #     if act != 'linear' or gain != 1 or clamp >= 0:
            #         dx = BiasActCudaGrad.apply(dy, x, b, y)

            # if self.needs_input_grad[1]:
            #     db = dx.sum([i for i in range(dx.ndim) if i != dim])
                    # 计算 dx（如果 x 或 b 需要梯度）
            if x.requires_grad or b.requires_grad:
                dx = dy
                if act != 'linear' or gain != 1 or clamp >= 0:
                    # 假设 BiasActCudaGrad 是一个自定义反向计算
                    dx = BiasActCudaGrad.apply(dy, x, b, y)
            
            # 计算 db（如果 b 需要梯度）
            if b.requires_grad:
                # 沿着除了 dim 以外的维度求和
                reduce_dims = [i for i in range(dx.ndim) if i != dim]
                db = dx.sum(reduce_dims)

            return dx, db

    # Backward op.
    class BiasActCudaGrad(jt.Function):
        def save_for_backward(self,*args):
            self.saved_tensors = args
            

        def execute(self, dy, x, b, y): # pylint: disable=arguments-differ
            # self.memory_format = torch.channels_last if dy.ndim > 2 and dy.stride(1) == 1 else torch.contiguous_format
            dx = bias_act_ops(dy, b, x, y, _null_tensor, 1, dim, spec.cuda_idx, alpha, gain, clamp)
            self.save_for_backward(
                dy if spec.has_2nd_grad else _null_tensor,
                x, b, y)
            return dx


        def grad(self, d_dx): # pylint: disable=arguments-differ
            # d_dx = d_dx.contiguous(memory_format=self.memory_format)
            dy, x, b, y = self.saved_tensors
            d_dy = None
            d_x = None
            d_b = None
            d_y = None

            if dy.requires_grad:
                d_dy = BiasActCudaGrad.apply(d_dx, x, b, y)

            if spec.has_2nd_grad and (x.requires_grad or y.requires_grad):
                d_x = bias_act_ops(d_dx, b, x, y, dy, 2, dim, spec.cuda_idx, alpha, gain, clamp)

            if spec.has_2nd_grad and b.requires_grad:
                d_b = d_x.sum([i for i in range(d_x.ndim) if i != dim])

            return d_dy, d_x, d_b, d_y

    # Add to cache.
    _bias_act_cuda_cache[key] = BiasActCuda
    return BiasActCuda

#----------------------------------------------------------------------------
def has_same_layout(x, y):
    if x.ndim != y.ndim:
        return False
    for i in range(x.ndim):
        if x.shape[i] != y.shape[i]:
            return False
        # if x.shape[i] >= 2 and x.stride()[i] != y.stride()[i]:
        #     return False
    return True

def bias_act_ops(x, b, xref, yref, dy, grad, dim, act, alpha, gain, clamp):
    # print(x.shape, b.shape, xref.shape, yref.shape, dy.shape, grad, dim, act, alpha, gain, clamp)
    # # print(x)
    # print(b)
    # print(xref)
    # print(yref)
    # print(dy)
    y = jt.zeros_like(x)
    with jt.flag_scope(compile_options=proj_options):
        (y,) = jt.code(
                        inputs=[x, b, xref, yref, dy],
                        outputs=[y],
                        data={
                                'dim':dim,
                                'act':act,
                                'alpha':alpha,
                                'gain':gain,
                                'clamp':clamp,
                                'grad':grad,
                            },
                        cuda_header=cuda_header,
                        cuda_src='''
                            // @alias(x, in0)
                            // @alias(b, in1)
                            // @alias(xref, in2)
                            // @alias(yref, in3)
                            // @alias(dy, in4)
                            // @alias(y, out0)

                            bias_act_kernel_params p;
                            p.x     = in0_p;
                            p.b     = (in1->numel()) ? in1_p : nullptr;
                            p.xref  = (in2->numel()) ? in2_p : nullptr;
                            p.yref  = (in3->numel()) ? in3_p : nullptr;
                            p.dy    = (in4->numel()) ? in4_p : nullptr;

                            //printf("numel:%d",in0->shape[2]);

                            p.y     = out0_p;
                            p.grad  = data["grad"];
                            p.act   = data["act"];
                            p.alpha = data["alpha"];
                            p.gain  = data["gain"];
                            p.clamp = data["clamp"];
                            p.sizeX = (int)in0->numel();
                            p.sizeB = (int)in1->numel();
                            int tmpstride = 1;  // 最后一维的stride是1
                            // 从倒数第二维开始向前计算
                            for (int i = 0; i <= data["dim"]; ++i) {
                                tmpstride = tmpstride * in0->shape[i];
                            }
                            tmpstride = (in0->numel())/(tmpstride);
                            p.stepB = (in1->numel()) ? tmpstride : 1;

                            void* kernel;
                            kernel = choose_bias_act_kernel<float>(p);
                            
                            p.loopX = 4;
                            int blockSize = 4 * 32;
                            int gridSize = (p.sizeX - 1) / (p.loopX * blockSize) + 1;
                            void* args[] = {&p};
                            cudaLaunchKernel(kernel, gridSize, blockSize, args, 0, get_current_cuda_stream());
                        '''
                    )
    jt.sync()
    # print("bias_act",y)
    return y


if __name__ == "__main__":
    x = jt.rand([4,4,4]).float()
    b = jt.rand([4]).float()
    # y = bias_act(x,b=b,act='sigmoid',impl='ref')
    # print(y)
    y = bias_act(x,b=b,act='sigmoid')
    print(y)