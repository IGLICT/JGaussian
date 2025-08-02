# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Fused multiply-add, with slightly faster gradients than `jt.addcmul()`."""

import jittor as jt

#----------------------------------------------------------------------------

def fma(a, b, c): # => a * b + c
    return _FusedMultiplyAdd.apply(a, b, c)

#----------------------------------------------------------------------------

class _FusedMultiplyAdd(jt.Function): # a * b + c
    def save_for_backward(self,*args):
        self.saved_tensors = args
    
    def execute(self, a, b, c): # pylint: disable=arguments-differ
        out = c + a * b
        # out = jt.addcmul(c, a, b)
        self.save_for_backward(a, b, c)
        self.c_shape = c.shape
        return out

    
    def backward(self, dout): # pylint: disable=arguments-differ
        a, b, c = self.saved_tensors
        c_shape = self.c_shape
        da = None
        db = None
        dc = None

        if a.requires_grad:
            da = _unbroadcast(dout * b, a.shape)

        if b.requires_grad:
            db = _unbroadcast(dout * a, b.shape)

        if c.requires_grad:
            dc = _unbroadcast(dout, c_shape)

        return da, db, dc

#----------------------------------------------------------------------------

def _unbroadcast(x, shape):
    extra_dims = x.ndim - len(shape)
    assert extra_dims >= 0
    dim = [i for i in range(x.ndim) if x.shape[i] > 1 and (i < extra_dims or shape[i - extra_dims] == 1)]
    if len(dim):
        x = x.sum(dim=dim, keepdim=True)
    if extra_dims:
        x = x.reshape(-1, *x.shape[extra_dims+1:])
    assert x.shape == shape
    return x

#----------------------------------------------------------------------------
