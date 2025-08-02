# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re
import contextlib
import numpy as np
import jittor as jt
import warnings
from pathlib import Path
import sys
parent_dir = str(Path(__file__).parent.parent.absolute())
sys.path.append(parent_dir)
import dnnlib
#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = jt.float32  # Jittor 默认浮点类型是 float32
    # Jittor 不直接支持 memory_format 参数，所以省略
    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        # 转换为 Jittor Tensor
        tensor = jt.array(value.copy(), dtype=dtype)
        # 广播到指定形状
        if shape is not None:
            tensor = jt.broadcast(tensor, shape)     
        # Jittor 默认是连续内存布局，无需额外处理
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):
        """
        Replace NaN, infinity, and -infinity values in input tensor.
        
        Args:
            input (jt.Var): Input tensor
            nan (float): Value to replace NaN with (default 0.0)
            posinf (float): Value to replace positive infinity with
            neginf (float): Value to replace negative infinity with
            out (jt.Var, optional): Output tensor
        
        Returns:
            jt.Var: Tensor with replacements applied
        """
        assert isinstance(input, jt.Var)
        
        # Get default infinity values if not specified
        if posinf is None:
            posinf = jt.float_info(input.dtype).max
        if neginf is None:
            neginf = jt.float_info(input.dtype).min
        
        # For Jittor implementation, we'll handle each case separately
        output = input.clone()
        
        # Replace NaN
        if nan != 0:
            output = jt.ternary(input.isnan(), jt.array(nan), output)
        
        # Replace positive infinity
        output = jt.ternary(input.isinf() & (input > 0), jt.array(posinf), output)
        # Replace negative infinity
        output = jt.ternary(input.isinf() & (input < 0), jt.array(neginf), output)
        
        if out is not None:
            out.assign(output)
            return out
        return output

#----------------------------------------------------------------------------
# Symbolic assert.

# try:
#     symbolic_assert = jt._assert # 1.8.0a0 # pylint: disable=protected-access
# except AttributeError:
#     symbolic_assert = jt.Assert # 1.7.0

#----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in jt.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672

@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, jt.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

#----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in jt.jit.trace().

# def assert_shape(tensor, ref_shape):
#     if tensor.ndim != len(ref_shape):
#         raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
#     for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
#         if ref_size is None:
#             pass
#         elif isinstance(ref_size, jt.Tensor):
#             with suppress_tracer_warnings(): # as_tensor results are registered as constants
#                 symbolic_assert(jt.equal(jt.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
#         elif isinstance(size, jt.Tensor):
#             with suppress_tracer_warnings(): # as_tensor results are registered as constants
#                 symbolic_assert(jt.equal(size, jt.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
#         elif size != ref_size:
#             raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

#----------------------------------------------------------------------------
# Function decorator that calls jt.autograd.profiler.record_function().

def profiled_function(fn):
    def decorator(*args, **kwargs):
        # with jt.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator

#----------------------------------------------------------------------------
# Sampler for jt.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteSampler(jt.dataset.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

#----------------------------------------------------------------------------
# Utilities for operating with jt.nn.Module parameters and buffers.

def params_and_buffers(module):
    assert isinstance(module, jt.nn.Module)
    # return list(module.parameters()) + list(module.buffers())
    return list(module.parameters())

def named_params_and_buffers(module):
    assert isinstance(module, jt.nn.Module)
    # return list(module.named_parameters()) + list(module.named_buffers())
    return list(module.named_parameters())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, jt.nn.Module)
    assert isinstance(dst_module, jt.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.update(src_tensors[name].requires_grad_(tensor.requires_grad))
            # tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)

#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, jt.nn.Module)
    if sync or not isinstance(module, jt.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

#----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, jt.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = nan_to_num(tensor)
        other = tensor.clone()
        jt.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname

#----------------------------------------------------------------------------
# Print summary table of module hierarchy.

def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, jt.nn.Module)
    # assert not isinstance(module, jt.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, jt.Var)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_pre_forward_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        # e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        # e.unique_buffers = []
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs

#----------------------------------------------------------------------------
