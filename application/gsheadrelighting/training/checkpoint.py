"""
Adapted from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L124
"""

from typing import Callable, Iterable, Sequence, Union

import jittor as jt


def checkpoint(
    func: Callable[..., Union[jt.Var, Sequence[jt.Var]]],
    inputs: Sequence[jt.Var],
    params: Iterable[jt.Var],
    flag: bool,
):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(jt.Function):

    def execute(self, run_function, length, *args):
        self.run_function = run_function
        self.input_tensors = list(args[:length])
        self.input_params = list(args[length:])
        with jt.no_grad():
            output_tensors = self.run_function(*self.input_tensors)
        return output_tensors

    
    def grad(self, *output_grads):
        self.input_tensors = [x.detach().requires_grad_(True) for x in self.input_tensors]
        with jt.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Var storage in place, which is not allowed for detach()'d
            # Vars.
            shallow_copies = [x.view_as(x) for x in self.input_tensors]
            output_tensors = self.run_function(*shallow_copies)
        # input_grads = jt.autograd.grad(
        #     output_tensors,
        #     self.input_tensors + self.input_params,
        #     output_grads,
        #     allow_unused=True,
        # )
        input_grads = jt.grad(
            output_tensors,
            self.input_tensors + self.input_params,
            (output_tensors * output_grads).sum(),
        )
        del self.input_tensors
        del self.input_params
        del output_tensors
        return (None, None) + input_grads