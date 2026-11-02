"""Kernel fusion for the elementwise graphs that dominate fitting time."""

import functools

import torch


def _plain(value):
    # Dynamo keeps parameter shapes static; a plain view (no copy, same
    # autograd graph) keeps the kernel count dynamic across insertions.
    return value.view_as(value) if isinstance(value, torch.nn.Parameter) else value


def fused_on_cuda(fn=None, *, training_only=False):
    """Run fn through torch.compile when its first argument is on CUDA.

    Compilation fuses long elementwise chains (forward and backward) into a few
    kernels. Where a different formulation fuses better, fn branches on
    torch.compiler.is_compiling(); both branches compute the same quantities
    up to floating-point rounding. CPU inputs run fn eagerly, so the CPU path
    is the unmodified reference.
    training_only also runs eagerly while autograd is disabled: it suits
    functions that matter only per training step, and bounds the number of
    compiled variants. TORCHDYNAMO_DISABLE=1 makes CUDA run eagerly too.
    """
    if fn is None:
        return functools.partial(fused_on_cuda, training_only=training_only)
    compiled = None

    @functools.wraps(fn)
    def dispatch(*args, **kwargs):
        nonlocal compiled
        if not args[0].is_cuda or (training_only and not torch.is_grad_enabled()):
            return fn(*args, **kwargs)
        if compiled is None:
            # Sample and kernel counts change with chunking and insertion.
            compiled = torch.compile(fn, dynamic=True)
        return compiled(*map(_plain, args), **kwargs)

    return dispatch
