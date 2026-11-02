"""Adam for row-aligned parameters that may gain freshly initialized rows."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from utils.fusion import fused_on_cuda


@fused_on_cuda
def rowwise_adam_update_(parameter, gradient, exp_avg, exp_avg_sq, row_age, lr, beta1, beta2, eps):
    """One in-place Adam update with per-row bias correction.

    Fused on CUDA, where the per-step update is otherwise bound by small
    kernel launches.
    """
    row_age.add_(1)
    exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

    age = row_age.to(dtype=parameter.dtype)
    # Python scalar bases avoid a synchronizing host-to-device copy.
    bias_correction1 = 1.0 - torch.pow(beta1, age)
    bias_correction2 = 1.0 - torch.pow(beta2, age)
    correction_shape = (parameter.shape[0],) + (1,) * (parameter.ndim - 1)
    bias_correction1 = bias_correction1.reshape(correction_shape)
    bias_correction2 = bias_correction2.reshape(correction_shape)
    denominator = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
    if torch.compiler.is_compiling():
        # The same update as a tensor product keeps the decaying rate
        # symbolic; addcdiv's scalar argument would recompile per value.
        parameter.add_(-lr * (exp_avg / bias_correction1 / denominator))
    else:
        parameter.addcdiv_(exp_avg / bias_correction1, denominator, value=-lr)


class RowwiseAdam(Optimizer):
    """Adam whose bias correction counts steps separately for each row.

    Rows appended at insertion start with zero moments and age, so their first
    updates match a freshly initialized Adam; incumbent rows keep their state.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        lr = float(lr)
        beta1, beta2 = (float(value) for value in betas)
        eps = float(eps)
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError("invalid Adam learning rate")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("invalid Adam beta values")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("invalid Adam epsilon")
        super().__init__(
            params,
            {"lr": lr, "betas": (beta1, beta2), "eps": eps},
        )

    @staticmethod
    def _initialize_state(parameter, state):
        if parameter.ndim < 1:
            raise ValueError("rowwise Adam parameters must have a row dimension")
        state["row_age"] = torch.zeros(
            parameter.shape[0],
            dtype=torch.int64,
            device=parameter.device,
        )
        state["exp_avg"] = torch.zeros_like(
            parameter, memory_format=torch.preserve_format
        )
        state["exp_avg_sq"] = torch.zeros_like(
            parameter, memory_format=torch.preserve_format
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("rowwise Adam does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    self._initialize_state(parameter, state)

                rowwise_adam_update_(
                    parameter, gradient, state["exp_avg"], state["exp_avg_sq"],
                    state["row_age"], group["lr"], beta1, beta2, group["eps"],
                )
        return loss

    @torch.no_grad()
    def append_parameter_rows(self, old_parameter, new_parameter):
        """Preserve incumbent row moments/ages; appended rows start with zero state."""
        if old_parameter.ndim < 1 or new_parameter.ndim < 1:
            raise ValueError("appended parameters must have a row dimension")
        if old_parameter.shape[1:] != new_parameter.shape[1:]:
            raise ValueError("appended parameter trailing shape changed")
        old_rows = old_parameter.shape[0]
        if new_parameter.shape[0] < old_rows:
            raise ValueError("appended parameter lost incumbent rows")

        matches = [(group, index) for group in self.param_groups
                   for index, parameter in enumerate(group["params"])
                   if parameter is old_parameter]
        if len(matches) != 1:
            raise ValueError("appended parameter must occur exactly once in the optimizer")
        group, index = matches[0]
        group["params"][index] = new_parameter

        old_state = self.state.pop(old_parameter, None)
        new_state = self.state[new_parameter]
        self._initialize_state(new_parameter, new_state)
        if not old_state:
            return
        new_state["row_age"][:old_rows].copy_(old_state["row_age"])
        new_state["exp_avg"][:old_rows].copy_(old_state["exp_avg"])
        new_state["exp_avg_sq"][:old_rows].copy_(old_state["exp_avg_sq"])

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for parameter, state in self.state.items():
            if "row_age" in state:
                state["row_age"] = state["row_age"].to(
                    device=parameter.device,
                    dtype=torch.int64,
                )
