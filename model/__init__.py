from model.asg import ASGKernelBank


def build_kernel_bank(args, **kwargs):
    return ASGKernelBank(args, **kwargs)


__all__ = ["ASGKernelBank", "build_kernel_bank"]
