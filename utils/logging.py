import logging
import os
import shutil
import sys


def prepare_run_directories(log_dir, *, overwrite=False):
    if os.path.exists(log_dir):
        if not overwrite:
            raise FileExistsError(f"Run directory already exists: {log_dir}. Use --overwrite_output for intentional replacement.")
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=False)


def create_worklog(log_dir, log_level, *, debug=False):
    """Console logger of a fit; with debug, also written to debug/log.txt."""
    resolved_level = getattr(logging, log_level, logging.INFO)
    worklog = logging.getLogger("spherical-kernels")
    worklog.setLevel(resolved_level)
    worklog.propagate = False
    for handler in worklog.handlers:
        handler.close()
    datefmt = "%Y/%m/%d %H:%M:%S"
    fmt = "\x1b[32m[{asctime}]\x1b[0m {message}" if sys.stdout.isatty() else "[{asctime}] {message}"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt, style="{"))
    worklog.handlers = [console_handler]
    if debug:
        debug_dir = os.path.join(log_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(debug_dir, "log.txt"), encoding="utf8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        worklog.addHandler(file_handler)
    return worklog


def method_name(args):
    """Paper name of the configured variant: ours, asg, p_only or q_only.

    Ablations: beta and ours_unnormalized; SKC codes neither.
    """
    if args.asg_footprint == "gaussian":
        return "asg"
    if args.asg_footprint == "beta":
        return "beta"
    if not args.asg_normalize_shape:
        return "ours_unnormalized"
    return {(True, True): "ours", (True, False): "p_only",
            (False, True): "q_only"}.get((args.asg_learn_p, args.asg_learn_q), "frozen")


def get_log_dir(args):
    stem = os.path.splitext(os.path.basename(args.input_path))[0]
    budget = args.densify_max_gaussians if args.densify_enabled else args.num_gaussians
    return os.path.join(args.output_dir, f"{stem}_{method_name(args)}_{budget}_s{args.seed}")
