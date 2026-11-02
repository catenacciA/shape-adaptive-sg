import dataclasses
import math
import random
from argparse import ArgumentParser, BooleanOptionalAction, Namespace, SUPPRESS
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class ResolvedTrainingConfig:
    """Complete training configuration; the defaults are the paper's "Ours" setup.

    The serialized configuration is the exact configuration consumed by a run.
    Method presets in cfgs/ override only the footprint and shape flags.
    """

    seed: int = 123
    device: str = "cuda:0"
    output_dir: str = "results/fits"
    data_root: str = "data"
    input_path: str = "octa/artist_workshop_octa_1024.exr"
    envmap_projection: str = "octahedral"
    latlong_reference: str | None = None
    overwrite_output: bool = False
    debug_artifacts: bool = False
    log_level: str = "INFO"

    num_gaussians: int = 3500
    densify_max_gaussians: int | None = 10000
    allocation_uniform_fraction: float = 0.10
    allocation_clip_quantile: float = 0.995
    anisotropy_init_ratio_max: float = 4.0
    kernel_width_ratio_max: float = 8.0

    asg_lambda_min: float = 5.0
    asg_footprint: str = "superelliptic"
    asg_learn_p: bool = True
    asg_learn_q: bool = True
    # Normalization ablation: false sets b_pq = 1 and c_pq = 1/pi for every (p, q).
    asg_normalize_shape: bool = True
    lr_asg_shape: float = 1.0e-3
    asg_lambda_max: float = 1.0e6
    asg_lambda_max_start: float = 5.0e4
    asg_lambda_warmup_steps: int = 1000
    lambda_asg_concentration: float = 1.0e-5

    num_steps: int = 10000
    geodesic_subdivision: Any = "auto"
    evaluation_direction_chunk_size: int = 8192
    evaluation_subdivision: int | None = 9
    hdr_scale: float = 0.0
    tonemap_exposures: str = "-2,0,2"
    tonemap_gamma: float = 2.2
    charbonnier_eps: float = 1.0e-3
    lr_mu: float = 8.0e-3
    lr_sigma: float = 4.0e-3
    lr_theta: float = 8.0e-3
    lr_coeff: float = 2.0e-2
    log_every: int = 1000
    batch_size: int = 8192
    grad_accum_steps: int = 2
    grad_clip_norm: float = 1.0
    lr_decay_start: float = 0.5
    lr_decay_end: float = 0.1
    renderer_chunk_size: int = 4000

    densify_enabled: bool = True
    densify_start_step: int = 1000
    densify_interval: int = 1000
    densify_stop_fraction: float = 0.65
    densify_add_fraction: float = 0.15
    densify_direction_chunk_size: int = 8192
    densify_min_linear_residual: float = 1.0e-6
    densify_min_relative_residual: float = 1.0e-4
    densify_candidate_oversample: int = 4
    densify_candidate_nms_radius_scale: float = 0.75
    densify_center_max_width_scale: float = 0.50

    log_dir: str | None = None


TRAINING_CONFIG_KEYS = frozenset(field.name for field in dataclasses.fields(ResolvedTrainingConfig))


def _require_finite(name, value, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def resolve_training_config(values):
    """Resolve defaults, reject unknown keys, and validate every value."""
    if isinstance(values, Namespace):
        values = vars(values)
    elif not isinstance(values, dict):
        values = vars(values)
    unknown = set(values) - TRAINING_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown training configuration keys: {', '.join(sorted(unknown))}")
    cfg = ResolvedTrainingConfig(**values)

    for name in ("device", "output_dir", "data_root", "input_path", "log_level"):
        value = getattr(cfg, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    for name in ("envmap_projection", "asg_footprint", "tonemap_exposures"):
        if not isinstance(getattr(cfg, name), str):
            raise ValueError(f"{name} must be a string")
    if cfg.envmap_projection not in {"latlong", "octahedral"}:
        raise ValueError("envmap_projection must be 'latlong' or 'octahedral'")
    if cfg.asg_footprint not in {"gaussian", "superelliptic", "beta"}:
        raise ValueError("asg_footprint must be 'gaussian', 'superelliptic' or 'beta'")
    if cfg.latlong_reference is not None and (
        not isinstance(cfg.latlong_reference, str) or not cfg.latlong_reference.strip()
    ):
        raise ValueError("latlong_reference must be null or a non-empty string")
    if cfg.log_dir is not None and (
        not isinstance(cfg.log_dir, str) or not cfg.log_dir.strip()
    ):
        raise ValueError("log_dir must be null or a non-empty string")
    if cfg.log_level.upper() not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise ValueError("log_level must name a standard Python logging level")

    if cfg.densify_enabled:
        if cfg.densify_max_gaussians is None:
            raise ValueError(
                "densify_max_gaussians is required when densification is enabled"
            )
        if (
            isinstance(cfg.densify_max_gaussians, bool)
            or not isinstance(cfg.densify_max_gaussians, int)
            or cfg.densify_max_gaussians <= 0
        ):
            raise ValueError("densify_max_gaussians must be a positive integer")
        if cfg.densify_max_gaussians < cfg.num_gaussians:
            raise ValueError(
                "densify_max_gaussians must be at least num_gaussians"
            )
    elif cfg.densify_max_gaussians is not None:
        raise ValueError(
            "densify_max_gaussians must be null when densification is disabled"
        )

    for name in (
        "overwrite_output",
        "debug_artifacts",
        "densify_enabled",
        "asg_learn_p",
        "asg_learn_q",
        "asg_normalize_shape",
    ):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{name} must be boolean")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(cfg.geodesic_subdivision, str):
        if cfg.geodesic_subdivision.strip().lower() != "auto":
            try:
                geodesic_value = int(cfg.geodesic_subdivision)
            except ValueError as exc:
                raise ValueError("geodesic_subdivision must be 'auto' or non-negative") from exc
            if geodesic_value < 0:
                raise ValueError("geodesic_subdivision must be non-negative")
    elif (
        isinstance(cfg.geodesic_subdivision, bool)
        or not isinstance(cfg.geodesic_subdivision, int)
        or cfg.geodesic_subdivision < 0
    ):
        raise ValueError("geodesic_subdivision must be 'auto' or non-negative")
    for name in (
        "num_gaussians",
        "num_steps",
        "log_every",
        "batch_size",
        "grad_accum_steps",
        "renderer_chunk_size",
        "evaluation_direction_chunk_size",
    ):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg.evaluation_subdivision is not None and (type(cfg.evaluation_subdivision) is not int or cfg.evaluation_subdivision < 0):
        raise ValueError("evaluation_subdivision must be null or a non-negative integer")
    for name in ("asg_lambda_warmup_steps",):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in (
        "densify_start_step",
        "densify_interval",
        "densify_direction_chunk_size",
        "densify_candidate_oversample",
    ):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "allocation_uniform_fraction",
        "densify_stop_fraction",
        "lr_decay_start",
        "lr_decay_end",
    ):
        _require_finite(name, getattr(cfg, name), minimum=0.0, maximum=1.0)
    _require_finite(
        "allocation_clip_quantile",
        cfg.allocation_clip_quantile,
        minimum=0.0,
        maximum=1.0,
    )
    if cfg.allocation_clip_quantile <= 0.0:
        raise ValueError("allocation_clip_quantile must be in (0, 1]")
    _require_finite("densify_add_fraction", cfg.densify_add_fraction, minimum=0.0, maximum=1.0)
    for name in (
        "lr_mu", "lr_sigma", "lr_theta", "lr_coeff", "lr_asg_shape",
        "grad_clip_norm", "tonemap_gamma", "charbonnier_eps",
    ):
        _require_finite(name, getattr(cfg, name), minimum=0.0)
        if getattr(cfg, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    _require_finite("hdr_scale", cfg.hdr_scale, minimum=0.0)
    for name in (
        "asg_lambda_min", "asg_lambda_max", "asg_lambda_max_start",
        "kernel_width_ratio_max", "anisotropy_init_ratio_max",
    ):
        _require_finite(name, getattr(cfg, name), minimum=0.0)
        if getattr(cfg, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "lambda_asg_concentration",
        "densify_min_linear_residual", "densify_min_relative_residual",
        "densify_candidate_nms_radius_scale",
        "densify_center_max_width_scale",
    ):
        _require_finite(name, getattr(cfg, name), minimum=0.0)
    if cfg.asg_lambda_min <= 0.0 or cfg.asg_lambda_max < cfg.asg_lambda_min:
        raise ValueError("ASG concentration bounds must be positive and ordered")
    if cfg.kernel_width_ratio_max < 1.0 or cfg.anisotropy_init_ratio_max < 1.0:
        raise ValueError("kernel anisotropy ratios must be at least one")
    if not cfg.asg_lambda_min <= cfg.asg_lambda_max_start <= cfg.asg_lambda_max:
        raise ValueError("asg_lambda_max_start must lie within the ASG bounds")
    try:
        exposures = tuple(
            float(part.strip())
            for part in cfg.tonemap_exposures.split(",")
            if part.strip()
        )
    except (AttributeError, ValueError) as exc:
        raise ValueError("tonemap_exposures must be a comma-separated numeric string") from exc
    if not exposures or not all(math.isfinite(value) for value in exposures):
        raise ValueError("tonemap_exposures must contain finite numeric values")
    return cfg


def config_as_dict(config):
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return dict(vars(config))


def load_yaml_mapping(path):
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def training_config_defaults():
    return dataclasses.asdict(ResolvedTrainingConfig())


def load_training_profile(path):
    overrides = load_yaml_mapping(path)
    unknown = set(overrides) - TRAINING_CONFIG_KEYS
    if unknown:
        raise ValueError(
            "unknown public training configuration keys: "
            + ", ".join(sorted(unknown))
        )
    values = training_config_defaults()
    values.update(overrides)
    return values


def _nullable_parser(value_type):
    def parse(text):
        if str(text).strip().lower() in {"none", "null"}:
            return None
        return value_type(text)

    return parse


def _parse_geodesic_subdivision(text):
    value = str(text).strip().lower()
    if value == "auto":
        return "auto"
    return int(value)


def _training_argument_type(name, default):
    if name == "geodesic_subdivision":
        return _parse_geodesic_subdivision
    if name in {"densify_max_gaussians", "evaluation_subdivision"}:
        return _nullable_parser(int)
    if name == "log_dir":
        return _nullable_parser(str)
    if isinstance(default, int) and not isinstance(default, bool):
        return int
    if isinstance(default, float):
        return float
    return str


def add_public_training_arguments(parser, *, defaults=None, suppress_defaults=False):
    defaults = training_config_defaults() if defaults is None else dict(defaults)
    for name in sorted(TRAINING_CONFIG_KEYS - {"latlong_reference"}):  # parse_training_cli adds it
        default = defaults[name]
        argument_default = SUPPRESS if suppress_defaults else default
        if isinstance(default, bool):
            parser.add_argument(
                f"--{name}",
                action=BooleanOptionalAction,
                default=argument_default,
            )
        else:
            parser.add_argument(
                f"--{name}",
                type=_training_argument_type(name, default),
                default=argument_default,
            )
    return parser


def budget_overrides(budget):
    """Paper budget rule: start from 35% of the primitives and insert up to the budget."""
    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be a positive integer")
    return {"num_gaussians": math.ceil(0.35 * budget), "densify_max_gaussians": budget,
            "densify_enabled": True}


def parse_training_cli(argv=None, *, default_config="cfgs/ours.yaml"):
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=default_config)
    pre_args, _ = pre_parser.parse_known_args(argv)
    values = load_training_profile(pre_args.config)

    parser = ArgumentParser(
        usage="%(prog)s input [--config CONFIG] [--budget N] [--latlong-reference MAP] [method settings ...]",
        description="Fit shape-adaptive spherical Gaussians to an HDR environment map, then "
                    "evaluate, encode and render the fit into results/fits/<run>/.",
    )
    parser.add_argument("input", nargs="?", type=Path,
                        help="HDR map (.exr or .hdr): square octahedral or 2:1 lat-long; "
                             "the projection is detected unless --envmap_projection is given")
    parser.add_argument("--config", default=pre_args.config,
                        help="method preset: cfgs/ours.yaml, asg.yaml, p_only.yaml, q_only.yaml, "
                             "beta.yaml or ours_unnormalized.yaml")
    parser.add_argument("--budget", type=int,
                        help="primitive budget N: start from ceil(0.35 N) and insert up to N")
    parser.add_argument("--latlong-reference", dest="latlong_reference", default=SUPPRESS,
                        help="original 1024x512 lat-long map for the PU21 image metrics, as in the "
                             "paper (default: the input resampled to lat-long)")
    add_public_training_arguments(
        parser.add_argument_group("method settings (defaults: the paper's; see utils/config.py)"),
        defaults=values,
        suppress_defaults=True,
    )
    parsed = vars(parser.parse_args(argv))
    parsed.pop("config")
    budget = parsed.pop("budget")
    source = parsed.pop("input")
    if source is not None:
        if {"data_root", "input_path"} & parsed.keys():
            parser.error("give the input map either positionally or with --data_root/--input_path")
        parsed.update(data_root=str(source.parent), input_path=source.name)
    if budget is not None:
        values.update(budget_overrides(budget))
    explicit_projection = ("envmap_projection" in parsed
                           or "envmap_projection" in load_yaml_mapping(pre_args.config))
    values.update(parsed)
    if not explicit_projection and {"data_root", "input_path"} & parsed.keys():
        from envmap.field import infer_projection
        try:
            values["envmap_projection"] = infer_projection(Path(values["data_root"], values["input_path"]))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    return values


def set_random_seed(seed):  # adapted from Image-GS (MIT, see LICENSE)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolved_namespace(values):
    return SimpleNamespace(**config_as_dict(resolve_training_config(values)))


def save_cfg(path: str, args, mode="w"):  # adapted from Image-GS (MIT, see LICENSE)
    with open(path, mode=mode, encoding="utf-8") as file:
        print("# Resolved training configuration", file=file)
        yaml.safe_dump(config_as_dict(args), file, default_flow_style=False, sort_keys=True)
