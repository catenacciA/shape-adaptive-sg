from pathlib import Path

from fit.evaluation import load_latlong_raster
from fit.metrics import load_lpips
from fit.trainer import EnvmapTrainer
from utils.config import parse_training_cli, resolved_namespace
from utils.logging import get_log_dir


def main(args):
    values = vars(args) if hasattr(args, "__dict__") else dict(args)
    args = resolved_namespace(values)
    source = Path(args.data_root, args.input_path)
    if not source.is_file():  # fail before creating the run directory
        raise FileNotFoundError(f"input map not found: {source}")
    if args.latlong_reference is not None:
        load_latlong_raster(args.latlong_reference)
    load_lpips()  # first use downloads the LPIPS weights: fail before fitting, not after
    if args.log_dir is None:
        args.log_dir = get_log_dir(args)
    EnvmapTrainer(args).optimize()


if __name__ == "__main__":
    main(parse_training_cli())
