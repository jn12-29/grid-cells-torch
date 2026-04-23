"""Shared CLI registration for config-backed section.key overrides."""

from grid_cells.common.config import str2bool


CONFIG_OVERRIDE_SPECS = {
    "task": {
        "env_size": {"type": float},
        "n_pc": {"type": int, "nargs": "+"},
        "pc_scale": {"type": float, "nargs": "+"},
        "n_hdc": {"type": int, "nargs": "+"},
        "hdc_concentration": {"type": float, "nargs": "+"},
        "seq_len": {"type": int},
        "neurons_seed": {"type": int},
        "velocity_noise": {"type": float, "nargs": "+"},
    },
    "model": {
        "nh_lstm": {"type": int},
        "nh_bottleneck": {"type": int},
        "dropout_rate": {"type": float},
        "bottleneck_has_bias": {"type": str2bool},
        "init_weight_disp": {"type": float},
    },
    "training": {
        "epochs": {"type": int},
        "steps_per_epoch": {"type": int},
        "batch_size": {"type": int},
        "data_path": {"type": str},
        "eval_batch_size": {"type": int},
        "eval_data_path": {"type": str},
        "optimizer": {"type": str},
        "lr": {"type": float},
        "momentum": {"type": float},
        "adamw_betas": {"type": float, "nargs": 2},
        "adamw_eps": {"type": float},
        "weight_decay": {"type": float},
        "grad_clip": {"type": float},
        "eval_every": {"type": int},
        "save_dir": {"type": str},
        "run_name": {"type": str},
        "timestamp_save_dir": {"type": str2bool},
        "use_tqdm": {"type": str2bool},
        "use_tensorboard": {"type": str2bool},
        "tensorboard_log_every": {"type": int},
        "eval_loader_batch_size": {"type": int},
        "eval_plot_every": {"type": int},
        "eval_num_workers": {"type": int},
        "eval_chunk_size": {"type": int},
        "eval_units_per_page": {"type": int},
        "eval_pdf_dpi": {"type": int},
    },
    "visualization": {
        "spatial_bins": {"type": int},
        "directional_bins": {"type": int},
        "anim_num_traj": {"type": int},
        "anim_fps": {"type": int},
        "anim_step": {"type": int},
        "anim_workers": {"type": int},
    },
    "data_generation": {
        "num_samples": {"type": int},
        "eval_num_samples": {"type": int},
        "num_workers": {"type": int},
        "progress_every": {"type": int},
        "vis_output": {"type": str},
        "anim_output": {"type": str},
        "progress_output": {"type": str},
        "eval_progress_output": {"type": str},
    },
}


def register_config_overrides(parser, sections=None):
    """Attach shared --section.key override arguments to a parser."""
    for section, fields in CONFIG_OVERRIDE_SPECS.items():
        if sections is not None and section not in sections:
            continue
        for field, spec in fields.items():
            parser.add_argument(
                f"--{section}.{field}",
                dest=f"{section}__{field}",
                default=None,
                **spec,
            )
    return parser
