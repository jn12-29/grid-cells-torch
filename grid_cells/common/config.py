"""Shared config and CLI override helpers."""

import argparse
from types import SimpleNamespace

import yaml


def dict_to_namespace(data: dict) -> SimpleNamespace:
    """Recursively convert dicts to nested namespaces."""
    namespace = SimpleNamespace()
    for key, value in data.items():
        setattr(
            namespace,
            key,
            dict_to_namespace(value) if isinstance(value, dict) else value,
        )
    return namespace


def namespace_to_dict(obj):
    """Recursively convert namespaces back to plain Python containers."""
    if isinstance(obj, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(obj).items()}
    if isinstance(obj, dict):
        return {key: namespace_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [namespace_to_dict(value) for value in obj]
    return obj


def str2bool(value):
    """Parse common textual boolean values for CLI overrides."""
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_config(path: str = "config.yaml") -> SimpleNamespace:
    """Load YAML config and return it as a nested namespace."""
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle)
    return dict_to_namespace(raw)


def apply_namespace_overrides(
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    skip_keys=("config", "data_path", "eval_data_path", "output", "eval_output"),
) -> SimpleNamespace:
    """Apply non-None CLI overrides to nested config namespaces in place."""
    for key, value in vars(args).items():
        if key in skip_keys or value is None:
            continue
        if "__" not in key:
            continue
        section, attr = key.split("__", 1)
        target = getattr(cfg, section, None)
        if target is None:
            raise AttributeError(f"Unknown config section: {section}")
        if not hasattr(target, attr):
            raise AttributeError(f"Unknown config field: {section}.{attr}")
        setattr(target, attr, value)
    return cfg
