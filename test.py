from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import flax.serialization
import joblib
import jax
import numpy as np
import pandas as pd
from scipy.io import loadmat

from spatiotemporal import BayesianNeuralFieldMAP


# =========================
# Dataset construction
# =========================

def default_reference_points(spatial_x: int, spatial_y: int) -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, spatial_y - 1),
        (spatial_x - 1, 0),
        (spatial_x - 1, spatial_y - 1),
    ]


def build_columns(num_reference_points: int, include_target: bool = True) -> list[str]:
    columns = ["time", "freq", "target_x", "target_y"]
    for i in range(num_reference_points):
        columns.extend([f"ref_signal_{i}", f"ref_dx_{i}", f"ref_dy_{i}"])
    if include_target:
        columns.append("target")
    return columns


def create_dataset(
    data: np.ndarray,
    reference_points: list[tuple[int, int]],
    target_points: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:

    spatial_x, spatial_y, freq_bands, time_steps = data.shape

    if target_points is None:
        target_points = [(x, y) for x in range(spatial_x) for y in range(spatial_y)]

    num_refs = len(reference_points)
    num_rows = len(target_points) * time_steps * freq_bands
    num_cols = 4 + 3 * num_refs + 1
    rows = np.empty((num_rows, num_cols), dtype=np.float32)

    row_id = 0
    for target_x, target_y in target_points:
        for t in range(time_steps):
            for f in range(freq_bands):
                row = [t, f, target_x, target_y]
                for ref_x, ref_y in reference_points:
                    row.extend([
                        data[ref_x, ref_y, f, t],
                        target_x - ref_x,
                        target_y - ref_y,
                    ])
                row.append(data[target_x, target_y, f, t])
                rows[row_id] = row
                row_id += 1

    return pd.DataFrame(rows, columns=build_columns(num_refs, include_target=True))


def split_by_time(
    df: pd.DataFrame,
    time_steps: int,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    train_steps = max(1, int(round(time_steps * train_ratio)))
    df_train = df[df["time"] < train_steps].copy()
    df_test = df[df["time"] >= train_steps].copy()
    return df_train, df_test, train_steps


def fit_signal_normalizer(
    df_train: pd.DataFrame,
    num_reference_points: int,
) -> dict[str, tuple[float, float, float]]:
    signal_cols = [f"ref_signal_{i}" for i in range(num_reference_points)]
    signal_cols.append("target")

    signal_min = float(df_train[signal_cols].min().min())
    signal_max = float(df_train[signal_cols].max().max())
    signal_range = signal_max - signal_min
    if signal_range < 1e-6:
        signal_range = 1.0

    return {col: (signal_min, signal_max, signal_range) for col in signal_cols}


def apply_normalizer(
    df: pd.DataFrame,
    norm_params: dict[str, tuple[float, float, float]],
) -> pd.DataFrame:
    df = df.copy()
    for col, (min_val, _, range_val) in norm_params.items():
        if col in df.columns:
            df[col] = (df[col] - min_val) / range_val
    return df


def inverse_target_normalization(
    y: np.ndarray,
    norm_params: dict[str, tuple[float, float, float]],
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    target_min, _, target_range = norm_params["target"]
    return y * target_range + target_min

def load_mat_data(data_path: Path, data_key: str | None) -> tuple[np.ndarray, str]:
    mat = loadmat(data_path)

    if data_key is not None and data_key in mat:
        key = data_key
    else:
        key = next(k for k in mat.keys() if not k.startswith("__"))

    data = np.asarray(mat[key])
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, but got shape {data.shape} from key '{key}'.")

    return data, key


def load_metadata(model_dir: Path, model_name: str) -> dict[str, Any]:
    metadata_path = model_dir / f"{model_name}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_model_config(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)

    if "interactions" in config and config["interactions"] is not None:
        config["interactions"] = [tuple(x) for x in config["interactions"]]

    if "target_coord_indices" in config and config["target_coord_indices"] is not None:
        config["target_coord_indices"] = tuple(config["target_coord_indices"])

    return config


def _maybe_integer_key(key: Any) -> int | None:
    if isinstance(key, int):
        return key
    key_str = str(key)
    if key_str.isdigit():
        return int(key_str)
    match = re.search(r"(\d+)$", key_str)
    if match:
        return int(match.group(1))
    return None


def _restore_nested(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {key: _restore_nested(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return tuple(_restore_nested(x) for x in obj)
    if isinstance(obj, tuple):
        return tuple(_restore_nested(x) for x in obj)
    return obj


def _coerce_top_level_params_to_sequence(params: Any) -> tuple:

    if not isinstance(params, Mapping):
        if isinstance(params, list):
            return tuple(_restore_nested(x) for x in params)
        if isinstance(params, tuple):
            return tuple(_restore_nested(x) for x in params)
        raise TypeError(f"Unsupported parameter container type: {type(params)!r}")

    # Common wrapper form: {"params": ...}
    if set(params.keys()) == {"params"}:
        return _coerce_top_level_params_to_sequence(params["params"])

    keys = list(params.keys())
    numeric_order = []
    for key in keys:
        idx = _maybe_integer_key(key)
        if idx is None:
            numeric_order = []
            break
        numeric_order.append((idx, key))

    if numeric_order:
        numeric_order = sorted(numeric_order, key=lambda x: x[0])
        actual = [x[0] for x in numeric_order]
        expected = list(range(len(numeric_order)))
        if actual == expected:
            return tuple(_restore_nested(params[key]) for _, key in numeric_order)

    return tuple(_restore_nested(params[key]) for key in keys)


def restore_params(model_dir: Path, model_name: str):
    model_path = model_dir / f"{model_name}.flax"
    if not model_path.exists():
        raise FileNotFoundError(f"Model parameter file not found: {model_path}")

    raw = model_path.read_bytes()

    try:
        loaded = flax.serialization.from_bytes(None, raw)
    except Exception:
        loaded = flax.serialization.msgpack_restore(raw)

    params = _coerce_top_level_params_to_sequence(loaded)

    if len(params) < 4:
        raise TypeError(
            f"Loaded parameter sequence is too short: len(params)={len(params)}. "
            "The checkpoint may not be the saved model.params_ from training."
        )

    _ = params[0]
    _ = params[3:]

    first_leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"Loaded model parameters: type=tuple, num_items={len(params)}, first_leaf_shape={getattr(first_leaf, 'shape', None)}")

    return params

def build_loaded_model(
    model_dir: Path,
    model_name: str,
    df_train: pd.DataFrame,
) -> BayesianNeuralFieldMAP:
    metadata = load_metadata(model_dir, model_name)
    if "model_config" not in metadata:
        raise KeyError("metadata does not contain 'model_config'.")

    model_config = clean_model_config(metadata["model_config"])
    model = BayesianNeuralFieldMAP(**model_config)

    _ = model.data_handler.get_train(df_train)

    model.params_ = restore_params(model_dir, model_name)
    return model


# =========================
# Metrics
# =========================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        raise ValueError("No valid finite samples for metric calculation.")

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps)) * 100.0)

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "num_samples": int(y_true.size),
    }


def evaluate(
    model: BayesianNeuralFieldMAP,
    df_test_norm: pd.DataFrame,
    norm_params: dict[str, tuple[float, float, float]],
) -> tuple[dict[str, float], pd.DataFrame]:
    _, yhat_quantiles = model.predict(
        df_test_norm,
        quantiles=(0.5,),
        approximate_quantiles=True,
    )

    y_pred_norm = np.asarray(yhat_quantiles[0]).reshape(-1)
    y_true_norm = df_test_norm["target"].to_numpy(dtype=np.float64).reshape(-1)

    y_pred = inverse_target_normalization(y_pred_norm, norm_params)
    y_true = inverse_target_normalization(y_true_norm, norm_params)

    metrics = regression_metrics(y_true, y_pred)

    pred_df = df_test_norm.copy().reset_index(drop=True)
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["error"] = pred_df["y_pred"] - pred_df["y_true"]
    pred_df["abs_error"] = np.abs(pred_df["error"])

    return metrics, pred_df


# =========================
# Main
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone test script for the trained spatiotemporal BNF model.")
    parser.add_argument("--data_path", type=str, default="new.mat", help="Path to the .mat data file.")
    parser.add_argument("--data_key", type=str, default=None, help="Key in the .mat file. If omitted, the first non-private key is used.")
    parser.add_argument("--model_dir", type=str, default="saved_models", help="Directory containing saved model files.")
    parser.add_argument("--model_name", type=str, default="spatiotemporal_model", help="Saved model name without extension.")
    parser.add_argument("--norm_path", type=str, default="norm_params.joblib", help="Path to saved normalization parameters.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Time-based train/test split ratio used during training.")
    parser.add_argument("--output_dir", type=str, default="test_outputs", help="Directory for test outputs.")
    # Jupyter/IPython automatically injects extra arguments such as "-f kernel.json".
    # parse_known_args() keeps this script usable both as a .py file and inside a notebook.
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    norm_path = Path(args.norm_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Standalone model testing")
    print("=" * 60)

    data, used_key = load_mat_data(data_path, args.data_key)
    spatial_x, spatial_y, freq_bands, time_steps = data.shape
    print(f"Data path : {data_path}")
    print(f"Data key  : {used_key}")
    print(f"Data shape: {data.shape}")

    reference_points = default_reference_points(spatial_x, spatial_y)
    num_reference_points = len(reference_points)
    print(f"Reference points: {reference_points}")

    df = create_dataset(data, reference_points)
    df = df.dropna().reset_index(drop=True)
    df_train_raw, df_test_raw, train_steps = split_by_time(df, time_steps, args.train_ratio)
    print(f"Train time steps: 0-{train_steps - 1}")
    print(f"Test time steps : {train_steps}-{time_steps - 1}")
    print(f"Train samples   : {len(df_train_raw):,}")
    print(f"Test samples    : {len(df_test_raw):,}")

    if norm_path.exists():
        norm_params = joblib.load(norm_path)
        print(f"Loaded normalization parameters: {norm_path}")
    else:
        norm_params = fit_signal_normalizer(df_train_raw, num_reference_points)
        print("norm_params.joblib not found; normalization parameters were refit from the training split.")

    df_train_norm = apply_normalizer(df_train_raw, norm_params)
    df_test_norm = apply_normalizer(df_test_raw, norm_params)

    model = build_loaded_model(model_dir, args.model_name, df_train_norm)

    first_leaf = jax.tree_util.tree_leaves(model.params_)[0]
    if hasattr(first_leaf, "shape") and len(first_leaf.shape) > 0:
        saved_device_dim = int(first_leaf.shape[0])
        current_device_count = int(jax.device_count())
        if saved_device_dim != current_device_count:
            print(
                "Warning: saved parameter device dimension "
                f"({saved_device_dim}) != current jax.device_count() ({current_device_count}). "
                "If prediction fails, test on the same device setting used for training."
            )

    metrics, pred_df = evaluate(model, df_test_norm, norm_params)

    print("\nTest metrics on original signal scale:")
    print(f"MSE         : {metrics['MSE']:.6f}")
    print(f"MAE         : {metrics['MAE']:.6f}")
    print(f"RMSE        : {metrics['RMSE']:.6f}")
    print(f"MAPE(%)     : {metrics['MAPE_percent']:.6f}")
    print(f"num_samples : {metrics['num_samples']}")

    metrics_path = output_dir / "test_metrics.json"
    pred_path = output_dir / "test_predictions.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print("\nSaved outputs:")
    print(f"- {metrics_path}")
    print(f"- {pred_path}")


if __name__ == "__main__":
    main()
