#!/usr/bin/env python3
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export PaddlePaddle inference models from HuggingFace to ONNX.

Downloads models from the PaddlePaddle HuggingFace organization, converts them
to ONNX using paddle2onnx, applies ORT compatibility fixes, and validates the
output with ONNX Runtime.

Usage:
    python tools/export_hf_models.py --model PaddlePaddle/SLANet_plus --output-dir /tmp/exports
    python tools/export_hf_models.py --all --output-dir /tmp/exports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from huggingface_hub import hf_hub_download
from onnx import helper

import paddle2onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


ort.set_default_logger_severity(3)


@dataclass
class ModelConfig:
    """Configuration for a PaddlePaddle model to export."""

    hf_repo: str
    model_filename: str = "inference.json"
    params_filename: str = "inference.pdiparams"
    opset_version: int = 17
    inputs: dict[str, tuple[tuple[int, ...], str]] | None = None
    upload_path: str | None = None
    # Shape to pin the tract-onnx validation load to (N, C, H, W), or None to
    # validate fully unpinned (batch/H/W symbolic). DBNet-style detection
    # models (PP-OCRv5/v6 *_det) cannot shape-infer under tract with a
    # symbolic H/W: the FPN's Resize-upsampled branch and its skip-connection
    # branch compute the output extent via different Div/MulInt chains over
    # the symbolic dimension, and tract's unifier cannot prove those two
    # symbolic expressions equal (confirmed against the published
    # v6/det/medium model — fails during into_optimized() at the FPN concat
    # with "Impossible to unify MulInt(8, ...)"). This is not an export bug to
    # fix: the consuming OCR pipeline pins detection models to a fixed square
    # canvas for exactly this reason, so validating with a pin here reflects
    # real usage. Do not remove this field to "simplify" the config —
    # recognizers and classifiers must stay on the stricter unpinned check,
    # since an unpinned failure there (e.g. the v6 rec height-clobbering
    # regression) is exactly the class of bug this validation exists to catch.
    tract_validation_shape: tuple[int, int, int, int] | None = None


_DET_TRACT_VALIDATION_SHAPE = (1, 3, 960, 960)


MODELS: dict[str, ModelConfig] = {
    "PP-OCRv5_server_det": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv5_server_det",
        tract_validation_shape=_DET_TRACT_VALIDATION_SHAPE,
    ),
    "PP-OCRv5_mobile_det": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv5_mobile_det",
        tract_validation_shape=_DET_TRACT_VALIDATION_SHAPE,
    ),
    "PP-OCRv5_server_rec": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv5_server_rec",
    ),
    "PP-OCRv5_mobile_rec": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv5_mobile_rec",
    ),
    "en_PP-OCRv5_mobile_rec": ModelConfig(
        hf_repo="PaddlePaddle/en_PP-OCRv5_mobile_rec",
    ),
    "PP-OCRv6_medium_det": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv6_medium_det",
        upload_path="v6/det/medium",
        tract_validation_shape=_DET_TRACT_VALIDATION_SHAPE,
    ),
    "PP-OCRv6_small_det": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv6_small_det",
        upload_path="v6/det/small",
        tract_validation_shape=_DET_TRACT_VALIDATION_SHAPE,
    ),
    "PP-OCRv6_tiny_det": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv6_tiny_det",
        upload_path="v6/det/tiny",
        tract_validation_shape=_DET_TRACT_VALIDATION_SHAPE,
    ),
    "PP-OCRv6_medium_rec": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv6_medium_rec",
        upload_path="v6/rec/medium",
    ),
    "PP-OCRv6_small_rec": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv6_small_rec",
        upload_path="v6/rec/small",
    ),
    "PP-OCRv6_tiny_rec": ModelConfig(
        hf_repo="PaddlePaddle/PP-OCRv6_tiny_rec",
        upload_path="v6/rec/tiny",
    ),
    "PP-DocLayoutV3": ModelConfig(
        hf_repo="PaddlePaddle/PP-DocLayoutV3",
        inputs={
            "im_shape": ((1, 2), "float32"),
            "image": ((1, 3, 800, 800), "float32"),
            "scale_factor": ((1, 2), "float32"),
        },
    ),
    "SLANet_plus": ModelConfig(
        hf_repo="PaddlePaddle/SLANet_plus",
    ),
    "SLANeXt_wired": ModelConfig(
        hf_repo="PaddlePaddle/SLANeXt_wired",
    ),
    "SLANeXt_wireless": ModelConfig(
        hf_repo="PaddlePaddle/SLANeXt_wireless",
    ),
    "RT-DETR-L_wired_table_cell_det": ModelConfig(
        hf_repo="PaddlePaddle/RT-DETR-L_wired_table_cell_det",
        inputs={
            "im_shape": ((1, 2), "float32"),
            "image": ((1, 3, 640, 640), "float32"),
            "scale_factor": ((1, 2), "float32"),
        },
    ),
    "RT-DETR-L_wireless_table_cell_det": ModelConfig(
        hf_repo="PaddlePaddle/RT-DETR-L_wireless_table_cell_det",
        inputs={
            "im_shape": ((1, 2), "float32"),
            "image": ((1, 3, 640, 640), "float32"),
            "scale_factor": ((1, 2), "float32"),
        },
    ),
    "PP-LCNet_x1_0_doc_ori": ModelConfig(
        hf_repo="PaddlePaddle/PP-LCNet_x1_0_doc_ori",
    ),
    "PP-LCNet_x1_0_textline_ori": ModelConfig(
        hf_repo="PaddlePaddle/PP-LCNet_x1_0_textline_ori",
    ),
    "PP-LCNet_x1_0_table_cls": ModelConfig(
        hf_repo="PaddlePaddle/PP-LCNet_x1_0_table_cls",
    ),
}


def _download_optional(repo: str, fname: str, target_dir: Path) -> None:
    """Download a best-effort file; a missing file is logged, not raised."""
    try:
        hf_hub_download(repo, fname, local_dir=str(target_dir))
        logger.info("Downloaded %s/%s", repo, fname)
    except Exception:
        logger.debug("No %s in %s", fname, repo)


def download_model(config: ModelConfig, target_dir: Path) -> Path:
    """Download model files from HuggingFace."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname in [config.model_filename, config.params_filename]:
        hf_hub_download(config.hf_repo, fname, local_dir=str(target_dir))
        logger.info("Downloaded %s/%s", config.hf_repo, fname)
    for optional in ("config.json", "inference.yml"):
        _download_optional(config.hf_repo, optional, target_dir)
    return target_dir


def export_to_onnx(config: ModelConfig, model_dir: Path, output_path: Path) -> None:
    """Export a PaddlePaddle model to ONNX."""
    model_file = str(model_dir / config.model_filename)
    params_file = str(model_dir / config.params_filename)

    logger.info("Exporting %s to ONNX (opset %d)...", config.hf_repo, config.opset_version)

    paddle2onnx.export(
        model_filename=model_file,
        params_filename=params_file,
        save_file=str(output_path),
        opset_version=config.opset_version,
        auto_upgrade_opset=False,
        verbose=False,
        enable_onnx_checker=True,
        enable_experimental_op=True,
        enable_optimize=True,
        deploy_backend="onnxruntime",
        calibration_file="",
        external_file="",
        export_fp16_model=False,
        optimize_tool="None",
    )
    logger.info("Exported to %s", output_path)


def _promote_scalar_to_1d(shape_proto, dtype: int) -> bool:
    """Change a scalar shape declaration to [1]. Returns True if changed."""
    dims = [d.dim_value or d.dim_param for d in shape_proto.dim]
    if dims == []:
        shape_proto.CopyFrom(helper.make_tensor_type_proto(dtype, [1]).tensor_type.shape)
        return True
    return False


def fix_ort_compatibility(model_path: Path) -> None:
    """Apply post-export fixes for ORT compatibility.

    Paddle2ONNX exports Loop nodes where the ONNX Loop spec mandates the
    iteration counter as a [1]-shaped tensor, but scalar ([]) shapes are used
    for loop-carried state, conditions, and If branch outputs. ORT's shape
    inference propagates [1] from the iteration counter through comparison and
    logical ops, then rejects the model when it hits a [] declaration.

    Strategy: selectively promote scalar shapes to [1] ONLY at the Loop/If
    boundary (inputs, outputs, graph-level feeds). Do NOT promote body-internal
    Constants, initializers, or value_info — those feed the If condition and
    other internal logic. Promoting them changes Reshape target shapes, breaking
    the autoregressive decoder (garbled token sequences).
    """
    model = onnx.load(str(model_path))
    fixed = False

    for node in model.graph.node:
        if node.op_type != "Loop":
            continue

        loop_inputs = set(node.input)

        for gnode in model.graph.node:
            if gnode.op_type == "Constant" and any(o in loop_inputs for o in gnode.output):
                for a in gnode.attribute:
                    if a.name == "value" and list(a.t.dims) == []:
                        a.t.dims[:] = [1]
                        fixed = True

        for init in model.graph.initializer:
            if init.name in loop_inputs and list(init.dims) == []:
                init.dims[:] = [1]
                fixed = True

        for attr in node.attribute:
            if attr.name != "body":
                continue
            body = attr.g

            for inp in body.input:
                fixed |= _promote_scalar_to_1d(inp.type.tensor_type.shape, inp.type.tensor_type.elem_type)
            for out in body.output:
                fixed |= _promote_scalar_to_1d(out.type.tensor_type.shape, out.type.tensor_type.elem_type)

            for n in body.node:
                if n.op_type == "If":
                    for if_attr in n.attribute:
                        if if_attr.name in ("then_branch", "else_branch"):
                            branch = if_attr.g
                            for out in branch.output:
                                fixed |= _promote_scalar_to_1d(
                                    out.type.tensor_type.shape,
                                    out.type.tensor_type.elem_type,
                                )

    if fixed:
        logger.info("Applied ORT Loop/If selective scalar-to-[1] shape fix")
        onnx.save(model, str(model_path))


# Axis names for the batch, height, and width slots of a 4D NCHW input.
# Plain identifiers (rather than "?"/"None"/"unk__*") because tract-onnx
# treats those specific tokens as an unresolvable unknown dimension (see
# tract-onnx 0.23.4 src/tensor.rs:48-52), while a named dim_param like these
# parses as a proper symbolic dimension it can reason about. This is a
# separate, hand-picked namespace from the "DynamicDimension.N" counter the
# paddle2onnx C++ core emits (paddle2onnx/mapper/onnx_helper.cc:163-171) for
# axes that were already dynamic (-1) in the source Paddle model, so the two
# never collide when both appear on the same graph.
_DYNAMIC_AXIS_NAMES = {0: "N", 2: "H", 3: "W"}


def make_inputs_dynamic(model_path: Path) -> None:
    """Make batch, height, and width dimensions dynamic for 4D image inputs.

    Applies to all 4D (NCHW) inputs in the graph. Non-image inputs like
    im_shape (1,2) and scale_factor (1,2) are naturally skipped since they
    are not 4D.

    A dimension is left untouched if it is already dynamic (has a
    `dim_param`, e.g. a "DynamicDimension.N" name from the C++ core) or if it
    already has a concrete `dim_value` — a fixed dimension is a deliberate
    part of the model contract (e.g. the recognizer's fixed height of 48, or
    a classifier's fixed input resolution) and must never be silently
    destroyed. Only a dimension with neither set (truly unspecified) is
    assigned a symbolic name.
    """
    model = onnx.load(str(model_path))
    changed = False

    for inp in model.graph.input:
        shape = inp.type.tensor_type.shape
        dims = list(shape.dim)
        if len(dims) != 4:
            continue

        for idx, axis_name in _DYNAMIC_AXIS_NAMES.items():
            dim = dims[idx]
            if dim.WhichOneof("value") is not None:
                # Already concrete (dim_value) or already named (dim_param).
                continue
            dim.dim_param = axis_name
            changed = True

    if changed:
        onnx.save(model, str(model_path))
        logger.info("Made image inputs dynamic (batch, height, width)")


def load_character_dict(model_dir: Path) -> list[str] | None:
    """Load the recognition character dict from config.json or inference.yml.

    PP-OCRv5 ships the dict in config.json under PostProcess.character_dict.
    PP-OCRv6 base repos have no config.json and carry the same key in
    inference.yml. config.json takes precedence when present so existing v5
    output stays byte-identical. Returns None for models with no character dict.
    """
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            char_dict = json.load(f).get("PostProcess", {}).get("character_dict")
        if char_dict:
            return char_dict

    yml_path = model_dir / "inference.yml"
    if yml_path.exists():
        with open(yml_path) as f:
            char_dict = yaml.safe_load(f).get("PostProcess", {}).get("character_dict")
        if char_dict:
            return char_dict

    return None


def embed_character_dict(model_dir: Path, onnx_path: Path) -> None:
    """Embed the recognition character dictionary into ONNX model metadata.

    Reads the dict (config.json for v5, inference.yml for v6) and stores it as
    newline-separated text under the metadata key "character", matching the
    RapidOCR convention that downstream consumers read the dict from.
    """
    char_dict = load_character_dict(model_dir)
    if not char_dict:
        return

    model = onnx.load(str(onnx_path))
    meta = model.metadata_props.add()
    meta.key = "character"
    meta.value = "\n".join(char_dict)
    onnx.save(model, str(onnx_path))
    logger.info("Embedded character dictionary (%d chars) into ONNX metadata", len(char_dict))


ONNX_DTYPE_TO_NP = {
    1: "float32",
    2: "uint8",
    3: "int8",
    5: "int16",
    6: "int32",
    7: "int64",
    10: "float16",
    11: "float64",
}


def _build_dummy_inputs(config: ModelConfig, session: ort.InferenceSession) -> dict[str, np.ndarray]:
    """Build dummy input tensors for validation, using config overrides or model metadata."""
    input_feed = {}
    for inp in session.get_inputs():
        if config.inputs and inp.name in config.inputs:
            shape, dtype = config.inputs[inp.name]
        else:
            shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
            dtype = inp.type.replace("tensor(", "").replace(")", "")
        np_dtype = {
            "float": np.float32,
            "double": np.float64,
            "int64": np.int64,
            "int32": np.int32,
        }.get(dtype, getattr(np, dtype, np.float32))
        input_feed[inp.name] = np.random.randn(*shape).astype(np_dtype)
    return input_feed


# tools/tract_check/ is a tiny standalone Rust binary crate (not part of any
# Cargo workspace) that loads an ONNX model through tract-onnx and confirms
# every input dimension resolves. There is no maintained Python binding for
# tract-onnx, so invoking this helper as a subprocess is the simplest robust
# way to run the tract half of dual-engine validation from this pipeline.
_TRACT_CHECK_DIR = Path(__file__).resolve().parent / "tract_check"
_TRACT_CHECK_BIN = _TRACT_CHECK_DIR / "target" / "release" / "tract_check"


def _ensure_tract_check_binary() -> Path:
    """Build (if needed) and return the path to the tract_check helper binary."""
    if _TRACT_CHECK_BIN.exists():
        return _TRACT_CHECK_BIN

    logger.info("Building tract_check helper binary (first run)...")
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=str(_TRACT_CHECK_DIR), check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "cargo not found: tract_check (tools/tract_check/) requires a Rust toolchain "
            "to build the tract-onnx validation helper. Install Rust (https://rustup.rs) "
            "or run `cargo build --release` in tools/tract_check/ manually."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Building tract_check failed: {error}") from error

    return _TRACT_CHECK_BIN


def validate_with_tract(model_path: Path, shape: tuple[int, int, int, int] | None = None) -> None:
    """Confirm the ONNX model loads and shape-resolves under tract-onnx.

    Complements `validate_onnx`'s ONNX Runtime check: an artifact is only
    considered valid once it loads cleanly under BOTH engines. This is what
    would have caught the v6 recognizer height-clobbering regression, since
    ORT tolerates the "?" dim_param token that tract cannot resolve.

    `shape`, when given, pins input 0 to that concrete (N, C, H, W) before
    loading — see `ModelConfig.tract_validation_shape` for why detection
    models need this and everything else must stay on the stricter, fully
    unpinned check (`shape=None`).
    """
    binary = _ensure_tract_check_binary()
    command = [str(binary), str(model_path)]
    if shape is not None:
        command += ["--shape", ",".join(str(dim) for dim in shape)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"tract-onnx failed to load {model_path} (shape={shape}): {result.stderr.strip()}")
    logger.info("tract-onnx load: PASSED (shape=%s)", shape or "unpinned")


def validate_onnx(config: ModelConfig, model_path: Path) -> dict:
    """Validate ONNX model with checker, ORT inference, and tract-onnx loading."""
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    logger.info("ONNX checker: PASSED")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    input_feed = _build_dummy_inputs(config, session)
    results = session.run(None, input_feed)

    outputs = []
    for i, r in enumerate(results):
        logger.info("Output %d: shape=%s dtype=%s", i, r.shape, r.dtype)
        outputs.append({"shape": list(r.shape), "dtype": str(r.dtype)})

    logger.info("ORT inference: PASSED")

    validate_with_tract(model_path, config.tract_validation_shape)

    with open(model_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    size = model_path.stat().st_size

    return {
        "model": config.hf_repo,
        "opset_version": model.opset_import[0].version,
        "nodes": len(model.graph.node),
        "sha256": sha256,
        "size_bytes": size,
        "outputs": outputs,
        "inputs": [
            {
                "name": i.name,
                "shape": [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim],
            }
            for i in model.graph.input
        ],
    }


def export_model(name: str, config: ModelConfig, output_dir: Path) -> dict | None:
    """Download, export, fix, and validate a single model."""
    logger.info("=" * 60)
    logger.info("Processing: %s (%s)", name, config.hf_repo)
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "paddle"
        download_model(config, model_dir)

        onnx_path = output_dir / f"{name}.onnx"
        export_to_onnx(config, model_dir, onnx_path)

        fix_ort_compatibility(onnx_path)
        make_inputs_dynamic(onnx_path)
        embed_character_dict(model_dir, onnx_path)

    info = validate_onnx(config, onnx_path)
    logger.info("SHA256: %s", info["sha256"])
    logger.info("Size: %d bytes (%.2f MB)", info["size_bytes"], info["size_bytes"] / 1024 / 1024)

    return info


def _try_export(name: str, config: ModelConfig, output_dir: Path) -> dict | None:
    """Attempt to export a model, returning None on failure."""
    try:
        return export_model(name, config, output_dir)
    except Exception:
        logger.exception("Failed to export %s", name)
        return None


def main():
    parser = argparse.ArgumentParser(description="Export PaddlePaddle HF models to ONNX")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=str,
        help="Model name (e.g. SLANet_plus) or HF repo (e.g. PaddlePaddle/SLANet_plus)",
    )
    group.add_argument(
        "--models",
        type=str,
        help="Comma-separated model names to export in one run (shared manifest)",
    )
    group.add_argument("--all", action="store_true", help="Export all configured models")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for ONNX files")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to write JSON manifest of exported models",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List models that would be exported without actually exporting",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_export: dict[str, ModelConfig] = {}
    if args.all:
        models_to_export = MODELS
    elif args.models:
        keys = [name.split("/")[-1] for name in args.models.split(",") if name.strip()]
        unknown = [key for key in keys if key not in MODELS]
        if unknown:
            logger.error("Unknown models: %s. Available: %s", unknown, list(MODELS.keys()))
            sys.exit(1)
        models_to_export = {key: MODELS[key] for key in keys}
    else:
        model_key = args.model.split("/")[-1] if "/" in args.model else args.model
        if model_key not in MODELS:
            logger.error("Unknown model: %s. Available: %s", model_key, list(MODELS.keys()))
            sys.exit(1)
        models_to_export = {model_key: MODELS[model_key]}

    if args.dry_run:
        logger.info("Dry run — would export %d models:", len(models_to_export))
        for name, config in models_to_export.items():
            logger.info("  %s (%s, opset %d)", name, config.hf_repo, config.opset_version)
        return

    manifest = {}
    failed = []

    for name, config in models_to_export.items():
        info = _try_export(name, config, output_dir)
        if info is not None:
            manifest[name] = info
        else:
            failed.append(name)

    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)

    if failed:
        logger.error("Failed models: %s", failed)
        sys.exit(1)

    logger.info("All exports completed successfully")


if __name__ == "__main__":
    main()
