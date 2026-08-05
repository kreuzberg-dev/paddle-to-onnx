"""Unit tests for the HuggingFace export tooling character-dict resolution.

Covers the PP-OCRv6 change: v6 base repos have no config.json, so the recognition
character dict must be read from inference.yml, while v5's config.json still wins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

export_hf_models = pytest.importorskip("export_hf_models")
onnx = pytest.importorskip("onnx")


def _make_4d_input(
    name: str,
    *,
    batch_dim_param: str = "",
    height: int | None = 48,
    width_dim_param: str = "",
) -> onnx.ValueInfoProto:
    """Build a 4D (NCHW) ValueInfoProto with fine-grained control over each dim.

    - batch (idx 0): dim_param if given, else left with neither dim_value nor
      dim_param set (a genuinely unspecified dim).
    - channels (idx 1): always a concrete 3.
    - height (idx 2): a concrete dim_value when given, else left unspecified.
    - width (idx 3): dim_param if given, else left unspecified.
    """
    value_info = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [None, 3, None, None])
    dims = value_info.type.tensor_type.shape.dim
    if batch_dim_param:
        dims[0].dim_param = batch_dim_param
    if height is not None:
        dims[2].dim_value = height
    if width_dim_param:
        dims[3].dim_param = width_dim_param
    return value_info


def _build_model_with_input(value_info: onnx.ValueInfoProto) -> onnx.ModelProto:
    node = onnx.helper.make_node("Identity", inputs=[value_info.name], outputs=["y"])
    output = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)
    graph = onnx.helper.make_graph([node], "test_graph", [value_info], [output])
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])


def test_load_character_dict_prefers_config_json_when_present(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"PostProcess": {"character_dict": ["a", "b"]}}))
    (tmp_path / "inference.yml").write_text("PostProcess:\n  character_dict:\n  - x\n  - y\n")

    assert export_hf_models.load_character_dict(tmp_path) == ["a", "b"]


def test_load_character_dict_falls_back_to_inference_yml_for_v6(tmp_path: Path) -> None:
    (tmp_path / "inference.yml").write_text("PostProcess:\n  name: CTCLabelDecode\n  character_dict:\n  - '!'\n  - a\n")

    assert export_hf_models.load_character_dict(tmp_path) == ["!", "a"]


def test_load_character_dict_returns_none_without_a_dict(tmp_path: Path) -> None:
    (tmp_path / "inference.yml").write_text("PostProcess:\n  name: DBPostProcess\n")

    assert export_hf_models.load_character_dict(tmp_path) is None


def test_load_character_dict_returns_none_when_no_metadata_files(tmp_path: Path) -> None:
    assert export_hf_models.load_character_dict(tmp_path) is None


def test_make_inputs_dynamic_preserves_concrete_height_for_recognizer_shape(tmp_path: Path) -> None:
    """Regression test for the v6 rec height-clobbering bug.

    Mirrors the real PP-OCRv6 recognizer input: batch unspecified, a concrete
    height of 48, and a width already named dynamic by the paddle2onnx C++
    core. Before the fix, the buggy `if not dims[idx].dim_param` guard treated
    the concrete height (empty dim_param, but a set dim_value) as "not yet
    dynamic" and overwrote it with the literal token "?", destroying the 48
    and making the model unresolvable under tract-onnx.
    """
    value_info = _make_4d_input(
        "x",
        batch_dim_param="",
        height=48,
        width_dim_param="DynamicDimension.1",
    )
    model = _build_model_with_input(value_info)
    model_path = tmp_path / "rec.onnx"
    onnx.save(model, str(model_path))

    export_hf_models.make_inputs_dynamic(model_path)

    dims = onnx.load(str(model_path)).graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param == "N"
    assert dims[1].dim_value == 3
    assert dims[2].dim_value == 48
    assert dims[2].dim_param == ""
    assert dims[3].dim_param == "DynamicDimension.1"


def test_make_inputs_dynamic_leaves_fully_concrete_classifier_shape_unchanged(tmp_path: Path) -> None:
    """A classifier with a fixed input resolution keeps its concrete H/W.

    Only the unspecified batch dim should become dynamic.
    """
    value_info = _make_4d_input("x", batch_dim_param="", height=224, width_dim_param="")
    dims = value_info.type.tensor_type.shape.dim
    dims[3].dim_value = 224  # concrete width too, unlike the default helper leaves unset
    model = _build_model_with_input(value_info)
    model_path = tmp_path / "cls.onnx"
    onnx.save(model, str(model_path))

    export_hf_models.make_inputs_dynamic(model_path)

    dims = onnx.load(str(model_path)).graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param == "N"
    assert dims[2].dim_value == 224
    assert dims[3].dim_value == 224


def test_make_inputs_dynamic_is_a_noop_when_already_fully_dynamic(tmp_path: Path) -> None:
    """A detector shape already named by the C++ core is left untouched."""
    value_info = _make_4d_input(
        "x",
        batch_dim_param="DynamicDimension.0",
        height=None,
        width_dim_param="DynamicDimension.2",
    )
    dims = value_info.type.tensor_type.shape.dim
    dims[2].dim_param = "DynamicDimension.1"
    model = _build_model_with_input(value_info)
    model_path = tmp_path / "det.onnx"
    onnx.save(model, str(model_path))

    export_hf_models.make_inputs_dynamic(model_path)

    dims = onnx.load(str(model_path)).graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param == "DynamicDimension.0"
    assert dims[2].dim_param == "DynamicDimension.1"
    assert dims[3].dim_param == "DynamicDimension.2"


def _tract_check_binary_available() -> bool:
    return export_hf_models._TRACT_CHECK_BIN.exists()


@pytest.mark.skipif(not _tract_check_binary_available(), reason="tract_check helper binary not built")
def test_validate_with_tract_accepts_named_symbolic_dims(tmp_path: Path) -> None:
    """A model exported through the fixed make_inputs_dynamic loads under tract."""
    value_info = _make_4d_input(
        "x",
        batch_dim_param="",
        height=48,
        width_dim_param="DynamicDimension.1",
    )
    model = _build_model_with_input(value_info)
    model_path = tmp_path / "rec_fixed.onnx"
    onnx.save(model, str(model_path))

    export_hf_models.make_inputs_dynamic(model_path)

    export_hf_models.validate_with_tract(model_path)  # must not raise


@pytest.mark.skipif(not _tract_check_binary_available(), reason="tract_check helper binary not built")
def test_validate_with_tract_rejects_the_question_mark_token(tmp_path: Path) -> None:
    """The pre-fix "?" dim_param token is exactly what tract cannot resolve."""
    value_info = _make_4d_input("x", batch_dim_param="DynamicDimension.0", height=None, width_dim_param="")
    dims = value_info.type.tensor_type.shape.dim
    dims[2].dim_param = "?"
    dims[3].dim_param = "DynamicDimension.1"
    model = _build_model_with_input(value_info)
    model_path = tmp_path / "rec_broken.onnx"
    onnx.save(model, str(model_path))

    with pytest.raises(RuntimeError, match="tract-onnx failed to load"):
        export_hf_models.validate_with_tract(model_path)


@pytest.mark.skipif(not _tract_check_binary_available(), reason="tract_check helper binary not built")
def test_validate_with_tract_shape_argument_is_passed_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`validate_with_tract` forwards `shape` to the subprocess as `--shape N,C,H,W`."""
    captured_commands: list[list[str]] = []
    real_run = export_hf_models.subprocess.run

    def spy_run(command, **kwargs):
        captured_commands.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(export_hf_models.subprocess, "run", spy_run)

    value_info = _make_4d_input("x", batch_dim_param="", height=None, width_dim_param="")
    model = _build_model_with_input(value_info)
    model_path = tmp_path / "shaped.onnx"
    onnx.save(model, str(model_path))

    export_hf_models.validate_with_tract(model_path, shape=(1, 3, 960, 960))

    assert captured_commands[-1][-2:] == ["--shape", "1,3,960,960"]


def _real_model_path(*relative_parts: str) -> Path | None:
    """Locate a real published PaddleOCR ONNX model in the local HF hub cache, if present."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--xberg-io--paddleocr-onnx-models"
    if not cache_root.is_dir():
        return None
    for snapshot_dir in (cache_root / "snapshots").glob("*"):
        candidate = snapshot_dir.joinpath(*relative_parts)
        if candidate.is_file():
            return candidate
    return None


_REAL_DET_MODEL = _real_model_path("v6", "det", "medium", "model.onnx")


@pytest.mark.skipif(_REAL_DET_MODEL is None, reason="published v6/det/medium model not in local HF cache")
def test_real_det_model_needs_a_pin_to_load_under_tract() -> None:
    """The published PP-OCRv6 medium detector reproduces the documented FPN unify failure.

    This is the exact defect `ModelConfig.tract_validation_shape` exists to
    accommodate: the model is legitimately pin-only, not broken.
    """
    with pytest.raises(RuntimeError, match="tract-onnx failed to load"):
        export_hf_models.validate_with_tract(_REAL_DET_MODEL)

    export_hf_models.validate_with_tract(_REAL_DET_MODEL, shape=(1, 3, 960, 960))  # must not raise


@pytest.mark.parametrize(
    "model_name",
    ["PP-OCRv5_server_det", "PP-OCRv5_mobile_det", "PP-OCRv6_medium_det", "PP-OCRv6_small_det", "PP-OCRv6_tiny_det"],
)
def test_detection_models_are_configured_with_a_tract_validation_shape(model_name: str) -> None:
    assert export_hf_models.MODELS[model_name].tract_validation_shape is not None


@pytest.mark.parametrize(
    "model_name",
    [
        "PP-OCRv5_server_rec",
        "PP-OCRv5_mobile_rec",
        "en_PP-OCRv5_mobile_rec",
        "PP-OCRv6_medium_rec",
        "PP-OCRv6_small_rec",
        "PP-OCRv6_tiny_rec",
        "PP-LCNet_x1_0_doc_ori",
        "PP-LCNet_x1_0_textline_ori",
        "PP-LCNet_x1_0_table_cls",
    ],
)
def test_non_detection_models_stay_on_the_unpinned_tract_check(model_name: str) -> None:
    assert export_hf_models.MODELS[model_name].tract_validation_shape is None
