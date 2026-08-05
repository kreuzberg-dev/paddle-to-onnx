//! Verifies that an ONNX model loads and shape-resolves under `tract-onnx`.
//!
//! Invoked as a subprocess from `tools/export_hf_models.py` so the Python
//! export pipeline can enforce that every exported model is loadable by both
//! ONNX Runtime and the pure-Rust `tract` backend (see `sceptre`'s
//! `TractBackend::load` for the equivalent production load path, which this
//! mirrors: `model_for_path` -> optional `with_input_fact` -> `into_optimized`
//! -> `into_runnable`).
//!
//! Most models (recognizers, classifiers) must load fully unpinned: batch,
//! and where applicable width, stay symbolic. DBNet-style detection models
//! are a documented exception (see `check_model`'s doc comment) and are
//! expected to load only once pinned to a concrete input shape.
//!
//! Usage:
//!   tract_check <path-to-model.onnx>
//!   tract_check <path-to-model.onnx> --shape 1,3,960,960
//!
//! Exit code 0 and `OK` on stdout means the model loaded cleanly. Any
//! failure prints the tract error chain to stderr and exits non-zero.

use std::env;
use std::process::ExitCode;

use tract_onnx::prelude::{DatumType, Framework, InferenceFact, InferenceModelExt, IntoRunnable};

fn main() -> ExitCode {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "tract_check".to_string());
    let Some(model_path) = args.next() else {
        eprintln!("usage: {program} <path-to-model.onnx> [--shape N,C,H,W]");
        return ExitCode::FAILURE;
    };

    let shape = match parse_shape_flag(&mut args) {
        Ok(shape) => shape,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(error) = check_model(&model_path, shape.as_deref()) {
        eprintln!("tract_check failed for {model_path}: {error:#}");
        return ExitCode::FAILURE;
    }

    println!("OK");
    ExitCode::SUCCESS
}

/// Parses an optional trailing `--shape N,C,H,W` flag from the remaining args.
fn parse_shape_flag(args: &mut env::Args) -> Result<Option<Vec<usize>>, String> {
    let Some(flag) = args.next() else {
        return Ok(None);
    };
    if flag != "--shape" {
        return Err(format!("unrecognized argument: {flag}"));
    }
    let value = args
        .next()
        .ok_or_else(|| "--shape requires a value, e.g. --shape 1,3,960,960".to_string())?;
    let shape = value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid --shape value {value:?}: {error}"))
        })
        .collect::<Result<Vec<usize>, String>>()?;
    Ok(Some(shape))
}

/// Loads `model_path` under tract-onnx, optionally pinning input 0 to `shape` first.
///
/// DBNet-style detection models (PP-OCRv5/v6 det) cannot shape-infer with a
/// symbolic H/W: the FPN's `Resize`-upsampled branch and its skip-connection
/// branch each compute the output extent via a different `Div`/`MulInt`
/// chain over the symbolic dimension, and tract's unifier cannot prove those
/// two symbolic expressions equal (only concrete values happen to coincide
/// after the floor-division rounding on both paths). This was confirmed
/// against the published `v6/det/medium` model — it fails during
/// `into_optimized()` at the FPN concat with:
///
///   Impossible to unify MulInt(8, Broadcast([Div(Add([Val(1), ...
///
/// and loads cleanly once pinned to a concrete canvas (verified at both
/// 640x640 and 960x960). This is not a bug in the ONNX export to fix; the
/// consuming pipeline pins detection models to a fixed square canvas for
/// exactly this reason, so `--shape` is the *expected* way to validate them.
/// Do not remove this parameter to "simplify" the CLI — recognizers and
/// classifiers should still be validated unpinned (see
/// `ModelConfig.tract_validation_shape` in `export_hf_models.py`).
fn check_model(model_path: &str, shape: Option<&[usize]>) -> tract_onnx::prelude::TractResult<()> {
    let mut model = tract_onnx::onnx().model_for_path(model_path)?;
    if let Some(shape) = shape {
        let fact = InferenceFact::dt_shape(DatumType::F32, shape);
        model = model.with_input_fact(0, fact)?;
    }
    let model = model.into_optimized()?.into_runnable()?;
    // Loading a runnable plan proves every input dimension either resolved
    // to a concrete value or a named symbol tract can reason about. We
    // deliberately do not run inference; ONNX Runtime already covers
    // numerical correctness (see validate_onnx in export_hf_models.py).
    drop(model);
    Ok(())
}
