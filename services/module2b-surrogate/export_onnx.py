"""export_onnx.py — Export FNO3D checkpoint to ONNX and benchmark CPU inference.

Usage:
    python export_onnx.py --ckpt /path/to/best_model.pt --out fno3d.onnx [--benchmark]

The exported ONNX model takes a fixed (1, 7, 128, 128, 32) float32 input and
returns (1, 5, 128, 128, 32) — the 5-channel residual prediction.

On Apple Silicon / ARM this serves as a proxy benchmark for OCI ARM A1 deployment.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.model_fno3d import FNO3D


def load_model(ckpt_path: Path, device: str = "cpu") -> FNO3D:
    """Load an FNO3D checkpoint. Supports both raw state_dict and {'model': ...} wrappers."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
        cfg = state.get("config", {})
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
        cfg = state.get("config", {})
    else:
        state_dict = state
        cfg = {}

    model = FNO3D(
        in_channels=cfg.get("in_channels", 7),
        out_channels=cfg.get("out_channels", 5),
        width=cfg.get("width", 32),
        modes=tuple(cfg.get("modes", (16, 16, 8))),
        n_layers=cfg.get("n_layers", 4),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] loaded FNO3D: {n_params/1e6:.2f}M params")
    return model


def export_onnx(model: FNO3D, out_path: Path, opset: int = 17) -> Path | None:
    """Export FNO3D to ONNX. Returns ONNX path if successful, else TorchScript path.

    FNO3D uses torch.fft.rfftn which is NOT supported by the legacy ONNX exporter.
    We try the new dynamo-based exporter first (opset 18+, supports DFT), then fall
    back to TorchScript trace if that fails.
    """
    dummy = torch.randn(1, 7, 128, 128, 32, dtype=torch.float32)

    # Attempt 1: dynamo-based ONNX exporter (supports FFT ops)
    print(f"[info] exporting to ONNX via dynamo (opset {opset})...")
    try:
        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamo=True,
        )
        print(f"[ok] ONNX saved: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
        return out_path
    except Exception as e:
        print(f"[warn] dynamo ONNX export failed: {type(e).__name__}: {str(e)[:200]}")

    # Attempt 2: legacy ONNX exporter (will fail on rfftn, but try for completeness)
    print("[info] trying legacy ONNX exporter...")
    try:
        torch.onnx.export(
            model, dummy, str(out_path),
            opset_version=opset,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        print(f"[ok] ONNX saved (legacy): {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
        return out_path
    except Exception as e:
        print(f"[warn] legacy ONNX export failed: {type(e).__name__}: {str(e)[:200]}")

    # Fallback: TorchScript trace
    print("[fallback] exporting to TorchScript trace...")
    ts_path = out_path.with_suffix(".ts.pt")
    traced = torch.jit.trace(model, dummy, strict=False)
    traced.save(str(ts_path))
    print(f"[ok] TorchScript saved: {ts_path} ({ts_path.stat().st_size/1e6:.1f} MB)")
    return ts_path


def benchmark(model: FNO3D, export_path: Path | None, n_warmup: int = 3, n_runs: int = 10) -> None:
    """Benchmark PyTorch CPU, plus ONNX Runtime or TorchScript if available."""
    dummy = torch.randn(1, 7, 128, 128, 32, dtype=torch.float32)

    # PyTorch CPU
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(dummy)
        pt_ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"[bench] PyTorch CPU (eager): {pt_ms:.1f} ms/inference ({n_runs} runs)")

    if not export_path or not export_path.exists():
        return

    suffix = export_path.suffix
    if suffix == ".onnx":
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
            np_input = dummy.numpy()
            for _ in range(n_warmup):
                _ = sess.run(None, {"input": np_input})
            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = sess.run(None, {"input": np_input})
            ort_ms = (time.perf_counter() - t0) / n_runs * 1000
            print(f"[bench] ONNX Runtime CPU: {ort_ms:.1f} ms/inference ({n_runs} runs)")
            print(f"[bench] speedup vs eager: {pt_ms/ort_ms:.2f}x")
        except Exception as e:
            print(f"[bench] ONNX Runtime failed: {e}")
    else:
        # TorchScript
        try:
            ts_model = torch.jit.load(str(export_path))
            ts_model.eval()
            with torch.no_grad():
                for _ in range(n_warmup):
                    _ = ts_model(dummy)
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    _ = ts_model(dummy)
                ts_ms = (time.perf_counter() - t0) / n_runs * 1000
            print(f"[bench] TorchScript CPU: {ts_ms:.1f} ms/inference ({n_runs} runs)")
            print(f"[bench] speedup vs eager: {pt_ms/ts_ms:.2f}x")
        except Exception as e:
            print(f"[bench] TorchScript failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, help="Path to FNO3D checkpoint (.pt). If omitted, uses random weights for benchmarking.")
    parser.add_argument("--out", type=Path, default=Path("fno3d.onnx"), help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark after export")
    parser.add_argument("--skip-export", action="store_true", help="Only benchmark (requires existing ONNX)")
    args = parser.parse_args()

    if args.ckpt:
        model = load_model(args.ckpt)
    else:
        print("[info] no checkpoint provided — using random weights (for benchmark timing only)")
        model = FNO3D()
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[info] FNO3D: {n_params/1e6:.2f}M params")

    export_path: Path | None = args.out
    if not args.skip_export:
        export_path = export_onnx(model, args.out, opset=args.opset)

    if args.benchmark:
        benchmark(model, export_path)


if __name__ == "__main__":
    main()
