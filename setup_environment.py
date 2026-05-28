#!/usr/bin/env python3
"""RAPiDock-Reloaded — platform-aware environment setup.

Detects your OS and GPU, then creates the 'rapidock' conda environment and
installs the correct PyTorch backend + PyG wheels.

Supported configurations
------------------------
  Linux  + NVIDIA GPU  (CUDA 12.8)   → torch==2.7.0+cu128, PyG cu128 wheels
  Linux  + AMD GPU     (ROCm 6.3)    → torch==2.7.0+rocm6.3, PyG CPU wheels
  Linux  + Intel GPU   (XPU/SYCL)   → torch==2.7.0 + intel-extension-for-pytorch
  Linux  + CPU only                  → torch (CPU), PyG CPU wheels
  macOS  Apple Silicon (MPS)         → torch (MPS), PyG CPU wheels
  macOS  Intel x86_64  (CPU)         → torch (CPU), PyG CPU wheels

Note: If you are using HybriDock-Pep, run scripts/setup_environment.py from
the repo root instead — it sets up both rapidock and score-env in one step.

Usage
-----
  python3 setup_environment.py             # auto-detect, interactive
  python3 setup_environment.py --dry-run   # print commands only
  python3 setup_environment.py --backend rocm   # force AMD ROCm
  python3 setup_environment.py --env-name myenv # override env name (default: rapidock)
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _cmd_ok(*cmd: str) -> bool:
    """Return True if command exits 0 within 5 s."""
    try:
        subprocess.run(list(cmd), capture_output=True, check=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _nvidia_present() -> bool:
    return shutil.which("nvidia-smi") is not None and _cmd_ok("nvidia-smi")


def _nvidia_cc() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip().splitlines()[0].strip()
    except Exception:
        return ""


def _amd_present() -> bool:
    if shutil.which("rocminfo") is not None and _cmd_ok("rocminfo"):
        return True
    if shutil.which("rocm-smi") is not None:
        return True
    return Path("/dev/kfd").exists()


def _intel_xpu_present() -> bool:
    if shutil.which("sycl-ls") is not None and _cmd_ok("sycl-ls"):
        return True
    drm = Path("/sys/class/drm")
    if drm.exists():
        for card in drm.iterdir():
            try:
                if (card / "device" / "vendor").read_text().strip() == "0x8086":
                    return True
            except OSError:
                continue
    return False


# ---------------------------------------------------------------------------
# Platform + backend configuration
# ---------------------------------------------------------------------------

def detect(force_backend: str | None = None) -> dict:
    """Detect OS + GPU and return a config dict."""
    os_name = platform.system()
    arch = platform.machine()
    cfg: dict = {"os": os_name, "arch": arch, "ipex": False}

    # macOS
    if os_name == "Darwin":
        cfg.update(
            backend="mps" if arch == "arm64" else "cpu",
            gpu_label="Apple Silicon MPS" if arch == "arm64" else "CPU (macOS Intel)",
            torch_index="",
            torch_pkg="torch",
            pyg_find="",
        )
        return cfg

    # Windows — CUDA or CPU only (ROCm on Windows is unsupported)
    if os_name == "Windows":
        if force_backend == "cuda" or (force_backend is None and _nvidia_present()):
            cfg.update(
                backend="cuda",
                gpu_label="NVIDIA GPU (CUDA 12.8)",
                torch_index="https://download.pytorch.org/whl/cu128",
                torch_pkg="torch==2.7.0",
                pyg_find="https://data.pyg.org/whl/torch-2.7.0+cu128.html",
            )
        else:
            cfg.update(
                backend="cpu",
                gpu_label="CPU",
                torch_index="",
                torch_pkg="torch",
                pyg_find="",
            )
        return cfg

    # Linux
    backend = force_backend or (
        "cuda" if _nvidia_present()
        else "rocm" if _amd_present()
        else "xpu" if _intel_xpu_present()
        else "cpu"
    )

    if backend == "cuda":
        cc = _nvidia_cc()
        cfg.update(
            backend="cuda",
            gpu_label=f"NVIDIA GPU — CUDA 12.8 (CC {cc})" if cc else "NVIDIA GPU — CUDA 12.8",
            torch_index="https://download.pytorch.org/whl/cu128",
            torch_pkg="torch==2.7.0",
            pyg_find="https://data.pyg.org/whl/torch-2.7.0+cu128.html",
        )
    elif backend == "rocm":
        cfg.update(
            backend="rocm",
            gpu_label="AMD GPU — ROCm 6.3",
            torch_index="https://download.pytorch.org/whl/rocm6.3",
            torch_pkg="torch==2.7.0",
            # No official PyG ROCm wheels: torch-scatter/sparse use CPU builds.
            # Core diffusion model convolutions still run on AMD GPU via ROCm.
            pyg_find="",
        )
    elif backend == "xpu":
        cfg.update(
            backend="xpu",
            gpu_label="Intel GPU — XPU (SYCL/Level-Zero)",
            torch_index="",
            torch_pkg="torch==2.7.0",
            pyg_find="",
            ipex=True,
        )
    else:
        cfg.update(
            backend="cpu",
            gpu_label="CPU (no GPU detected)",
            torch_index="",
            torch_pkg="torch",
            pyg_find="",
        )
    return cfg


# ---------------------------------------------------------------------------
# Conda environment YAML
# ---------------------------------------------------------------------------

def make_env_yaml(cfg: dict, env_name: str) -> str:
    """Generate a conda env YAML for the detected platform."""
    lines = [
        f"name: {env_name}",
        "",
        "# Generated by setup_environment.py — do not edit manually.",
        "# Re-run setup_environment.py to regenerate for your platform.",
        "",
        "channels:",
        "  - conda-forge",
        "  - defaults",
        "",
        "dependencies:",
        "  - python=3.10",
        "  - pip",
        "  - numpy",
        "  - scipy",
        "  - pandas",
        "  - scikit-learn",
        "  - pyyaml",
        "  - tqdm",
        "  - biopython",
        "  - fair-esm",
        "  - mdanalysis",
        "  - rdkit",
    ]

    if cfg["os"] == "Darwin":
        lines.append("  - llvm-openmp")
    elif cfg["os"] == "Linux":
        lines.append("  - _openmp_mutex=*=*_omp")

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------

def _run(cmd: list[str], dry_run: bool) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    if not dry_run:
        r = subprocess.run(cmd, env=os.environ.copy())
        if r.returncode != 0:
            print(f"\n[ERROR] Command failed (exit {r.returncode})")
            sys.exit(r.returncode)


def _conda_python(env_name: str) -> str:
    conda = shutil.which("conda") or ""
    if conda:
        base = Path(conda).resolve().parent.parent
        p = base / "envs" / env_name / "bin" / "python3"
        if p.exists():
            return str(p)
    for base in [
        Path.home() / "miniconda3",
        Path.home() / "miniforge3",
        Path.home() / "anaconda3",
        Path("/opt/conda"),
    ]:
        p = base / "envs" / env_name / "bin" / "python3"
        if p.exists():
            return str(p)
    return f"<conda>/envs/{env_name}/bin/python3"


# ---------------------------------------------------------------------------
# Install steps
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent

_BASE_YMLS = {
    "Darwin":  _REPO_ROOT.parent.parent / "envs" / "rapidock-env-macos.yml",
    "Linux":   _REPO_ROOT.parent.parent / "envs" / "rapidock-env.yml",
    "Windows": _REPO_ROOT.parent.parent / "envs" / "rapidock-env.yml",
}


def install(cfg: dict, env_name: str, dry_run: bool) -> None:
    print(f"\n── Creating conda env '{env_name}' [{cfg['gpu_label']}] ──")

    # Prefer the hybridock-pep env yml if available, else generate one
    base_yml = _BASE_YMLS.get(cfg["os"])
    if base_yml and base_yml.exists():
        yml_path = base_yml
    else:
        # Standalone use: generate a minimal env yaml
        yaml_text = make_env_yaml(cfg, env_name)
        yml_path = _REPO_ROOT / "rapidock_env_generated.yml"
        print(f"  Writing {yml_path}")
        if not dry_run:
            yml_path.write_text(yaml_text)
        else:
            print(f"  (dry-run) would write:\n{yaml_text}")

    _run(["conda", "env", "create", "-f", str(yml_path), "-n", env_name, "--yes"], dry_run)

    py = _conda_python(env_name)
    pip = [py, "-m", "pip", "install"]

    # PyTorch
    torch_cmd = [*pip, cfg["torch_pkg"], "torchvision", "torchaudio"]
    if cfg["torch_index"]:
        torch_cmd += ["--index-url", cfg["torch_index"]]
    _run(torch_cmd, dry_run)

    # Intel IPEX
    if cfg.get("ipex"):
        _run([*pip, "intel-extension-for-pytorch"], dry_run)

    # PyG scatter / sparse / cluster
    pyg = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"]
    if cfg["pyg_find"]:
        _run([*pip, *pyg, "-f", cfg["pyg_find"]], dry_run)
    else:
        _run([*pip, *pyg], dry_run)

    print(f"\n  ✓ '{env_name}' ready.\n")


def print_next_steps(cfg: dict, env_name: str) -> None:
    backend = cfg["backend"]
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\n  Device: {cfg['gpu_label']}\n")
    print("Next steps:")
    print(f"  1. conda activate {env_name}")
    print("  2. Download model weights from https://zenodo.org/records/14193621/")
    print("     → train_models/CGTensorProductEquivariantModel/rapidock_local.pt")
    print("  3. python inference.py --help")

    if backend == "rocm":
        print("""
  AMD ROCm note:
    torch-scatter/torch-sparse run on CPU (no ROCm wheels).
    Core diffusion model ops are GPU-accelerated via ROCm.
    Build torch-scatter from source for full GPU coverage:
      pip install torch-scatter --no-binary :all:
""")
    if backend == "xpu":
        print("""
  Intel XPU note:
    Verify intel-extension-for-pytorch:
      python3 -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
    If ipex import fails, inference falls back to CPU automatically.
""")
    if backend == "mps":
        print("""
  macOS MPS note:
    PYTORCH_ENABLE_MPS_FALLBACK=1 and KMP_DUPLICATE_LIB_OK=TRUE are set
    automatically by inference.py before torch is imported.
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAPiDock-Reloaded environment setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--backend", choices=["cuda", "rocm", "xpu", "mps", "cpu"],
                        help="Force a compute backend (overrides auto-detection)")
    parser.add_argument("--env-name", default="rapidock",
                        help="Conda environment name (default: rapidock)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    if shutil.which("conda") is None:
        print("ERROR: conda not found. Install Miniconda/Miniforge first:")
        print("  https://github.com/conda-forge/miniforge/releases")
        sys.exit(1)

    cfg = detect(force_backend=args.backend)

    print("\n" + "=" * 60)
    print("RAPiDock-Reloaded  —  Environment Setup")
    print("=" * 60)
    print(f"  OS:      {cfg['os']} {cfg['arch']}")
    print(f"  GPU:     {cfg['gpu_label']}")
    print(f"  PyTorch: {cfg['torch_pkg']}")
    if cfg["torch_index"]:
        print(f"  Index:   {cfg['torch_index']}")
    print(f"  Env:     {args.env_name}")
    print("=" * 60)

    if args.dry_run:
        print("\n  ── DRY RUN — no commands will be executed ──\n")
    elif not args.yes:
        answer = input("\nProceed? (y/n): ").strip().lower()
        if answer != "y":
            print("Aborted. Run with --yes to skip this prompt.")
            sys.exit(0)

    install(cfg, env_name=args.env_name, dry_run=args.dry_run)
    print_next_steps(cfg, env_name=args.env_name)


if __name__ == "__main__":
    main()
