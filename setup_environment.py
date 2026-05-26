#!/usr/bin/env python3
"""Platform-aware environment setup for RAPiDock-Reloaded.

Detects your OS and GPU, writes a tailored newdock_env.yaml, then
optionally creates the conda environment and installs pip dependencies.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


def detect_platform():
    """Detect OS, architecture, and GPU capability."""
    system = platform.system()
    machine = platform.machine()

    info = {
        'os': system,
        'arch': machine,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
        # pytorch is always the base conda channel; cuda_channel is added only for CUDA
        'cuda_channel': None,
        # pip package name for torch
        'torch_pip': 'torch',
    }

    if system == 'Darwin':  # macOS — MPS via plain pytorch channel
        info['device'] = 'mps'

    elif system == 'Linux':
        # Prefer CUDA, fall back to ROCm, fall back to CPU
        if _cmd_ok(['nvidia-smi']):
            info['device'] = 'cuda'
            info['cuda_channel'] = 'pytorch-cuda=12.1'
        elif _cmd_ok(['rocm-smi']):
            info['device'] = 'rocm'
            info['torch_pip'] = 'torch-rocm'
        else:
            info['device'] = 'cpu'

    elif system == 'Windows':
        if _cmd_ok(['nvidia-smi']):
            info['device'] = 'cuda'
            info['cuda_channel'] = 'pytorch-cuda=12.1'
        else:
            info['device'] = 'cpu'

    else:
        info['device'] = 'cpu'

    return info


def _cmd_ok(cmd):
    """Return True if a command exits without error."""
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def generate_env_yaml(info):
    """Build the conda environment YAML string for the detected platform."""
    # Channels: pytorch is always first; add pytorch-cuda only when needed
    channels = ['pytorch']
    if info['cuda_channel']:
        channels.append(info['cuda_channel'])
    channels += ['pyg', 'conda-forge']

    lines = ['name: newDock', '', 'channels:']
    for ch in channels:
        lines.append(f'  - {ch}')

    lines += [
        '',
        'dependencies:',
        '  - python=3.10',
        '  - pip',
        '  - nomkl',
        '  - numpy',
        '  - scipy',
        '  - pandas',
        '  - scikit-learn',
        '  - pyyaml',
        '  - tqdm',
        '  - requests',
        '  - jinja2',
        '  - setuptools',
        '  - wheel',
        '  - joblib',
        '  - typing_extensions',
        '  - biopython',
        '  - MDAnalysis',
    ]

    # OpenMP runtime — platform-specific:
    #   macOS/arm64: llvm-openmp (conda-forge) — Apple ships LLVM; _openmp_mutex is Linux-only
    #   Linux: _openmp_mutex=*=*_omp — metapackage that selects gomp vs llvm-omp
    #   Windows: bundled with MSVC runtime, no extra package needed
    if info['os'] == 'Darwin':
        lines.append('  - llvm-openmp')
    elif info['os'] == 'Linux':
        lines.append('  - _openmp_mutex=*=*_omp')

    lines += [
        '',
        '  - pip:',
        f"      - {info['torch_pip']}",
        '      - torchvision',
        '      - torchaudio',
        '      - torch_geometric',
    ]

    return '\n'.join(lines) + '\n'


def print_summary(info):
    """Print detected configuration."""
    print('\n' + '=' * 60)
    print('RAPiDock-Reloaded  —  Environment Detection')
    print('=' * 60)
    print(f"  OS:              {info['os']}")
    print(f"  Architecture:    {info['arch']}")
    print(f"  Compute Device:  {info['device'].upper()}")
    if info['cuda_channel']:
        print(f"  CUDA Channel:    {info['cuda_channel']}")
    print(f"  Python Version:  {info['python_version']}")
    print('=' * 60 + '\n')


def write_env_file(yaml_content, filename='newdock_env.yaml'):
    """Write generated YAML to newdock_env.yaml."""
    with open(filename, 'w') as f:
        f.write(yaml_content)
    print(f'✓ Written {filename}')


def create_conda_env(env_file='newdock_env.yaml', env_name='newDock'):
    """Create (or update) the conda environment from the YAML file."""
    if not Path(env_file).exists():
        print(f'ERROR: Environment file not found: {env_file}')
        sys.exit(1)

    print(f"\nCreating conda environment '{env_name}' from {env_file} …\n")
    try:
        subprocess.run(
            ['conda', 'env', 'create', '-f', env_file, '-n', env_name, '--yes'],
            check=True,
        )
        print(f"\n✓ Environment '{env_name}' created.\n")
        print(f'  conda activate {env_name}\n')
    except subprocess.CalledProcessError as e:
        print(f'ERROR: conda env create failed: {e}')
        sys.exit(1)
    except FileNotFoundError:
        print('ERROR: conda not found. Install Miniconda or Anaconda first.')
        sys.exit(1)


def install_pip_dependencies(env_name='newDock'):
    """Install requirement.txt inside the conda environment."""
    req_file = 'requirement.txt'
    if not Path(req_file).exists():
        print(f'Warning: {req_file} not found — skipping pip step.')
        return

    print(f"Installing pip dependencies into '{env_name}' …\n")
    try:
        subprocess.run(
            f'conda run -n {env_name} pip install --no-build-isolation -r {req_file}',
            shell=True,
            check=True,
        )
        print('\n✓ Pip dependencies installed.\n')
    except subprocess.CalledProcessError as e:
        print(f'ERROR: pip install failed: {e}')
        sys.exit(1)


def main():
    info = detect_platform()
    print_summary(info)

    yaml_content = generate_env_yaml(info)
    write_env_file(yaml_content)

    print(f"\n  Device:   {info['device'].upper()}")
    if info['cuda_channel']:
        print(f"  Channel:  {info['cuda_channel']}")
    print()

    answer = input('Proceed to create the conda environment? (y/n): ').strip().lower()
    if answer != 'y':
        print('\nStopped after generating newdock_env.yaml.')
        print('Run manually:\n')
        print('  conda env create -f newdock_env.yaml -n newDock')
        print('  conda activate newDock')
        print('  pip install --no-build-isolation -r requirement.txt')
        sys.exit(0)

    create_conda_env()
    install_pip_dependencies()

    print('=' * 60)
    print('Setup Complete!')
    print('=' * 60)
    print('\nNext steps:')
    print('1. Activate the environment:')
    print('     conda activate newDock')
    print('\n2. Download pre-trained models from:')
    print('     https://zenodo.org/records/14193621/')
    print('   Place them in: train_models/CGTensorProductEquivariantModel/')
    print('\n3. Run inference:')
    print('     python inference.py --complex_name <name> ...')
    if info['os'] == 'Darwin':
        print('   (macOS: KMP_DUPLICATE_LIB_OK and MPS_FALLBACK are set automatically)')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
